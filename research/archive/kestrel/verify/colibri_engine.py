import os
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from typing import Optional
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)
from accelerate import init_empty_weights
from kestrel.backends.nvfp4_loader import NVFP4Loader, dequantize_nvfp4_weight


torch.set_default_dtype(torch.bfloat16)


class StreamingExperts(nn.Module):
    def __init__(self, config, layer_idx, engine_ref):
        super().__init__()
        self.num_experts = config.num_experts
        self.moe_intermediate_size = config.moe_intermediate_size
        self.layer_idx = layer_idx
        self.engine_ref = engine_ref
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states, top_k_index, top_k_weights):
        engine = self.engine_ref()
        unique_experts = torch.unique(top_k_index)
        gate_up, down_proj = engine._load_active_experts_gpu(self.layer_idx, unique_experts)
        N = gate_up.shape[0]
        moe = self.moe_intermediate_size

        if N == 0:
            return torch.zeros_like(hidden_states)

        w_gate = gate_up[:, :moe, :].transpose(1, 2)
        w_up = gate_up[:, moe:, :].transpose(1, 2)
        h_exp = hidden_states.expand(N, -1, -1)
        gate = torch.bmm(h_exp, w_gate)
        up = torch.bmm(h_exp, w_up)
        act = F.silu(gate) * up
        out = torch.bmm(act, down_proj.transpose(1, 2))
        wgts = top_k_weights[0, :N].reshape(N, 1, 1)
        return (out * wgts).sum(dim=0, keepdim=True).to(out.dtype)


class ColibriEngine:
    def __init__(
        self,
        model_name: str = "nvidia/Qwen3.5-122B-A10B-NVFP4",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        full_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = getattr(full_config, "text_config", full_config)
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.layer_types = self.config.layer_types

        self.layers: list[Qwen3_5MoeDecoderLayer] = []
        self.embed_tokens: nn.Embedding | None = None
        self.rotary: Qwen3_5MoeTextRotaryEmbedding | None = None
        self.norm: Qwen3_5MoeRMSNorm | None = None
        self.lm_head: nn.Linear | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        hub_name = model_name.replace("/", "--")
        cache_base = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), f"models--{hub_name}")
        if os.path.exists(os.path.join(cache_base, "model.safetensors.index.json")):
            self.snap_dir = cache_base
        else:
            snaps = os.path.join(cache_base, "snapshots")
            if os.path.exists(snaps):
                versions = sorted(os.listdir(snaps))
                self.snap_dir = os.path.join(snaps, versions[-1]) if versions else cache_base
            else:
                self.snap_dir = cache_base

        self._nvfp4_loader = NVFP4Loader(model_dir=self.snap_dir)
        self.moe_intermediate_size = self.config.moe_intermediate_size

        self._build_model()

        self._transfer_stream = torch.cuda.Stream()
        self._compute_stream = torch.cuda.Stream()
        self._transfer_event = torch.cuda.Event()

    def _find_shard_for_key(self, key: str) -> str | None:
        idx_path = os.path.join(self.snap_dir, "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            return None
        with open(idx_path) as f:
            idx = json.load(f)
        shard = idx["weight_map"].get(key)
        if shard is None:
            return None
        path = os.path.join(self.snap_dir, shard)
        return path if os.path.exists(path) else None

    def _load_tensor(self, key: str, device: str = "cpu") -> torch.Tensor | None:
        shard_path = self._find_shard_for_key(key)
        if shard_path is None:
            return None
        with safe_open(shard_path, framework="pt", device=device) as sf:
            if key not in sf.keys():
                return None
            return sf.get_tensor(key)

    def _load_bf16_tensor_to_cpu(self, key: str) -> torch.Tensor | None:
        t = self._load_tensor(key, device="cpu")
        if t is None:
            alt = key.removeprefix("model.language_model.")
            t = self._load_tensor(alt, device="cpu")
        if t is None:
            return None
        return t.to(dtype=self.dtype, device="cpu", copy=False)

    def _preload_dense_to_cpu(self):
        print("Preloading dense weights to CPU...")
        self._cpu_dense: dict[str, torch.Tensor] = {}
        total_mb = 0
        for i in range(self.num_layers):
            prefix = f"model.language_model.layers.{i}"
            lt = self.layer_types[i]
            keys = [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ]
            if lt == "linear_attention":
                for k in ["in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b"]:
                    keys.append(f"{prefix}.linear_attn.{k}.weight")
                keys += [
                    f"{prefix}.linear_attn.conv1d.weight",
                    f"{prefix}.linear_attn.A_log",
                    f"{prefix}.linear_attn.dt_bias",
                    f"{prefix}.linear_attn.norm.weight",
                ]
                bias_key = f"{prefix}.linear_attn.norm.bias"
                if self._find_shard_for_key(bias_key):
                    keys.append(bias_key)
            else:
                for k in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    keys.append(f"{prefix}.self_attn.{k}.weight")
                for k in ["q_norm", "k_norm"]:
                    keys.append(f"{prefix}.self_attn.{k}.weight")
            keys += [
                f"{prefix}.mlp.gate.weight",
                f"{prefix}.mlp.shared_expert_gate.weight",
                f"{prefix}.mlp.shared_expert.gate_proj.weight",
                f"{prefix}.mlp.shared_expert.up_proj.weight",
                f"{prefix}.mlp.shared_expert.down_proj.weight",
            ]
            for k in keys:
                t = self._load_bf16_tensor_to_cpu(k)
                if t is not None:
                    self._cpu_dense[k] = t.contiguous()
                    total_mb += t.numel() * t.element_size() / 1024**2
        print(f"  CPU dense: {total_mb:.0f} MB ({total_mb/1024:.2f} GB)")

    def _preload_experts_to_gpu(self):
        print("GPU expert cache initialized (LRU, on-demand loading)...")
        self._expert_cache: dict[tuple[int, int], tuple] = {}
        self._expert_cache_keys: list[tuple[int, int]] = []
        experts_per_pass = self.num_layers * self.config.num_experts_per_tok  # 48 * 8 = 384
        self._expert_cache_max = experts_per_pass * 2  # 768 to avoid thrashing

    def _assign_dense_to_layer(self, layer: Qwen3_5MoeDecoderLayer, layer_idx: int):
        prefix = f"model.language_model.layers.{layer_idx}"
        lt = self.layer_types[layer_idx]
        def assign(attr_path, tensor):
            parts = attr_path.split(".")
            obj = layer
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))

        def to_gpu(t):
            return t.cuda(non_blocking=True)

        assign("input_layernorm.weight", to_gpu(self._cpu_dense[f"{prefix}.input_layernorm.weight"]))
        assign("post_attention_layernorm.weight", to_gpu(self._cpu_dense[f"{prefix}.post_attention_layernorm.weight"]))
        if lt == "linear_attention":
            for k in ["in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b"]:
                assign(f"linear_attn.{k}.weight", to_gpu(self._cpu_dense[f"{prefix}.linear_attn.{k}.weight"]))
            assign("linear_attn.conv1d.weight", to_gpu(self._cpu_dense[f"{prefix}.linear_attn.conv1d.weight"]))
            assign("linear_attn.A_log", to_gpu(self._cpu_dense[f"{prefix}.linear_attn.A_log"]))
            assign("linear_attn.dt_bias", to_gpu(self._cpu_dense[f"{prefix}.linear_attn.dt_bias"]))
            assign("linear_attn.norm.weight", to_gpu(self._cpu_dense[f"{prefix}.linear_attn.norm.weight"]))
            bias_key = f"{prefix}.linear_attn.norm.bias"
            if bias_key in self._cpu_dense:
                assign("linear_attn.norm.bias", to_gpu(self._cpu_dense[bias_key]))
        else:
            for k in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                assign(f"self_attn.{k}.weight", to_gpu(self._cpu_dense[f"{prefix}.self_attn.{k}.weight"]))
            for k in ["q_norm", "k_norm"]:
                assign(f"self_attn.{k}.weight", to_gpu(self._cpu_dense[f"{prefix}.self_attn.{k}.weight"]))
        assign("mlp.gate.weight", to_gpu(self._cpu_dense[f"{prefix}.mlp.gate.weight"]))
        assign("mlp.shared_expert_gate.weight", to_gpu(self._cpu_dense[f"{prefix}.mlp.shared_expert_gate.weight"]))
        assign("mlp.shared_expert.gate_proj.weight", to_gpu(self._cpu_dense[f"{prefix}.mlp.shared_expert.gate_proj.weight"]))
        assign("mlp.shared_expert.up_proj.weight", to_gpu(self._cpu_dense[f"{prefix}.mlp.shared_expert.up_proj.weight"]))
        assign("mlp.shared_expert.down_proj.weight", to_gpu(self._cpu_dense[f"{prefix}.mlp.shared_expert.down_proj.weight"]))
        layer.eval()

    def _unassign_dense_from_layer(self, layer_idx: int):
        layer = self.layers[layer_idx]
        for name, _ in list(layer.named_parameters()):
            parts = name.split(".")
            obj = layer
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], nn.Parameter(
                    torch.empty(0, dtype=self.dtype, device="meta"),
                    requires_grad=False,
                ))

    def _load_or_make(self, key, dtype, device):
        w = self._load_tensor(key, device="cpu")
        if w is not None:
            return w.to(dtype=dtype, device=device)
        alt = key.removeprefix("model.language_model.")
        w = self._load_tensor(alt, device="cpu")
        if w is not None:
            return w.to(dtype=dtype, device=device)
        return None

    def _build_model(self):
        with init_empty_weights():
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, dtype=self.dtype)
            self.rotary = Qwen3_5MoeTextRotaryEmbedding(self.config)
            self.norm = Qwen3_5MoeRMSNorm(self.hidden_size, eps=self.config.rms_norm_eps)
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False, dtype=self.dtype)
            for i in range(self.num_layers):
                layer = Qwen3_5MoeDecoderLayer(self.config, i)
                layer.mlp.experts = StreamingExperts(self.config, i, lambda e=self: e)
                self.layers.append(layer)

        def _move_meta(module, device):
            for name, param in list(module.named_parameters(recurse=False)):
                if param.device.type == "meta":
                    new_p = nn.Parameter(
                        torch.empty(param.shape, dtype=param.dtype, device=device),
                        requires_grad=param.requires_grad,
                    )
                    setattr(module, name, new_p)
            for child in module.children():
                _move_meta(child, device)

        _move_meta(self.embed_tokens, self.device)
        _move_meta(self.norm, self.device)
        _move_meta(self.lm_head, self.device)
        self.rotary = self.rotary.to(self.device)

        embed_w = self._load_or_make("model.language_model.embed_tokens.weight", self.dtype, self.device)
        if embed_w is not None:
            self.embed_tokens = nn.Embedding.from_pretrained(embed_w, freeze=True, padding_idx=None)
        norm_w = self._load_or_make("model.language_model.norm.weight", self.dtype, self.device)
        if norm_w is not None:
            self.norm = Qwen3_5MoeRMSNorm(self.hidden_size, eps=self.config.rms_norm_eps)
            self.norm.weight = nn.Parameter(norm_w)
        head_w = self._load_or_make("model.language_model.lm_head.weight", self.dtype, self.device)
        if head_w is not None:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False, dtype=self.dtype)
            self.lm_head.weight = nn.Parameter(head_w)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        embed_gb = self.embed_tokens.weight.numel() * self.embed_tokens.weight.element_size() / 1024**3
        head_gb = self.lm_head.weight.numel() * self.lm_head.weight.element_size() / 1024**3
        print(f"  Embed: {embed_gb:.2f} GB, LM Head: {head_gb:.2f} GB")
        print(f"  VRAM before preload: allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB")

        self._preload_dense_to_cpu()
        self._preload_experts_to_gpu()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  VRAM after preload: allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB")

    def _load_active_experts_gpu(self, layer_idx: int, expert_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eids = expert_indices.tolist()
        gate_ups = []
        downs = []
        for eid in eids:
            key = (layer_idx, eid)
            cached = self._expert_cache.get(key)
            if cached is None:
                cached = self._load_expert_from_disk(layer_idx, eid)
                self._expert_cache[key] = cached
                self._expert_cache_keys.append(key)
                if len(self._expert_cache_keys) > self._expert_cache_max:
                    old_key = self._expert_cache_keys.pop(0)
                    self._expert_cache.pop(old_key, None)
            gp_p, gp_s, gp_s2, up_p, up_s, up_s2, dn_p, dn_s, dn_s2 = cached
            gu = torch.cat([
                dequantize_nvfp4_weight(gp_p, gp_s, gp_s2),
                dequantize_nvfp4_weight(up_p, up_s, up_s2),
            ], dim=0).to(dtype=self.dtype)
            dn = dequantize_nvfp4_weight(dn_p, dn_s, dn_s2).to(dtype=self.dtype)
            gate_ups.append(gu)
            downs.append(dn)
        gu = torch.stack(gate_ups) if gate_ups else torch.empty(0, dtype=self.dtype, device="cuda")
        dn = torch.stack(downs) if downs else torch.empty(0, dtype=self.dtype, device="cuda")
        return gu, dn

    def _load_expert_from_disk(self, layer_idx: int, expert_id: int):
        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}"
        shard_name = self._nvfp4_loader.weight_map.get(f"{prefix}.gate_proj.weight")
        if shard_name is None:
            return None
        shard_path = os.path.join(self.snap_dir, shard_name)
        with safe_open(shard_path, framework="pt", device="cuda") as sf:
            tensors = (
                sf.get_tensor(f"{prefix}.gate_proj.weight"),
                sf.get_tensor(f"{prefix}.gate_proj.weight_scale"),
                sf.get_tensor(f"{prefix}.gate_proj.weight_scale_2"),
                sf.get_tensor(f"{prefix}.up_proj.weight"),
                sf.get_tensor(f"{prefix}.up_proj.weight_scale"),
                sf.get_tensor(f"{prefix}.up_proj.weight_scale_2"),
                sf.get_tensor(f"{prefix}.down_proj.weight"),
                sf.get_tensor(f"{prefix}.down_proj.weight_scale"),
                sf.get_tensor(f"{prefix}.down_proj.weight_scale_2"),
            )
        return tensors

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        torch.cuda.synchronize()

        if not hasattr(self, '_transfer_stream'):
            self._transfer_stream = torch.cuda.Stream()
            self._compute_stream = torch.cuda.Stream()
            self._transfer_event = torch.cuda.Event()

        transfer = self._transfer_stream
        compute = self._compute_stream
        ev = self._transfer_event
        saved_top_k: dict[int, int] = {}

        with torch.cuda.stream(compute):
            if input_ids is not None:
                h = self.embed_tokens(input_ids)
                _, seq_len = input_ids.shape
            else:
                h = inputs_embeds
                _, seq_len = inputs_embeds.shape[:2]
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            pos_emb = self.rotary(h, position_ids)

        with torch.cuda.stream(transfer):
            self._assign_dense_to_layer(self.layers[0], 0)
            ev.record(transfer)

        for i in range(self.num_layers):
            compute.wait_event(ev)

            if i + 1 < self.num_layers:
                with torch.cuda.stream(transfer):
                    self._assign_dense_to_layer(self.layers[i + 1], i + 1)
                    ev.record(transfer)

            with torch.cuda.stream(compute):
                layer = self.layers[i]
                if top_k is not None and top_k != layer.mlp.gate.top_k:
                    saved_top_k[i] = layer.mlp.gate.top_k
                    layer.mlp.gate.top_k = top_k

                out = layer(
                    hidden_states=h,
                    position_embeddings=pos_emb,
                    attention_mask=None,
                    position_ids=position_ids,
                )
                h = out[0] if isinstance(out, tuple) else out

                if i in saved_top_k:
                    layer.mlp.gate.top_k = saved_top_k.pop(i)
                self._unassign_dense_from_layer(i)

        with torch.cuda.stream(compute):
            h = self.norm(h)
            logits = self.lm_head(h)

        torch.cuda.synchronize()
        return logits, None

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        torch.cuda.synchronize()

        logits, _ = self.forward(input_ids=input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token.item()]

        for _ in range(max_new_tokens - 1):
            logits, _ = self.forward(input_ids=next_token)
            if do_sample and temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())

        torch.cuda.synchronize()
        return self.tokenizer.decode(generated, skip_special_tokens=True)
