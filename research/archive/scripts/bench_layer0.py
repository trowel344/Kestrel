"""Single Qwen3.6-35B-A3B layer benchmark with real weights."""
import torch
import torch.nn as nn
import time
import os
from safetensors import safe_open
from dataclasses import dataclass
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)

torch.set_default_dtype(torch.bfloat16)


def load_shard(snap_dir, shard_name, keys):
    path = os.path.join(snap_dir, shard_name)
    result = {}
    with safe_open(path, framework="pt", device="cpu") as sf:
        for key in keys:
            if key in sf.keys():
                result[key] = sf.get_tensor(key)
    return result


def set_param(module, attr_path, weight):
    parts = attr_path.split('.')
    obj = module
    for part in parts[:-1]:
        obj = getattr(obj, part)
    getattr(obj, parts[-1]).data = weight


class LayerBench:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.layer = None
        self.embedding = None
        self.rotary = None
        self.config = None

    def load(self):
        model_name = "Qwen/Qwen3.6-35B-A3B"
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        snap = os.path.join(cache_dir, 'models--Qwen--Qwen3.6-35B-A3B', 'snapshots',
                            '995ad96eacd98c81ed38be0c5b274b04031597b0')

        full_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = full_config.text_config

        keys_s1 = [
            'model.language_model.embed_tokens.weight',
            'model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
            'model.language_model.layers.0.linear_attn.in_proj_z.weight',
            'model.language_model.layers.0.linear_attn.out_proj.weight',
            'model.language_model.layers.0.mlp.experts.gate_up_proj',
        ]
        keys_s2 = [
            'model.language_model.layers.0.input_layernorm.weight',
            'model.language_model.layers.0.linear_attn.A_log',
            'model.language_model.layers.0.linear_attn.conv1d.weight',
            'model.language_model.layers.0.linear_attn.dt_bias',
            'model.language_model.layers.0.linear_attn.in_proj_a.weight',
            'model.language_model.layers.0.linear_attn.in_proj_b.weight',
            'model.language_model.layers.0.linear_attn.norm.weight',
            'model.language_model.layers.0.linear_attn.norm.bias',
            'model.language_model.layers.0.mlp.experts.down_proj',
            'model.language_model.layers.0.mlp.gate.weight',
            'model.language_model.layers.0.mlp.shared_expert.down_proj.weight',
            'model.language_model.layers.0.mlp.shared_expert.gate_proj.weight',
            'model.language_model.layers.0.mlp.shared_expert.up_proj.weight',
            'model.language_model.layers.0.mlp.shared_expert_gate.weight',
            'model.language_model.layers.0.post_attention_layernorm.weight',
        ]

        w = {}
        w.update(load_shard(snap, 'model-00001-of-00026.safetensors', keys_s1))
        w.update(load_shard(snap, 'model-00002-of-00026.safetensors', keys_s2))

        emb_weight = w['model.language_model.embed_tokens.weight']
        self.embedding = nn.Embedding(emb_weight.shape[0], emb_weight.shape[1],
                                       _weight=emb_weight).to(self.device)

        self.layer = Qwen3_5MoeDecoderLayer(self.config, 0).to(self.device)
        self.layer.eval()

        # Load weights
        prefix = 'model.language_model.layers.0'
        mapping = {
            'input_layernorm.weight': f'{prefix}.input_layernorm.weight',
            'linear_attn.in_proj_qkv.weight': f'{prefix}.linear_attn.in_proj_qkv.weight',
            'linear_attn.in_proj_z.weight': f'{prefix}.linear_attn.in_proj_z.weight',
            'linear_attn.out_proj.weight': f'{prefix}.linear_attn.out_proj.weight',
            'linear_attn.in_proj_a.weight': f'{prefix}.linear_attn.in_proj_a.weight',
            'linear_attn.in_proj_b.weight': f'{prefix}.linear_attn.in_proj_b.weight',
            'linear_attn.conv1d.weight': f'{prefix}.linear_attn.conv1d.weight',
            'linear_attn.A_log': f'{prefix}.linear_attn.A_log',
            'linear_attn.dt_bias': f'{prefix}.linear_attn.dt_bias',
            'linear_attn.norm.weight': f'{prefix}.linear_attn.norm.weight',
            'post_attention_layernorm.weight': f'{prefix}.post_attention_layernorm.weight',
            'mlp.gate.weight': f'{prefix}.mlp.gate.weight',
            'mlp.shared_expert_gate.weight': f'{prefix}.mlp.shared_expert_gate.weight',
            'mlp.shared_expert.gate_proj.weight': f'{prefix}.mlp.shared_expert.gate_proj.weight',
            'mlp.shared_expert.up_proj.weight': f'{prefix}.mlp.shared_expert.up_proj.weight',
            'mlp.shared_expert.down_proj.weight': f'{prefix}.mlp.shared_expert.down_proj.weight',
            'mlp.experts.gate_up_proj': f'{prefix}.mlp.experts.gate_up_proj',
            'mlp.experts.down_proj': f'{prefix}.mlp.experts.down_proj',
        }
        for attr_path, weight_key in mapping.items():
            set_param(self.layer, attr_path, w[weight_key].to(self.device))

        if 'linear_attn.norm.bias' in w.get('linear_attn.norm.bias', {}):
            set_param(self.layer, 'linear_attn.norm.bias', w['linear_attn.norm.bias'].to(self.device))

        if hasattr(self.layer.linear_attn.norm, 'bias') and \
                f'{prefix}.linear_attn.norm.bias' in w:
            set_param(self.layer, 'linear_attn.norm.bias',
                      w[f'{prefix}.linear_attn.norm.bias'].to(self.device))

        self.rotary = Qwen3_5MoeTextRotaryEmbedding(self.config).to(self.device)

        expert_params = w[f'{prefix}.mlp.experts.gate_up_proj'].numel() + \
                        w[f'{prefix}.mlp.experts.down_proj'].numel()
        print(f"Layer: {sum(p.numel() for p in self.layer.parameters()):,} params")
        print(f"  Experts: {expert_params:,} ({expert_params*2/1024**3:.2f} GB)")
        print(f"  Embedding: {emb_weight.numel():,} params")

    @torch.inference_mode()
    def benchmark_prefill(self, seq_len):
        hidden = torch.randn(1, seq_len, self.config.hidden_size, device=self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        pos_emb = self.rotary(hidden, position_ids)
        torch.cuda.synchronize()

        for _ in range(3):
            self.layer(hidden, position_embeddings=pos_emb)
            torch.cuda.synchronize()

        n = 10
        start = time.perf_counter()
        for _ in range(n):
            self.layer(hidden, position_embeddings=pos_emb)
            torch.cuda.synchronize()
        avg = (time.perf_counter() - start) * 1000 / n
        return {"seq_len": seq_len, "ms": avg, "tok_s": seq_len / (avg / 1000)}

    @torch.inference_mode()
    def benchmark_decode(self, n_tokens=200):
        hidden = torch.randn(1, 1, self.config.hidden_size, device=self.device)
        position_ids = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        pos_emb = self.rotary(hidden, position_ids)
        torch.cuda.synchronize()

        for _ in range(10):
            self.layer(hidden, position_embeddings=pos_emb)
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_tokens):
            self.layer(hidden, position_embeddings=pos_emb)
            torch.cuda.synchronize()
        avg = (time.perf_counter() - start) * 1000 / n_tokens
        return {"ms": avg, "tok_s": 1000 / avg}

    def benchmark_pcie(self):
        gate_up = self.layer.mlp.experts.gate_up_proj.data.cpu()
        down = self.layer.mlp.experts.down_proj.data.cpu()
        total_bytes = gate_up.numel() * gate_up.element_size() + \
                      down.numel() * down.element_size()
        n = 10

        total_in = 0
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = gate_up.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
            total_in += (time.perf_counter() - t0) * 1000

        total_out = 0
        gpu_t = gate_up.to(self.device)
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _cpu = gpu_t.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            total_out += (time.perf_counter() - t0) * 1000
            del _cpu

        avg_in = total_in / n
        avg_out = total_out / n
        return {
            "total_mb": total_bytes / 1024 / 1024,
            "in_ms": avg_in,
            "out_ms": avg_out,
            "bw_in": (total_bytes / (avg_in / 1000)) / 1024**3,
            "bw_out": (total_bytes / (avg_out / 1000)) / 1024**3,
        }

    def benchmark_layer_breakdown(self):
        """Benchmark attention vs MoE separately."""
        hidden = torch.randn(1, 1, self.config.hidden_size, device=self.device)
        position_ids = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        pos_emb = self.rotary(hidden, position_ids)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(10):
            self.layer(hidden, position_embeddings=pos_emb)
            torch.cuda.synchronize()

        # Attention only
        n = 100
        start = time.perf_counter()
        for _ in range(n):
            h = self.layer.input_layernorm(hidden)
            self.layer.linear_attn(h)
            torch.cuda.synchronize()
        attn_ms = (time.perf_counter() - start) * 1000 / n

        # MoE only
        h = self.layer.post_attention_layernorm(hidden)
        start = time.perf_counter()
        for _ in range(n):
            self.layer.mlp(h)
            torch.cuda.synchronize()
        moe_ms = (time.perf_counter() - start) * 1000 / n

        return {"attention_ms": attn_ms, "moe_ms": moe_ms}


if __name__ == "__main__":
    bench = LayerBench()
    print("Loading...")
    bench.load()
    print()

    print("=== PCIe ===")
    p = bench.benchmark_pcie()
    print(f"  {p['total_mb']:.0f} MB: in={p['in_ms']:.1f}ms ({p['bw_in']:.1f} GB/s), "
          f"out={p['out_ms']:.1f}ms ({p['bw_out']:.1f} GB/s)")

    print()
    print("=== Layer Breakdown (decode) ===")
    b = bench.benchmark_layer_breakdown()
    print(f"  Attention: {b['attention_ms']:.3f}ms")
    print(f"  MoE:       {b['moe_ms']:.3f}ms")
    print(f"  Total:     {b['attention_ms'] + b['moe_ms']:.3f}ms")

    print()
    print("=== Prefill ===")
    for sl in [128, 512, 2048, 4096]:
        r = bench.benchmark_prefill(sl)
        full = r['tok_s'] / 40
        print(f"  seq={sl:5d}: {r['ms']:.1f}ms/layer ({r['tok_s']:.0f} tok/s/layer, "
              f"{full:.0f} tok/s full)")

    print()
    print("=== Decode ===")
    r = bench.benchmark_decode(200)
    full_decode = r['ms'] * 40
    print(f"  {r['ms']:.3f}ms per layer")
    print(f"  Full model: {full_decode:.1f}ms/tok ({1000/full_decode:.1f} tok/s)")
    print(f"  With 8-token speculation: {full_decode/8:.1f}ms/tok equiv "
          f"({8000/full_decode:.1f} tok/s equiv)")
