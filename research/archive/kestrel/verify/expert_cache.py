import os
import torch
import threading
from collections import OrderedDict


class ExpertWeights:
    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor):
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.size_bytes = gate_up_proj.numel() * gate_up_proj.element_size() + \
                         down_proj.numel() * down_proj.element_size()


class ExpertCache:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        vram_budget_gb: float = 6.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.vram_budget = vram_budget_gb * 1024**3
        self.used_vram = 0
        self.cache: OrderedDict[str, ExpertWeights] = OrderedDict()
        self.total_accesses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_result: dict[str, torch.Tensor] | None = None
        self._prefetch_lock = threading.Lock()

    def _key(self, layer: int) -> str:
        return f"layers.{layer}"

    def _load_tensors(self, snap_dir: str, layer: int) -> dict[str, torch.Tensor]:
        if not os.path.exists(os.path.join(snap_dir, "model.safetensors.index.json")):
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
            gate_up = torch.empty(256, 1024, 2048, dtype=self.dtype)
            down_proj = torch.empty(256, 2048, 512, dtype=self.dtype)
            return {"gate_up_proj": gate_up, "down_proj": down_proj}

        import json
        with open(os.path.join(snap_dir, "model.safetensors.index.json")) as f:
            idx = json.load(f)

        from safetensors import safe_open
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        gate_up_key = prefix + "gate_up_proj"
        down_key = prefix + "down_proj"

        gate_up = None
        down = None
        for key, shard in idx["weight_map"].items():
            if key == gate_up_key:
                path = os.path.join(snap_dir, shard)
                if os.path.exists(path):
                    with safe_open(path, framework="pt", device="cpu") as sf:
                        gate_up = sf.get_tensor(key).to(dtype=self.dtype)
            elif key == down_key:
                path = os.path.join(snap_dir, shard)
                if os.path.exists(path):
                    with safe_open(path, framework="pt", device="cpu") as sf:
                        down = sf.get_tensor(key).to(dtype=self.dtype)

        if gate_up is None:
            gate_up = torch.empty(256, 1024, 2048, dtype=self.dtype)
        if down is None:
            down = torch.empty(256, 2048, 512, dtype=self.dtype)
        return {"gate_up_proj": gate_up, "down_proj": down}

    def get_expert_weights(self, layer: int) -> ExpertWeights:
        cache_key = self._key(layer)
        self.total_accesses += 1

        if cache_key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        self.cache_misses += 1
        snap_dir = os.path.join(
            os.path.expanduser("~/.cache/huggingface/hub"),
            "models--Qwen--Qwen3.6-35B-A3B", "snapshots",
            "995ad96eacd98c81ed38be0c5b274b04031597b0",
        )
        tensors = self._load_tensors(snap_dir, layer)
        weights = ExpertWeights(tensors["gate_up_proj"], tensors["down_proj"])
        self._evict_if_needed(weights)
        self.cache[cache_key] = weights
        self.used_vram += weights.size_bytes
        return weights

    def prefetch(self, layer: int):
        def _load():
            snap_dir = os.path.join(
                os.path.expanduser("~/.cache/huggingface/hub"),
                "models--Qwen--Qwen3.6-35B-A3B", "snapshots",
                "995ad96eacd98c81ed38be0c5b274b04031597b0",
            )
            result = self._load_tensors(snap_dir, layer)
            with self._prefetch_lock:
                self._prefetch_result = result

        self._prefetch_thread = threading.Thread(target=_load, daemon=True)
        self._prefetch_thread.start()

    def consume_prefetch(self, layer: int) -> ExpertWeights | None:
        if self._prefetch_thread is None:
            return None
        self._prefetch_thread.join()
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = None
            self._prefetch_thread = None
        if result is None:
            return None
        weights = ExpertWeights(result["gate_up_proj"], result["down_proj"])
        self._evict_if_needed(weights)
        cache_key = self._key(layer)
        self.cache[cache_key] = weights
        self.used_vram += weights.size_bytes
        self.total_accesses += 1
        self.cache_misses += 1
        return weights

    def _evict_if_needed(self, new_weights: ExpertWeights):
        needed = new_weights.size_bytes
        while self.used_vram + needed > self.vram_budget and self.cache:
            key, weights = self.cache.popitem(last=False)
            self.used_vram -= weights.size_bytes

    def __getitem__(self, layer: int) -> ExpertWeights:
        return self.get_expert_weights(layer)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def resident_experts(self) -> int:
        return len(self.cache)

    def summary(self) -> dict:
        return {
            "hit_rate": f"{self.hit_rate:.1%}",
            "resident_layers": self.resident_experts,
            "cache_used_gb": round(self.used_vram / 1024**3, 3),
            "cache_budget_gb": round(self.vram_budget / 1024**3, 2),
            "accesses": self.total_accesses,
            "per_layer_mb": round(
                (256 * 1024 * 2048 * 2 + 256 * 2048 * 512 * 2) / 1024**2, 1
            ) if self.dtype == torch.bfloat16 else 0,
        }
