import torch
import threading
from collections import OrderedDict
from ..backends.nvfp4_loader import NVFP4Loader
from .predictive_cache import PredictiveExpertCache
from .hot_set import HotExpertSet


class TieredExpertWeights:
    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, tier: int = 0):
        self.gate_up = gate_up
        self.down = down
        self.tier = tier
        self.size_bytes = gate_up.numel() * gate_up.element_size() + \
                         down.numel() * down.element_size()

    def to(self, device: str):
        return TieredExpertWeights(
            self.gate_up.to(device),
            self.down.to(device),
            self.tier,
        )


class CacheLoadError(RuntimeError):
    pass


class MultiTierCache:
    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        n_embd: int,
        n_ff: int,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        l1_budget_gb: float = 2.0,
        l2_budget_gb: float = 32.0,
        hot_per_layer: int = 20,
        strict: bool = True,
    ):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_embd = n_embd
        self.n_ff = n_ff
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.dtype = dtype
        self.strict = strict
        self.l1_budget = int(l1_budget_gb * 1024**3)
        self.l2_budget = int(l2_budget_gb * 1024**3)
        self.l1_used = 0
        self.l2_used = 0
        self.l1_cache: OrderedDict[tuple[int, int], TieredExpertWeights] = OrderedDict()
        self.l2_cache: OrderedDict[tuple[int, int], TieredExpertWeights] = OrderedDict()
        self.hot_set = HotExpertSet(n_layers, n_experts, hot_per_layer)
        self._predictive = PredictiveExpertCache(n_layers, n_experts, capacity=hot_per_layer * n_layers)
        self._loader = NVFP4Loader(model_dir) if model_dir else None
        self._prefetch_lock = threading.Lock()
        self._prefetch_queue: list[tuple[int, int]] = []
        self._prefetch_thread: threading.Thread | None = None
        self.accesses = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.load_errors = 0

    def _load_weights(self, layer: int, expert: int) -> TieredExpertWeights | None:
        if self._loader is None:
            return None
        try:
            # L2 is genuinely host-resident. Loading directly onto CUDA here
            # caused large transient allocations and made the old "RAM cache"
            # another VRAM cache.
            gu, dn = self._loader.load_experts(
                layer, [expert], dtype=self.dtype, device="cpu"
            )
            gate_up = gu[0]
            down = dn[0]
        except Exception as exc:
            self.load_errors += 1
            if self.strict:
                raise CacheLoadError(
                    f"Failed to load layer {layer} expert {expert}"
                ) from exc
            return None
        return TieredExpertWeights(gate_up, down, tier=2)

    def _make_l1_room(self, required_bytes: int) -> bool:
        if required_bytes > self.l1_budget:
            return False
        while self.l1_used + required_bytes > self.l1_budget and self.l1_cache:
            _, evicted = self.l1_cache.popitem(last=False)
            self.l1_used -= evicted.size_bytes
        return self.l1_used + required_bytes <= self.l1_budget

    def get(self, layer: int, expert: int) -> TieredExpertWeights | None:
        self.accesses += 1
        key = (layer, expert)
        if key in self.l1_cache:
            self.l1_hits += 1
            self.l1_cache.move_to_end(key)
            return self.l1_cache[key]
        if key in self.l2_cache:
            self.l2_hits += 1
            l2_w = self.l2_cache[key]
            self.l2_cache.move_to_end(key)
            if self._make_l1_room(l2_w.size_bytes):
                w = l2_w.to(str(self.device))
                w.tier = 1
                self.l1_cache[key] = w
                self.l1_used += w.size_bytes
                return w
            return l2_w
        self.misses += 1
        w = self._load_weights(layer, expert)
        if w is None:
            return None
        if self._make_l1_room(w.size_bytes):
            w = w.to(str(self.device))
            w.tier = 1
            self.l1_cache[key] = w
            self.l1_used += w.size_bytes
        else:
            self.l2_cache[key] = w
            self.l2_used += w.size_bytes
            self._evict_l2()
        return w

    def _evict_l1(self):
        while self.l1_used > self.l1_budget and self.l1_cache:
            key, w = self.l1_cache.popitem(last=False)
            self.l1_used -= w.size_bytes

    def _evict_l2(self):
        while self.l2_used > self.l2_budget and self.l2_cache:
            key, w = self.l2_cache.popitem(last=False)
            self.l2_used -= w.size_bytes

    def prefetch_to_l2(self, layer: int, expert: int):
        key = (layer, expert)
        if key in self.l1_cache or key in self.l2_cache:
            return
        w = self._load_weights(layer, expert)
        if w is None:
            return
        w.tier = 2
        with self._prefetch_lock:
            self.l2_cache[key] = w
            self.l2_used += w.size_bytes
            self._evict_l2()

    def lookahead(self, scan_result: list[list[int]]):
        self.hot_set.update(scan_result)
        for layer_idx, experts in enumerate(scan_result):
            for eid in experts:
                self._predictive.record_route(layer_idx, eid)
                self.prefetch_to_l2(layer_idx, eid)
        predicted = self._predictive.predict_hot_all()
        for layer_idx, experts in predicted.items():
            for eid in experts:
                self.prefetch_to_l2(layer_idx, eid)
        self._promote_predicted_to_l1()

    def _promote_predicted_to_l1(self):
        for layer_idx in range(self.n_layers):
            hot = self.hot_set.get_hot(layer_idx)
            for eid in hot:
                key = (layer_idx, eid)
                if key not in self.l1_cache and key in self.l2_cache:
                    l2_w = self.l2_cache[key]
                    if self._make_l1_room(l2_w.size_bytes):
                        w = l2_w.to(str(self.device))
                        w.tier = 1
                        self.l1_cache[key] = w
                        self.l1_used += w.size_bytes

    def record_route(self, layer: int, expert: int):
        self._predictive.record_route(layer, expert)

    def ensure_hot_l1(self):
        self.hot_set.recalculate()
        self._promote_predicted_to_l1()

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l2_hits + self.misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        total = self.l1_hits + self.l2_hits + self.misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def total_hit_rate(self) -> float:
        total = self.l1_hits + self.l2_hits + self.misses
        hits = self.l1_hits + self.l2_hits
        return hits / total if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "l1_hit_rate": f"{self.l1_hit_rate:.1%}",
            "l2_hit_rate": f"{self.l2_hit_rate:.1%}",
            "total_hit_rate": f"{self.total_hit_rate:.1%}",
            "l1_used_gb": round(self.l1_used / 1024**3, 3),
            "l1_budget_gb": round(self.l1_budget / 1024**3, 2),
            "l2_used_gb": round(self.l2_used / 1024**3, 3),
            "l2_budget_gb": round(self.l2_budget / 1024**3, 2),
            "l1_resident": len(self.l1_cache),
            "l2_resident": len(self.l2_cache),
            "accesses": self.accesses,
            "load_errors": self.load_errors,
        }
