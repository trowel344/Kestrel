import math
import threading
from collections import OrderedDict, Counter


class PredictiveExpertCache:
    def __init__(self, n_layers: int, n_experts: int, capacity: int):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.capacity = capacity
        self._cache: OrderedDict[tuple[int, int], object] = OrderedDict()
        self._frequency: list[Counter] = [Counter() for _ in range(n_layers)]
        self._access = 0
        self._hits = 0
        self._lock = threading.Lock()

    def get(self, layer: int, expert: int):
        key = (layer, expert)
        with self._lock:
            self._access += 1
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, layer: int, expert: int, weights):
        key = (layer, expert)
        with self._lock:
            self._cache[key] = weights
            self._cache.move_to_end(key)
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def record_route(self, layer: int, expert: int):
        self._frequency[layer][expert] += 1

    def predict_hot(self, layer: int, top_k: int = 20):
        most_common = self._frequency[layer].most_common(top_k)
        return [eid for eid, _ in most_common]

    def predict_hot_all(self, top_k: int = 20) -> dict[int, list[int]]:
        return {
            layer: self.predict_hot(layer, top_k=top_k)
            for layer in range(self.n_layers)
        }

    @property
    def hit_rate(self):
        return self._hits / self._access if self._access > 0 else 0.0

    def summary(self):
        return {
            "hit_rate": f"{self.hit_rate:.1%}",
            "resident": len(self._cache),
            "capacity": self.capacity,
            "accesses": self._access,
        }
