from collections import deque, Counter


class ExpertPredictor:
    def __init__(self, n_layers: int, n_experts: int, history_window: int = 64, top_k: int = 20):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self._history: list[deque] = [deque(maxlen=history_window) for _ in range(n_layers)]

    def record_routing(self, layer: int, expert_ids: list[int]):
        self._history[layer].extend(expert_ids)

    def record_scan_result(self, scan_result: list[list[int]]):
        for layer_idx, experts in enumerate(scan_result):
            self.record_routing(layer_idx, experts)

    def predict_hot(self, layer: int) -> list[int]:
        if not self._history[layer]:
            return list(range(self.top_k))
        freq = Counter(self._history[layer])
        return [eid for eid, _ in freq.most_common(self.top_k)]

    def predict_hot_all(self) -> dict[int, list[int]]:
        return {layer: self.predict_hot(layer) for layer in range(self.n_layers)}

    def get_frequencies(self, layer: int) -> dict[int, int]:
        return dict(Counter(self._history[layer]))

    def entropy(self, layer: int) -> float:
        import math
        if not self._history[layer]:
            return float("inf")
        freq = Counter(self._history[layer])
        total = sum(freq.values())
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def reset(self):
        self._history = [deque(maxlen=self._history[0].maxlen) for _ in range(self.n_layers)]
