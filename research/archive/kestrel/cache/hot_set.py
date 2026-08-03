class HotExpertSet:
    def __init__(self, n_layers: int, n_experts: int, capacity_per_layer: int = 20):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.capacity_per_layer = capacity_per_layer
        self._hot_sets: list[set[int]] = [set() for _ in range(n_layers)]
        self._frequencies: list[dict[int, int]] = [{} for _ in range(n_layers)]
        self._total_routes: list[int] = [0 for _ in range(n_layers)]

    def update(self, scan_result: list[list[int]]):
        for layer_idx, experts in enumerate(scan_result):
            for eid in experts:
                self._frequencies[layer_idx][eid] = self._frequencies[layer_idx].get(eid, 0) + 1
            self._total_routes[layer_idx] += len(experts)

    def update_from_predictor(self, predictor) -> None:
        for layer_idx in range(self.n_layers):
            freqs = predictor.get_frequencies(layer_idx)
            for eid, count in freqs.items():
                self._frequencies[layer_idx][eid] = self._frequencies[layer_idx].get(eid, 0) + count
            self._total_routes[layer_idx] += sum(freqs.values())

    def recalculate(self):
        for layer_idx in range(self.n_layers):
            sorted_experts = sorted(
                self._frequencies[layer_idx].items(),
                key=lambda x: x[1],
                reverse=True
            )
            self._hot_sets[layer_idx] = {
                eid for eid, _ in sorted_experts[:self.capacity_per_layer]
            }

    def get_hot(self, layer: int) -> set[int]:
        return self._hot_sets[layer]

    def get_all_hot(self) -> dict[int, set[int]]:
        return {i: self._hot_sets[i] for i in range(self.n_layers)}

    def is_hot(self, layer: int, expert_id: int) -> bool:
        return expert_id in self._hot_sets[layer]

    def coverage(self, layer: int, experts: list[int]) -> float:
        if not experts:
            return 0.0
        hot = self._hot_sets[layer]
        covered = sum(1 for e in experts if e in hot)
        return covered / len(experts)

    def entropy(self, layer: int) -> float:
        import math
        freq = self._frequencies[layer]
        total = sum(freq.values())
        if total == 0:
            return float("inf")
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def vram_bytes(self, expert_bytes: int = 0) -> int:
        if expert_bytes == 0:
            return 0
        total_experts = sum(len(s) for s in self._hot_sets)
        return total_experts * expert_bytes

    def reset(self):
        self._hot_sets = [set() for _ in range(self.n_layers)]
        self._frequencies = [{} for _ in range(self.n_layers)]
        self._total_routes = [0 for _ in range(self.n_layers)]
