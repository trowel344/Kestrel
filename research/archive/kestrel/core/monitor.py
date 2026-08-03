from collections import deque


class SlidingMetrics:
    def __init__(self, window: int = 20):
        self.window = window
        self.tau_history: deque[float] = deque(maxlen=window)
        self.cache_hit_history: deque[float] = deque(maxlen=window)
        self.draft_confidence_history: deque[float] = deque(maxlen=window)
        self.verify_latency_history: deque[float] = deque(maxlen=window)
        self.draft_latency_history: deque[float] = deque(maxlen=window)
        self.disk_queue_history: deque[int] = deque(maxlen=window)
        self.tokens_generated = 0.0
        self.cycles_run = 0

    def record_cycle(
        self,
        tau: float,
        cache_hit_rate: float,
        draft_confidence: float,
        verify_latency: float,
        draft_latency: float,
        disk_queue_depth: int = 0,
    ):
        self.tau_history.append(tau)
        self.cache_hit_history.append(cache_hit_rate)
        self.draft_confidence_history.append(draft_confidence)
        self.verify_latency_history.append(verify_latency)
        self.draft_latency_history.append(draft_latency)
        self.disk_queue_history.append(disk_queue_depth)
        self.tokens_generated += tau
        self.cycles_run += 1

    def avg_tau(self) -> float:
        if not self.tau_history:
            return 0.0
        return sum(self.tau_history) / len(self.tau_history)

    def avg_cache_hit(self) -> float:
        if not self.cache_hit_history:
            return 1.0
        return sum(self.cache_hit_history) / len(self.cache_hit_history)

    def avg_draft_confidence(self) -> float:
        if not self.draft_confidence_history:
            return 0.0
        return sum(self.draft_confidence_history) / len(self.draft_confidence_history)

    def avg_verify_latency(self) -> float:
        if not self.verify_latency_history:
            return 0.0
        return sum(self.verify_latency_history) / len(self.verify_latency_history)

    def avg_disk_queue(self) -> float:
        if not self.disk_queue_history:
            return 0.0
        return sum(self.disk_queue_history) / len(self.disk_queue_history)

    @property
    def tokens_per_second(self) -> float:
        total_latency = sum(self.verify_latency_history) + sum(self.draft_latency_history)
        if total_latency == 0:
            return 0.0
        return sum(self.tau_history) / total_latency

    @property
    def speculation_efficiency(self) -> float:
        if self.cycles_run == 0:
            return 0.0
        return self.tokens_generated / self.cycles_run

    def summary(self) -> dict:
        return {
            "tau": round(self.avg_tau(), 2),
            "cache_hit_rate": f"{self.avg_cache_hit():.1%}",
            "draft_confidence": round(self.avg_draft_confidence(), 3),
            "tokens/sec": round(self.tokens_per_second, 2),
            "spec_efficiency": round(self.speculation_efficiency, 2),
            "cycles": self.cycles_run,
            "tokens": round(self.tokens_generated, 2),
        }
