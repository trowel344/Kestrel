from .monitor import SlidingMetrics
from .strategies import Strategy, SpecMode, STRATEGY_NONE, STRATEGY_MTP


class AdaptiveController:
    def __init__(self, window: int = 20):
        self.metrics = SlidingMetrics(window=window)
        self.current_strategy: Strategy = STRATEGY_NONE
        self.phase = "prefill"

        self.CACHE_LOW_THRESHOLD = 0.4
        self.DISK_QUEUE_HIGH = 4
        self.DRAFT_CONFIDENCE_LOW = 0.4

    def decide_strategy(self) -> Strategy:
        tau = self.metrics.avg_tau()
        cache_hit = self.metrics.avg_cache_hit()
        draft_conf = self.metrics.avg_draft_confidence()
        disk_q = self.metrics.avg_disk_queue()

        if self.phase == "prefill":
            return STRATEGY_NONE

        if self.phase == "terminal":
            return STRATEGY_NONE

        # Kestrel's DFlash self-speculation was invalidated for MoE models.
        # The production controller therefore makes the only decision it can
        # currently enforce truthfully: enable or disable llama.cpp MTP.
        if tau < 1.5:
            return STRATEGY_NONE

        if cache_hit < self.CACHE_LOW_THRESHOLD:
            return STRATEGY_NONE

        if draft_conf < self.DRAFT_CONFIDENCE_LOW:
            return STRATEGY_NONE

        if disk_q > self.DISK_QUEUE_HIGH:
            return STRATEGY_NONE

        return STRATEGY_MTP

    def update(self, **metrics):
        self.metrics.record_cycle(
            tau=metrics.get("tau", 0),
            cache_hit_rate=metrics.get("cache_hit_rate", 1.0),
            draft_confidence=metrics.get("draft_confidence", 0.0),
            verify_latency=metrics.get("verify_latency", 0.0),
            draft_latency=metrics.get("draft_latency", 0.0),
            disk_queue_depth=metrics.get("disk_queue_depth", 0),
        )
        self.current_strategy = self.decide_strategy()

    def set_phase(self, phase: str):
        self.phase = phase

    def should_speculate(self) -> bool:
        return self.current_strategy.spec_mode != SpecMode.NONE

    def get_block_size(self) -> int:
        return self.current_strategy.block_size

    def summary(self) -> dict:
        base = self.metrics.summary()
        base["strategy"] = self.current_strategy.label()
        base["phase"] = self.phase
        return base
