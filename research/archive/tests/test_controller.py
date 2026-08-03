import unittest

from kestrel.core.controller import AdaptiveController
from kestrel.core.monitor import SlidingMetrics
from kestrel.core.strategies import SpecMode


class ControllerTests(unittest.TestCase):
    def test_only_supported_mtp_strategy_is_selected(self):
        controller = AdaptiveController(window=3)
        controller.set_phase("decode")
        for _ in range(3):
            controller.update(
                tau=2.5,
                cache_hit_rate=0.9,
                draft_confidence=0.8,
                verify_latency=0.1,
                draft_latency=0.01,
            )
        self.assertEqual(controller.current_strategy.spec_mode, SpecMode.MTP)

    def test_pressure_disables_speculation(self):
        controller = AdaptiveController(window=1)
        controller.set_phase("decode")
        controller.update(
            tau=3,
            cache_hit_rate=0.9,
            draft_confidence=0.8,
            verify_latency=0.1,
            draft_latency=0.01,
            disk_queue_depth=8,
        )
        self.assertEqual(controller.current_strategy.spec_mode, SpecMode.NONE)

    def test_throughput_uses_same_sliding_window_for_tokens_and_latency(self):
        metrics = SlidingMetrics(window=2)
        for tau in (1, 2, 3):
            metrics.record_cycle(tau, 1, 1, 1, 0)
        self.assertEqual(metrics.tokens_per_second, 2.5)
        self.assertEqual(metrics.tokens_generated, 6)


if __name__ == "__main__":
    unittest.main()
