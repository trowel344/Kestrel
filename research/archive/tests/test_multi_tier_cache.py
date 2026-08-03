import unittest


class MultiTierCacheSafetyTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from kestrel.cache.multi_tier import CacheLoadError, MultiTierCache
        except ImportError as exc:
            self.skipTest(f"research dependencies are not installed: {exc}")
        self.torch = torch
        self.CacheLoadError = CacheLoadError
        self.MultiTierCache = MultiTierCache

    def make_cache(self, strict=True):
        return self.MultiTierCache(
            n_layers=1,
            n_experts=2,
            n_embd=4,
            n_ff=2,
            model_dir="",
            device="cpu",
            l1_budget_gb=0,
            l2_budget_gb=0.001,
            strict=strict,
        )

    def test_load_failure_never_fabricates_random_weights(self):
        class BrokenLoader:
            def load_experts(self, *args, **kwargs):
                raise OSError("broken shard")

        cache = self.make_cache(strict=True)
        cache._loader = BrokenLoader()
        with self.assertRaises(self.CacheLoadError):
            cache.get(0, 0)
        self.assertEqual(cache.load_errors, 1)
        self.assertEqual(len(cache.l1_cache), 0)
        self.assertEqual(len(cache.l2_cache), 0)

    def test_zero_vram_budget_keeps_loaded_expert_in_l2(self):
        torch = self.torch

        class CpuLoader:
            def load_experts(self, *args, **kwargs):
                return (
                    torch.ones(1, 4, 4, dtype=torch.bfloat16),
                    torch.ones(1, 2, 4, dtype=torch.bfloat16),
                )

        cache = self.make_cache()
        cache._loader = CpuLoader()
        weights = cache.get(0, 0)
        self.assertEqual(weights.gate_up.device.type, "cpu")
        self.assertEqual(len(cache.l1_cache), 0)
        self.assertEqual(len(cache.l2_cache), 1)


if __name__ == "__main__":
    unittest.main()
