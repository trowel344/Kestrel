import unittest

from kestrel.cache.predictive_cache import PredictiveExpertCache


class PredictiveCacheTests(unittest.TestCase):
    def test_predict_hot_all_exists_and_is_layer_scoped(self):
        cache = PredictiveExpertCache(2, 8, 4)
        cache.record_route(0, 3)
        cache.record_route(0, 3)
        cache.record_route(1, 5)
        self.assertEqual(cache.predict_hot_all(top_k=1), {0: [3], 1: [5]})


if __name__ == "__main__":
    unittest.main()
