import unittest

from kestrel.analysis.moe_routes import (
    allocate_pool_slots,
    analyze,
    analyze_temporal_reuse,
    parse_routes,
)


class MoeRouteAnalysisTests(unittest.TestCase):
    def test_parses_routes_and_keeps_tensor_identity(self):
        routes = parse_routes(
            "noise\n"
            "[moe-route] tensor=00000000000000ab expert-bytes=1024 type=q4_0 ids=3,7\n"
        )
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].tensor, 0xAB)
        self.assertEqual(routes[0].experts, (3, 7))

    def test_budget_is_split_proportionally_between_pool_shapes(self):
        routes = parse_routes(
            "[moe-route] tensor=01 expert-bytes=1024 type=q4_0 ids=1\n"
            "[moe-route] tensor=02 expert-bytes=2048 type=q4_0 ids=1\n"
        )
        slots = allocate_pool_slots(routes, budget_mib=3, experts_per_tensor=4096)
        self.assertEqual(slots[(1024, "q4_0")], 1024)
        self.assertEqual(slots[(2048, "q4_0")], 1024)

    def test_lru_reports_expected_reuse(self):
        routes = parse_routes(
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=1\n"
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=2\n"
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=1\n"
        )
        rows = analyze(routes, budgets_mib=[2], experts_per_tensor=256)
        lru = next(row for row in rows if row["policy"] == "lru")
        self.assertEqual(lru["hits"], 1)
        self.assertEqual(lru["accesses"], 3)

    def test_lfu_uses_lru_to_break_frequency_ties(self):
        routes = parse_routes(
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=1\n"
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=2\n"
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=3\n"
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=2\n"
        )
        rows = analyze(routes, budgets_mib=[2], experts_per_tensor=256)
        lfu = next(row for row in rows if row["policy"] == "lfu")
        self.assertEqual(lfu["hits"], 1)

    def test_layer_partition_prevents_cyclic_scan_thrashing(self):
        routes = parse_routes(
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=1\n"
            "[moe-route] tensor=02 expert-bytes=1048576 type=q4_0 ids=2\n"
            "[moe-route] tensor=01 expert-bytes=1048576 type=q4_0 ids=1\n"
            "[moe-route] tensor=02 expert-bytes=1048576 type=q4_0 ids=2\n"
        )
        rows = analyze(routes, budgets_mib=[2], experts_per_tensor=256)
        layer_lru = next(row for row in rows if row["policy"] == "layer-lru")
        self.assertEqual(layer_lru["hits"], 2)

    def test_temporal_reuse_is_online_and_reports_peak_working_set(self):
        routes = parse_routes(
            "[moe-route] tensor=01 expert-bytes=1024 type=q4_0 ids=1,2\n"
            "[moe-route] tensor=01 expert-bytes=1024 type=q4_0 ids=2,3\n"
            "[moe-route] tensor=02 expert-bytes=2048 type=q4_0 ids=4\n"
            "[moe-route] tensor=01 expert-bytes=1024 type=q4_0 ids=1,3\n"
        )
        row = analyze_temporal_reuse(routes, windows=[2])[0]

        # Current routes are checked before insertion: expert 2 hits on the
        # second route, then experts 1 and 3 both hit on the fourth.
        self.assertEqual(row["hits"], 3)
        self.assertEqual(row["accesses"], 7)
        # Tensor 01 retains experts 1, 2, 3 while tensor 02 retains expert 4.
        self.assertEqual(row["peak_bytes"], 3 * 1024 + 2 * 1024)

    def test_temporal_reuse_rejects_nonpositive_windows(self):
        with self.assertRaises(ValueError):
            analyze_temporal_reuse([], windows=[0])


if __name__ == "__main__":
    unittest.main()
