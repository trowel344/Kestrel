import pytest

from kestrel.analysis.moe_routes import (
    Route,
    allocate_pool_slots,
    analyze,
    analyze_temporal_reuse,
    parse_routes,
    simulate,
    simulate_partitioned,
)


def test_parse_routes_line():
    text = "[moe-route] tensor=ab12 expert-bytes=4096 type=q4_0 ids=1,2,3"
    routes = parse_routes(text)
    assert len(routes) == 1
    assert routes[0].tensor == 0xAB12
    assert routes[0].expert_bytes == 4096
    assert routes[0].weight_type == "q4_0"
    assert routes[0].experts == (1, 2, 3)


def test_parse_routes_multiple():
    text = "\n".join(
        [
            "[moe-route] tensor=1 expert-bytes=100 type=q4_0 ids=1",
            "[moe-route] tensor=2 expert-bytes=200 type=q1_0 ids=4,5",
        ]
    )
    routes = parse_routes(text)
    assert len(routes) == 2
    assert routes[1].pool == (200, "q1_0")


def test_parse_routes_empty():
    assert parse_routes("no routes here") == []


def test_simulate_zero_capacity():
    routes = [
        Route(1, 100, "q4_0", (1, 2)),
        Route(1, 100, "q4_0", (1, 2)),
    ]
    hits, accesses = simulate(routes, {(100, "q4_0"): 0}, "lru")
    assert hits == 0
    assert accesses == 4


def test_simulate_lru_hits():
    routes = [
        Route(1, 100, "q4_0", (1,)),
        Route(1, 100, "q4_0", (1,)),
    ]
    hits, accesses = simulate(routes, {(100, "q4_0"): 4}, "lru")
    assert hits == 1
    assert accesses == 2


def test_simulate_lfu_differs_from_lru():
    routes = [
        Route(1, 100, "q4_0", (1,)),
        Route(2, 100, "q4_0", (2,)),
        Route(1, 100, "q4_0", (1,)),
        Route(2, 100, "q4_0", (2,)),
        Route(3, 100, "q4_0", (3,)),
        Route(1, 100, "q4_0", (1,)),
        Route(2, 100, "q4_0", (2,)),
        Route(3, 100, "q4_0", (3,)),
    ]
    cap = {(100, "q4_0"): 2}
    lru = simulate(routes, cap, "lru")
    lfu = simulate(routes, cap, "lfu")
    assert lru != lfu
    # LFU keeps the two hottest (1 and 2); all their re-accesses hit.
    assert lfu[0] >= lru[0]


def test_simulate_partitioned_bounds_capacity():
    routes = [
        Route(1, 100, "q4_0", (1, 2)),
        Route(2, 100, "q4_0", (3, 4)),
        Route(1, 100, "q4_0", (1, 2)),
    ]
    hits, accesses = simulate_partitioned(routes, {(100, "q4_0"): 4})
    assert accesses == 6
    assert hits == 2


def test_allocate_pool_slots_respects_budget():
    routes = [
        Route(1, 100, "q4_0", tuple(range(256))),
        Route(2, 100, "q4_0", tuple(range(256))),
    ]
    slots = allocate_pool_slots(routes, 1, 256)  # 1 MiB budget
    total_bytes = slots[(100, "q4_0")] * 100
    assert total_bytes <= 1 * 1024 * 1024


def test_analyze_returns_rows():
    routes = [Route(1, 100, "q4_0", (1, 2)) for _ in range(5)]
    rows = analyze(routes, [64, 128], 2)
    assert len(rows) == 6  # 2 budgets x 3 policies (lru, lfu, layer-lru)
    for row in rows:
        assert 0.0 <= row["hit_rate"] <= 100.0


def test_analyze_includes_layer_lru():
    routes = [Route(1, 100, "q4_0", (1,)) for _ in range(3)]
    rows = analyze(routes, [64], 2)
    assert {row["policy"] for row in rows} == {"lru", "lfu", "layer-lru"}


def test_temporal_reuse_window():
    routes = [
        Route(1, 100, "q4_0", (1,)),
        Route(1, 100, "q4_0", (2,)),
        Route(1, 100, "q4_0", (1,)),
    ]
    rows = analyze_temporal_reuse(routes, [2])
    assert rows[0]["window"] == 2
    assert rows[0]["accesses"] == 3
    # window of 2 retains last 2 expert sets; third access to 1 hits
    assert rows[0]["hits"] == 1


def test_temporal_reuse_peak_bytes():
    routes = [
        Route(1, 100, "q4_0", (1,)),
        Route(2, 100, "q4_0", (2,)),
    ]
    rows = analyze_temporal_reuse(routes, [1])
    # each tensor holds its own 1-set window: peak = 2 tensors * 100 bytes
    assert rows[0]["peak_bytes"] == 200


def test_temporal_reuse_zero_window_raises():
    with pytest.raises(ValueError):
        analyze_temporal_reuse([], [0])


def _reference_temporal_reuse(routes, window):
    from collections import deque

    histories = {}
    tensor_bytes = {}
    resident_bytes = 0
    peak = 0
    hits = 0
    accesses = 0
    for route in routes:
        history = histories.setdefault(route.tensor, deque(maxlen=window))
        resident = set().union(*history) if history else set()
        hits += sum(expert in resident for expert in route.experts)
        accesses += len(route.experts)
        old = tensor_bytes.get(route.tensor, 0)
        history.append(set(route.experts))
        new = len(set().union(*history)) * route.expert_bytes
        tensor_bytes[route.tensor] = new
        resident_bytes += new - old
        peak = max(peak, resident_bytes)
    return {"hits": hits, "accesses": accesses, "peak_bytes": peak, "final_bytes": resident_bytes}


def test_temporal_reuse_matches_union_reference():
    """The incremental per-tensor counter must equal the set-union definition."""
    import random

    rng = random.Random(42)
    routes = []
    for _ in range(500):
        routes.append(
            Route(
                rng.randint(0, 7),
                100,
                "q4_0",
                tuple(rng.sample(range(40), rng.randint(1, 6))),
            )
        )
    for window in (1, 2, 3, 8):
        ours = analyze_temporal_reuse(routes, [window])[0]
        reference = _reference_temporal_reuse(routes, window)
        assert {key: ours[key] for key in ("hits", "accesses", "peak_bytes", "final_bytes")} == reference


def test_lfu_compaction_is_behavior_preserving(monkeypatch):
    """Rebuilding a stale LFU heap from live state must not change hits."""
    from kestrel.analysis import moe_routes

    routes = [Route(index, 100, "q4_0", (index,)) for index in range(300)] * 10
    cap = {(100, "q4_0"): 400}
    baseline = simulate(routes, cap, "lfu")
    monkeypatch.setattr(moe_routes, "_compact_lfu_heap", lambda *_args, **_kwargs: None)
    assert simulate(routes, cap, "lfu") == baseline
    assert baseline[1] == 3000
