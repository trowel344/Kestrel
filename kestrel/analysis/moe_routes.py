from __future__ import annotations

import argparse
import heapq
import re
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

ROUTE_RE = re.compile(
    r"\[moe-route\] tensor=([0-9a-fA-F]+) "
    r"expert-bytes=(\d+) type=([^ ]+) ids=([0-9,-]+)"
)


@dataclass(frozen=True)
class Route:
    tensor: int
    expert_bytes: int
    weight_type: str
    experts: tuple[int, ...]

    @property
    def pool(self) -> tuple[int, str]:
        return self.expert_bytes, self.weight_type


def parse_routes(text: str) -> list[Route]:
    routes: list[Route] = []
    for match in ROUTE_RE.finditer(text):
        experts = tuple(int(value) for value in match.group(4).split(","))
        routes.append(
            Route(
                tensor=int(match.group(1), 16),
                expert_bytes=int(match.group(2)),
                weight_type=match.group(3),
                experts=experts,
            )
        )
    return routes


def allocate_pool_slots(
    routes: list[Route], budget_mib: int, experts_per_tensor: int
) -> dict[tuple[int, str], int]:
    tensors_by_pool: dict[tuple[int, str], set[int]] = defaultdict(set)
    for route in routes:
        tensors_by_pool[route.pool].add(route.tensor)
    weights = {
        pool: expert_bytes * len(tensors)
        for pool, tensors in tensors_by_pool.items()
        for expert_bytes in (pool[0],)
    }
    total_weight = sum(weights.values())
    budget = budget_mib * 1024 * 1024
    result: dict[tuple[int, str], int] = {}
    for pool, weight in weights.items():
        expert_bytes = pool[0]
        share = budget * weight // max(1, total_weight)
        possible = len(tensors_by_pool[pool]) * experts_per_tensor
        result[pool] = min(possible, share // expert_bytes)
    return result


def _cache_simulate(
    routes: list[Route],
    capacities: dict[tuple[int, str], int] | dict[tuple[tuple[int, str], int], int],
    *,
    group_key,
    cache_key,
    eviction: str,
) -> tuple[int, int]:
    """Generic expert-cache hit simulation.

    ``group_key(route)`` selects the sub-cache a route's experts occupy,
    ``cache_key(route, expert)`` the key of an expert inside that sub-cache,
    and ``eviction`` the replacement order (``"lfu"`` or a FIFO/LRU ordering).

    LFU heap entries are ``(frequency, last_access, key)``; updates re-push the
    entry and stale copies are discarded lazily during eviction, matching the
    runtime policy without an O(cache-size) scan on every miss.
    """
    hits = 0
    accesses = 0
    if eviction == "lfu":
        states: dict[tuple[int, str], dict] = {group: {} for group in capacities}
        heaps: dict[tuple[int, str], list] = {group: [] for group in capacities}
        clock = 0
        for route in routes:
            group = group_key(route)
            state = states.setdefault(group, {})
            heap = heaps.setdefault(group, [])
            capacity = capacities.get(group, 0)
            for expert in route.experts:
                accesses += 1
                clock += 1
                key = cache_key(route, expert)
                current = state.get(key)
                if current is not None:
                    hits += 1
                    updated = (current[0] + 1, clock)
                    state[key] = updated
                    heapq.heappush(heap, (updated[0], updated[1], key))
                    continue
                if capacity <= 0:
                    continue
                if len(state) >= capacity:
                    while heap:
                        frequency, last_access, victim = heapq.heappop(heap)
                        if state.get(victim) == (frequency, last_access):
                            del state[victim]
                            break
                state[key] = (1, clock)
                heapq.heappush(heap, (1, clock, key))
        return hits, accesses

    caches: dict = {group: OrderedDict() for group in capacities}
    for route in routes:
        group = group_key(route)
        cache = caches.setdefault(group, OrderedDict())
        capacity = capacities.get(group, 0)
        for expert in route.experts:
            accesses += 1
            key = cache_key(route, expert)
            if key in cache:
                hits += 1
                cache[key] += 1
                cache.move_to_end(key)
                continue
            if capacity <= 0:
                continue
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[key] = 1
    return hits, accesses


def simulate(
    routes: list[Route],
    capacities: dict[tuple[int, str], int],
    policy: str,
) -> tuple[int, int]:
    return _cache_simulate(
        routes,
        capacities,
        group_key=lambda route: route.pool,
        cache_key=lambda route, expert: (route.tensor, expert),
        eviction="lfu" if policy == "lfu" else "lru",
    )


def simulate_partitioned(
    routes: list[Route], capacities: dict[tuple[int, str], int]
) -> tuple[int, int]:
    tensors_by_pool: dict[tuple[int, str], list[int]] = defaultdict(list)
    for route in routes:
        if route.tensor not in tensors_by_pool[route.pool]:
            tensors_by_pool[route.pool].append(route.tensor)
    tensor_capacities: dict[tuple[tuple[int, str], int], int] = {}
    for pool, tensors in tensors_by_pool.items():
        total = capacities.get(pool, 0)
        base, remainder = divmod(total, max(1, len(tensors)))
        for index, tensor in enumerate(tensors):
            tensor_capacities[(pool, tensor)] = base + (index < remainder)
    return _cache_simulate(
        routes,
        tensor_capacities,
        group_key=lambda route: (route.pool, route.tensor),
        cache_key=lambda route, expert: expert,
        eviction="lru",
    )


def analyze_temporal_reuse(
    routes: list[Route], windows: list[int]
) -> list[dict[str, int | float]]:
    """Measure the Q4 working set needed to reuse recent per-tensor routes.

    A window of N retains the union of the last N routed expert sets for every
    expert tensor. ``peak_bytes`` is the maximum backing-store footprint seen
    over the trace, not merely the final snapshot. Hits are measured before
    inserting the current route, matching what an online cache can know.
    """

    rows: list[dict[str, int | float]] = []
    for window in windows:
        if window <= 0:
            raise ValueError(f"temporal window must be positive, got {window}")
        histories: dict[int, deque[set[int]]] = defaultdict(
            lambda window=window: deque(maxlen=window)
        )
        tensor_bytes: dict[int, int] = {}
        resident_bytes = 0
        peak_bytes = 0
        hits = 0
        accesses = 0

        for route in routes:
            history = histories[route.tensor]
            resident = set().union(*history) if history else set()
            hits += sum(expert in resident for expert in route.experts)
            accesses += len(route.experts)

            old_bytes = tensor_bytes.get(route.tensor, 0)
            history.append(set(route.experts))
            new_bytes = len(set().union(*history)) * route.expert_bytes
            tensor_bytes[route.tensor] = new_bytes
            resident_bytes += new_bytes - old_bytes
            peak_bytes = max(peak_bytes, resident_bytes)

        rows.append(
            {
                "window": window,
                "hits": hits,
                "accesses": accesses,
                "hit_rate": 100.0 * hits / accesses if accesses else 0.0,
                "peak_bytes": peak_bytes,
                "final_bytes": resident_bytes,
            }
        )
    return rows


def analyze(
    routes: list[Route], budgets_mib: list[int], experts_per_tensor: int = 256
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for budget in budgets_mib:
        capacities = allocate_pool_slots(routes, budget, experts_per_tensor)
        for policy in ("lru", "lfu", "layer-lru"):
            if policy == "layer-lru":
                hits, accesses = simulate_partitioned(routes, capacities)
            else:
                hits, accesses = simulate(routes, capacities, policy)
            rows.append(
                {
                    "budget_mib": budget,
                    "policy": policy,
                    "slots": sum(capacities.values()),
                    "hits": hits,
                    "accesses": accesses,
                    "hit_rate": 100.0 * hits / accesses if accesses else 0.0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate Q4 expert-cache sizes from a Kestrel MoE route trace"
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--budgets", default="1024,2048,3072,4096")
    parser.add_argument(
        "--reuse-windows",
        default="1,2,4,8,16,32",
        help="comma-separated per-tensor recent-route windows",
    )
    parser.add_argument("--experts-per-tensor", type=int, default=256)
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]
    routes = parse_routes(args.log.read_text(errors="replace"))
    if not routes:
        raise SystemExit("no [moe-route] records found")
    print("budget_mib policy slots hits accesses hit_rate")
    for row in analyze(routes, budgets, args.experts_per_tensor):
        print(
            f"{row['budget_mib']:10d} {row['policy']:>6s} "
            f"{row['slots']:5d} {row['hits']:8d} {row['accesses']:8d} "
            f"{row['hit_rate']:8.2f}%"
        )
    windows = [int(value) for value in args.reuse_windows.split(",")]
    print("\nwindow hits accesses hit_rate peak_gib final_gib")
    for row in analyze_temporal_reuse(routes, windows):
        print(
            f"{row['window']:6d} {row['hits']:8d} {row['accesses']:8d} "
            f"{row['hit_rate']:8.2f}% {row['peak_bytes']/2**30:8.2f} "
            f"{row['final_bytes']/2**30:9.2f}"
        )


if __name__ == "__main__":
    main()
