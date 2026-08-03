#!/usr/bin/env python3
"""Dependency-light policy simulation.

This checks controller stability under changing conditions. It deliberately
does not claim a throughput speedup; real throughput belongs in benchmark.py.
"""

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel import AdaptiveController


SCENARIOS = {
    "healthy_mtp": dict(tau=2.6, cache_hit_rate=0.9, confidence=0.8, disk=0),
    "cold_cache": dict(tau=2.6, cache_hit_rate=0.2, confidence=0.8, disk=0),
    "low_acceptance": dict(tau=1.1, cache_hit_rate=0.9, confidence=0.8, disk=0),
    "io_pressure": dict(tau=2.6, cache_hit_rate=0.9, confidence=0.8, disk=8),
}


def simulate(name: str, values: dict, cycles: int = 30) -> dict:
    controller = AdaptiveController(window=5)
    controller.set_phase("decode")
    history = []
    for _ in range(cycles):
        controller.update(
            tau=values["tau"],
            cache_hit_rate=values["cache_hit_rate"],
            draft_confidence=values["confidence"],
            verify_latency=0.1,
            draft_latency=0.01,
            disk_queue_depth=values["disk"],
        )
        history.append(controller.current_strategy.spec_mode.value)
    return {
        "name": name,
        "strategies": dict(Counter(history)),
        "summary": controller.summary(),
    }


if __name__ == "__main__":
    print("Kestrel controller policy simulation")
    for scenario_name, scenario in SCENARIOS.items():
        result = simulate(scenario_name, scenario)
        print(f"  {scenario_name:16s} {result['strategies']}")
