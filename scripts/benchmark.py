#!/usr/bin/env python3
"""Benchmark a real GGUF through Kestrel's llama.cpp backend."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.core.pipeline import InferencePipeline

PROMPTS = {
    "code": "Write a Python merge sort function with concise comments.",
    "reasoning": "Solve 3x^2 + 7x - 6 = 0 and show the key steps.",
    "prose": "In five sentences, write about a robot discovering poetry.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to a runnable GGUF")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--cpu-moe", action="store_true")
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="opt in to an MTP draft block known to exist in this GGUF",
    )
    args = parser.parse_args()

    pipeline = InferencePipeline(
        gguf_path=str(Path(args.model).expanduser().resolve()),
        n_gpu_layers="auto",
        n_ctx=args.ctx_size,
        spec_type="mtp" if args.mtp else "none",
        cpu_moe=args.cpu_moe,
    )
    results = {}
    try:
        for name, prompt in PROMPTS.items():
            started = time.perf_counter()
            text, metrics = pipeline.generate(prompt, max_tokens=args.tokens)
            metrics["wall_seconds"] = round(time.perf_counter() - started, 3)
            metrics["preview"] = text[:120].replace("\n", " ")
            results[name] = metrics
            rate = metrics.get("output_tokens_per_second")
            print(f"{name}: {rate if rate is not None else 'unknown'} tok/s")
    finally:
        pipeline.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
