"""CLI handler for deterministic tests against an already-running model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..errors import InputError
from ..evals.model_eval import DEFAULT_MANIFEST, run_model_evaluation
from ..util import write_atomic


def cmd_evaluate(args):
    """Run the bounded frontier manifest; never starts or mutates a model."""
    if args.max_cases is not None and args.max_cases > len(DEFAULT_MANIFEST.cases):
        raise InputError(f"--max-cases cannot exceed {len(DEFAULT_MANIFEST.cases)}")
    report = run_model_evaluation(
        endpoint=args.endpoint,
        model=args.model,
        seed=args.seed,
        timeout=args.timeout,
        max_cases=args.max_cases,
        artifact=args.artifact,
        include_sha256=args.sha256,
    )
    if args.output:
        target = Path(args.output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(target, json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote evaluation report to {target}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1
