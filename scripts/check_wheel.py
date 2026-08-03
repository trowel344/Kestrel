#!/usr/bin/env python3
"""Fail CI when a Kestrel wheel crosses the documented production boundary."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

FORBIDDEN_PARTS = {
    "research",
    "benchmark_results",
    "tests",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".bin",
    ".env",
    ".key",
    ".pem",
}
REQUIRED = {
    "kestrel/cli.py",
    "kestrel/model_store.py",
    "kestrel/gguf/metadata.py",
    "kestrel/providers/ollama.py",
}


def inspect_wheel(path: Path) -> list[str]:
    findings = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    for name in sorted(names):
        item = Path(name)
        if any(part in FORBIDDEN_PARTS for part in item.parts):
            findings.append(f"forbidden package path: {name}")
        if item.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden packaged file type: {name}")
    for required in sorted(REQUIRED - names):
        findings.append(f"required runtime file missing: {required}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    findings = inspect_wheel(args.wheel)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}")
        raise SystemExit(1)
    print(f"Wheel production boundary: PASS ({args.wheel})")


if __name__ == "__main__":
    main()
