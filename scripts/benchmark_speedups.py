#!/usr/bin/env python3
"""Benchmark the loading/startup speedups shipped in the 1.2.0 polish pass.

Each measurement reproduces, at the Python level, the overhead the release
removed, using fake subprocesses that emulate the latency of the real tools
(so no real llama.cpp binary or GGUF model is required in this environment).
The end-to-end gain on hardware depends on how slow the real ``llama-cli``
probe, ``ollama list``, and cold page-cache reads actually are.

Benchmarks (median of RUNS runs):

  1. llama-cli capability probe : cold (2 subprocess spawns) vs warm (cache hit)
  2. ollama list                 : cold (spawn) vs warm (in-cache, TTL)
  3. --warm-cache pre-read       : cost of the bounded read vs the cold full read

Exit code is 0 on success.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

PROBE_LATENCY_S = 0.20    # per --help/--version (real builds init a CUDA context)
OLLAMA_LATENCY_S = 0.12
RAMDISK_MIB = 256
RUNS = 3


def fmt(v: float) -> str:
    return f"{v:8.2f} ms"


def saved_line(cold: float, warm: float) -> str:
    avoided = cold - warm
    if warm >= 0.01:
        return f"  speedup {cold / warm:6.1f}x  (avoids {avoided:,.2f} ms)\n"
    return (
        "  warm path is a resident-file/JSON read, so the wall-clock is below\n"
        f"  measurement noise; the released value is the {avoided:,.1f} ms avoided\n"
    )


def median(times: list[float]) -> float:
    return statistics.median(times)


def write_fake_binary(path: Path, sleep_s: float, stdout: str) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        f"time.sleep({sleep_s!r})\n"
        f"sys.stdout.write({stdout!r})\n"
    )
    path.chmod(0o755)


# --------------------------------------------------------------------------- #
# 1) llama-cli capability probe
# --------------------------------------------------------------------------- #
def bench_capability_probe(work: Path) -> tuple[float, float]:
    from kestrel.backends import llama_cpp

    bin_dir = work / "llama-bin"
    bin_dir.mkdir()
    help_text = "--help --temp --threads --spec-type draft-mtp,none\n"
    write_fake_binary(bin_dir / "llama-cli", PROBE_LATENCY_S, help_text)
    model = work / "model.bin"

    # Point the process at the fake llama-cli for the whole benchmark.
    os.environ["KESTREL_LLAMA_CPP_DIR"] = str(bin_dir)
    warm_dir = work / "cap-warm"
    warm_dir.mkdir()

    def make_backend() -> llama_cpp.LlamaCppBackend:
        return llama_cpp.LlamaCppBackend(
            model_path=str(model), llama_cpp_dir=str(bin_dir)
        )

    cold_times: list[float] = []
    warm_times: list[float] = []
    for i in range(RUNS):
        # Cold: fresh cache dir forces the --help/--version subprocess probe.
        cold_dir = work / f"cap-cold-{i}"
        cold_dir.mkdir()
        os.environ["XDG_CACHE_HOME"] = str(cold_dir)
        b = make_backend()
        t0 = time.perf_counter()
        b.capabilities()
        cold_times.append((time.perf_counter() - t0) * 1000)

        # Warm: real persistent cache; the second call is a cache hit.
        os.environ["XDG_CACHE_HOME"] = str(warm_dir)
        b = make_backend()
        b.capabilities()  # prime the persistent cache (first launch)
        t0 = time.perf_counter()
        b.capabilities()
        warm_times.append((time.perf_counter() - t0) * 1000)

    return median(cold_times), median(warm_times)


# --------------------------------------------------------------------------- #
# 2) ollama list (TTL cache)
# --------------------------------------------------------------------------- #
def bench_ollama_list(work: Path) -> tuple[float, float]:
    from kestrel import model_store

    bin_dir = work / "ollama-bin"
    bin_dir.mkdir()
    out = (
        "NAME\tID\tSIZE\tMODIFIED\n"
        "qwen3.5:122b\tabc123\t70 GB\t3 days ago\n"
        "kimi-k3:1t\tdef456\t120 GB\t1 week ago\n"
        "embed:v3\t78f9ea\t150 MB\t1 month ago\n"
    )
    write_fake_binary(bin_dir / "ollama", OLLAMA_LATENCY_S, out)

    prev_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{prev_path}"
    model_store._ollama_list_cache = None

    def one() -> float:
        t0 = time.perf_counter()
        model_store.list_ollama_models()
        return (time.perf_counter() - t0) * 1000

    cold: list[float] = []
    warm: list[float] = []
    for _ in range(RUNS):
        model_store._ollama_list_cache = None
        cold.append(one())
    for _ in range(RUNS):
        warm.append(one())  # stays within the 4 s TTL

    os.environ["PATH"] = prev_path or os.environ.get("PATH", "")

    return median(cold), median(warm)


# --------------------------------------------------------------------------- #
# 3) --warm-cache bounded pre-read vs a cold full read
# --------------------------------------------------------------------------- #
def bench_page_cache(work: Path) -> tuple[float, float, float]:
    from kestrel.cli import _warm_page_cache

    blob = work / "blob.bin"
    with open(blob, "wb") as f:
        for _ in range(RAMDISK_MIB):
            f.write(b"x" * (1024 * 1024))

    def full_read(*, drop: bool) -> float:
        fd = os.open(blob, os.O_RDONLY)
        if drop and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        t0 = time.perf_counter()
        with os.fdopen(fd, "rb") as f:
            while f.read(8 * 1024 * 1024):
                pass
        return (time.perf_counter() - t0) * 1000

    def warm_cost() -> float:
        t0 = time.perf_counter()
        _warm_page_cache([str(blob)])
        return (time.perf_counter() - t0) * 1000

    cold = median([cold_read(blob) for _ in range(RUNS)])
    warm = median([warm_read(blob) for _ in range(RUNS)])
    warm_cache = median([warm_cost() for _ in range(RUNS)])
    return cold, warm, warm_cache


def cold_read(blob: Path) -> float:
    fd = os.open(blob, os.O_RDONLY)
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    t0 = time.perf_counter()
    with os.fdopen(fd, "rb") as f:
        while f.read(8 * 1024 * 1024):
            pass
    return (time.perf_counter() - t0) * 1000


def warm_read(blob: Path) -> float:
    with open(blob, "rb") as f:
        t0 = time.perf_counter()
        while f.read(8 * 1024 * 1024):
            pass
    return (time.perf_counter() - t0) * 1000


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print(f"Kestrel speedup benchmarks ({RUNS} runs, median)")

        print("\n=== 1) llama-cli capability probe (persistent cache) ===")
        cold, warm = bench_capability_probe(work)
        print(f"  cold (spawn --help + --version) : {fmt(cold)}")
        print(f"  warm (cached)                    : {fmt(warm)}")
        print(saved_line(cold, warm).rstrip())

        print("\n=== 2. ollama list (4 s TTL cache) ===")
        ocold, owarm = bench_ollama_list(work)
        print(f"  cold (spawn `ollama list`)       : {fmt(ocold)}")
        print(f"  warm (in-cache, menu redraw)     : {fmt(owarm)}")
        print(saved_line(ocold, owarm).rstrip())

        print("\n=== 3. --warm-cache pre-read vs cold full read ===")
        pr, wr, wc = bench_page_cache(work)
        print(f"  cold full read ({RAMDISK_MIB} MiB, cache dropped): {fmt(pr)}")
        print(f"  warm full read ({RAMDISK_MIB} MiB, resident)    : {fmt(wr)}")
        if wr:
            print(f"  read speedup                    : {pr / wr:6.1f}x")
        print(f"  --warm-cache bounded cost        : {fmt(wc)}")
        if wc:
            print(
                f"  warm throughput                 : {RAMDISK_MIB / (wc / 1000):.0f} MiB/s "
                f"(primes the header+tensors so llama.cpp loads from cache)"
            )

        print(
            "\nNotes:\n"
            "  - These are per-launch/redraw wall-clock deltas here, not token throughput.\n"
            "  - End-to-end effects must be re-validated on real hardware with the real\n"
            "    GGUF + llama-cli (see the release gate runbook)."
        )
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.exit(main())
