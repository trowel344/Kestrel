"""Hardware probes: GPU, RAM, CPU power policy, page-cache warming.

Every probe is memoized briefly via :func:`kestrel.util.ttl_cache` so repeated
reads in one invocation (or rapid menu redraws) do not re-spawn ``nvidia-smi``
or re-read ``/proc/meminfo`` on every call. Results age out so live hardware
stays reasonably fresh in a long-running interactive session.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..util import ttl_cache


@ttl_cache(seconds=5)
def detect_gpu() -> dict | None:
    """Return the aggregated GPU profile across all devices.

    The returned dict keeps the historical single-GPU shape (``name``,
    ``vram_total_mb``, ``vram_free_mb``) for callers, plus ``count`` and
    ``devices`` so multi-GPU rigs can fit and tensor-split over combined VRAM.
    """
    return _aggregate_gpu(detect_gpus())


def _aggregate_gpu(devices: list[dict]) -> dict | None:
    if not devices:
        return None
    if len(devices) == 1:
        return {**devices[0], "count": 1, "devices": devices}
    return {
        "name": ", ".join(device["name"] for device in devices),
        "vram_total_mb": sum(device["vram_total_mb"] for device in devices),
        "vram_free_mb": sum(device["vram_free_mb"] for device in devices),
        "count": len(devices),
        "devices": devices,
    }


def detect_gpus() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        devices = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            # Split the memory columns from the right so a vendor name containing a
            # comma ("Foo, Inc. ...") does not misalign the following columns.
            head, total, free = line.rsplit(",", 2)
            devices.append(
                {
                    "name": head.strip() or "unknown",
                    "vram_total_mb": int(total.strip()),
                    "vram_free_mb": int(free.strip()),
                }
            )
        return devices
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []


@ttl_cache(seconds=5)
def _available_ram_mib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size / 1024**2)
    except (ValueError, OSError, AttributeError):
        return 0


@ttl_cache(seconds=5)
def _memory_snapshot() -> dict:
    fields = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                fields[key] = int(raw.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    swap_total = fields.get("SwapTotal", 0)
    swap_free = fields.get("SwapFree", 0)
    return {
        "ram_total_mib": fields.get("MemTotal", 0),
        "ram_available_mib": fields.get("MemAvailable", _available_ram_mib()),
        "swap_total_mib": swap_total,
        "swap_used_mib": max(0, swap_total - swap_free),
    }


def _cpu_power_policy() -> dict:
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq")

    def read(path: Path) -> str | None:
        try:
            return path.read_text().strip() or None
        except OSError:
            return None

    no_turbo = read(Path("/sys/devices/system/cpu/intel_pstate/no_turbo"))
    return {
        "governor": read(base / "scaling_governor"),
        "energy_performance_preference": read(base / "energy_performance_preference"),
        "turbo_enabled": None if no_turbo is None else no_turbo == "0",
    }


def _warm_page_cache(paths: list[str]) -> None:
    """Prime the OS page cache for model files before llama.cpp starts.

    A bounded pre-read (max 64 MiB from the start, which covers the GGUF
    header, metadata, and first tensors) plus a Linux ``WILLNEED`` hint for the
    rest. Bounded on purpose so a cold launch is never made meaningfully slower.
    Split-GGUF shards are pre-read concurrently so the warm does not run
    shard-by-shard.
    """
    if not paths:
        return
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            except OSError:
                pass
            finally:
                os.close(fd)

    def pread(path: str) -> None:
        try:
            with open(path, "rb") as f:
                remaining = 64 * 1024 * 1024
                while remaining > 0:
                    chunk = f.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
        except OSError:
            return

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as pool:
        list(pool.map(pread, paths))
