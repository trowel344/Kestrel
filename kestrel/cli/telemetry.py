"""Live telemetry: a lightweight hardware + server-rate dashboard for serve."""

from __future__ import annotations

import sys
import threading
import urllib.error
import urllib.request
from typing import TextIO

from . import probes

# llama-server Prometheus metric names that carry the current generation rate.
_TPS_METRICS = ("llm_tokens_per_second", "llm_token_s", "llama_perf_token_s")


def _metric_value(text: str) -> float | None:
    """Best-effort token/s from llama-server ``/metrics`` text.

    Matches the first gauge whose name is a known token-rate metric. Returns
    ``None`` when the endpoint served no matching metric (e.g. a build without
    metrics support), so the dashboard shows ``tok/s n/a`` instead of failing.
    """
    for name in _TPS_METRICS:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped.startswith(name):
                continue
            fields = stripped.split()
            if len(fields) == 2:
                try:
                    return float(fields[1])
                except ValueError:
                    continue
    return None


def _server_tps(host: str, port: int) -> float | None:
    probe_host = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}.get(host, host)
    if ":" in probe_host and not probe_host.startswith("["):
        probe_host = f"[{probe_host}]"
    try:
        with urllib.request.urlopen(f"http://{probe_host}:{port}/metrics", timeout=2) as resp:
            if resp.status != 200:
                return None
            body = resp.read(256 * 1024).decode("utf-8", errors="replace")
            return _metric_value(body)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def live_dashboard(
    stop: threading.Event,
    *,
    host: str,
    port: int,
    interval: float = 2.0,
    out: TextIO | None = None,
) -> None:
    """Poll VRAM/RAM/swap (and server token rate when available) until ``stop``.

    Each poll overwrites a single status line with ``\\r`` so a foreground
    ``kestrel serve`` gets a live dashboard without spamming the log.
    """
    out = out or sys.stderr
    while not stop.is_set():
        gpu = probes.detect_gpu()
        memory = probes._memory_snapshot()
        tps = _server_tps(host, port)
        vram = f"{gpu['vram_free_mb'] / 1024:.1f}/{gpu['vram_total_mb'] / 1024:.1f} GiB" if gpu else "n/a"
        tps_text = f"{tps:.1f} tok/s" if tps is not None else "tok/s n/a"
        line = (
            f"\rKestrel {tps_text} | VRAM free {vram} | "
            f"RAM free {memory['ram_available_mib'] / 1024:.1f} GiB | "
            f"swap used {memory['swap_used_mib'] / 1024:.1f} GiB\x1b[K"
        )
        out.write(line)
        out.flush()
        stop.wait(interval)
    out.write("\r\x1b[K")
    out.flush()
