#!/usr/bin/env python3
"""Full local GGUF performance, correctness, and stability suite.

This harness keeps llama-server resident so multiple evaluations do not reload
the model. Results are checkpointed after every request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.backends.llama_cpp import resolve_llama_binary


def default_server_binary() -> str:
    """Prefer the MoE-native llama-server build, falling back to ~/llama.cpp."""
    found = resolve_llama_binary("llama-server")
    if found:
        return found
    return str(Path(os.path.expanduser("~/llama.cpp")) / "build" / "bin" / "llama-server")


@dataclass(frozen=True)
class ServerCapabilities:
    flags: frozenset[str]
    spec_types: frozenset[str]

    def supports(self, flag: str) -> bool:
        return flag in self.flags


ALL_CAPS = ServerCapabilities(
    flags=frozenset(
        {
            "--fit",
            "--fit-target",
            "--cpu-moe",
            "--moe-cache",
            "--mmap",
            "--no-warmup",
            "--threads",
            "--threads-batch",
            "--cache-type-k",
            "--flash-attn",
            "--metrics",
            "--reasoning",
            "--spec-type",
            "--override-kv",
            "--spec-draft-n-max",
            "--spec-draft-ngl",
            "--spec-draft-cpu-moe",
            "--spec-draft-model",
            "--spec-draft-hf",
        }
    ),
    spec_types=frozenset({"none", "draft-mtp", "draft-simple", "ngram-cache"}),
)


def parse_server_help(help_text: str) -> ServerCapabilities:
    """Extract the supported long flags and --spec-type values from --help text."""
    flags = {token.rstrip(",;:") for token in help_text.split() if token.startswith("--")}
    match = re.search(r"--spec-type\s+([^\n]+)", help_text)
    spec_types = set()
    if match:
        spec_types = {item for item in match.group(1).split()[0].split(",")}
    return ServerCapabilities(frozenset(flags), frozenset(spec_types))


def detect_server_capabilities(server: str) -> ServerCapabilities:
    """Probe a llama-server binary for the flags and spec types it supports."""
    try:
        result = subprocess.run(
            [server, "--help"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return ServerCapabilities(frozenset(), frozenset())
    if result.returncode != 0:
        return ServerCapabilities(frozenset(), frozenset())
    return parse_server_help(result.stdout + result.stderr)


ACCURACY_CASES = [
    {
        "id": "arithmetic_multiply",
        "prompt": "Return only the integer result of 17 * 23.",
        "patterns": [r"\b391\b"],
    },
    {
        "id": "algebra_linear",
        "prompt": "Solve 2x + 5 = 17. Return only the value of x.",
        "patterns": [r"^\s*6\s*$", r"\bx\s*=\s*6\b"],
    },
    {
        "id": "sequence",
        "prompt": "What is the next number: 2, 4, 8, 16? Return only the number.",
        "patterns": [r"\b32\b"],
    },
    {
        "id": "logic",
        "prompt": (
            "All kestrels are birds. No birds are mammals. Can a kestrel be a "
            "mammal under these premises? Answer only yes or no."
        ),
        "patterns": [r"^\s*no[.!]?\s*$"],
    },
    {
        "id": "factual_capital",
        "prompt": "What is the capital of Australia? Answer with only the city.",
        "patterns": [r"\bcanberra\b"],
    },
    {
        "id": "reading_comprehension",
        "prompt": (
            "Read this fact: The brass key is inside the blue drawer. "
            "Question: what color is the drawer? Answer with one word."
        ),
        "patterns": [r"^\s*blue[.!]?\s*$"],
    },
    {
        "id": "python_semantics",
        "prompt": (
            "In Python, what does len([10, 20, 30]) return? "
            "Return only the integer."
        ),
        "patterns": [r"^\s*3\s*$"],
    },
    {
        "id": "instruction_exact",
        "prompt": 'Output exactly this text and nothing else: KESTREL_OK',
        "patterns": [r"^\s*KESTREL_OK\s*$"],
        "case_sensitive": True,
    },
]


@dataclass
class ResourceSample:
    timestamp: float
    rss_mib: float
    swap_mib: float
    gpu_used_mib: int | None
    gpu_temperature_c: int | None
    gpu_power_w: float | None


class ResourceMonitor:
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(2):
            rss_mib = swap_mib = 0.0
            try:
                status = Path(f"/proc/{self.process.pid}/status").read_text()
                rss = re.search(r"^VmRSS:\s+(\d+)", status, re.MULTILINE)
                swap = re.search(r"^VmSwap:\s+(\d+)", status, re.MULTILINE)
                rss_mib = int(rss.group(1)) / 1024 if rss else 0.0
                swap_mib = int(swap.group(1)) / 1024 if swap else 0.0
            except (OSError, ValueError):
                pass
            gpu_used = None
            gpu_temperature = None
            gpu_power = None
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if parts and int(parts[0]) == self.process.pid:
                        gpu_used = int(parts[1])
                        break
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
                gpu_temperature = int(parts[0])
                gpu_power = float(parts[1])
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.samples.append(
                ResourceSample(
                    time.time(), rss_mib, swap_mib, gpu_used, gpu_temperature, gpu_power
                )
            )

    def summary(self) -> dict:
        if not self.samples:
            return {}
        return {
            "peak_rss_mib": round(max(x.rss_mib for x in self.samples), 1),
            "peak_process_swap_mib": round(max(x.swap_mib for x in self.samples), 1),
            "peak_gpu_mib": max(
                (x.gpu_used_mib for x in self.samples if x.gpu_used_mib is not None),
                default=None,
            ),
            "peak_gpu_temperature_c": max(
                (x.gpu_temperature_c for x in self.samples if x.gpu_temperature_c is not None),
                default=None,
            ),
            "peak_gpu_power_w": max(
                (x.gpu_power_w for x in self.samples if x.gpu_power_w is not None),
                default=None,
            ),
            "sample_count": len(self.samples),
        }


def http_json(url: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_health(base_url: str, process: subprocess.Popen, timeout: int = 900) -> dict:
    started = time.monotonic()
    last = {}
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during load with {process.returncode}")
        try:
            last = http_json(f"{base_url}/health", timeout=5)
            if last.get("status") == "ok":
                last["load_wait_seconds"] = round(time.monotonic() - started, 3)
                return last
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy in {timeout}s; last={last}")


def chat_request(base_url: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the user's requested output format exactly. "
                    "Do not explain unless asked."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }
    started = time.perf_counter()
    response = http_json(f"{base_url}/v1/chat/completions", body, timeout=1800)
    wall = time.perf_counter() - started
    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    return {
        "prompt": prompt,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "wall_seconds": round(wall, 3),
        "raw_response": response,
    }


def answer_text(content: str) -> str:
    """Remove llama.cpp's empty reasoning envelope before exact-match scoring."""
    return re.sub(
        r"^\s*<think>.*?</think>\s*",
        "",
        content,
        count=1,
        flags=re.DOTALL,
    )


def score_case(case: dict, content: str) -> bool:
    flags = 0 if case.get("case_sensitive") else re.IGNORECASE
    answer = answer_text(content)
    return any(re.search(pattern, answer, flags) for pattern in case["patterns"])


def assess_stability(content: str) -> dict:
    """Apply deterministic minimum checks; factual coherence still needs review."""

    answer = answer_text(content).strip()
    words = re.findall(r"[\w'-]+", answer.lower())
    trigrams = [tuple(words[index:index + 3]) for index in range(max(0, len(words) - 2))]
    unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 0.0
    factual_markers = sum(
        marker in answer.lower()
        for marker in ("rayleigh", "scatter", "wavelength", "atmosphere", "blue")
    )
    valid_utf8 = "\ufffd" not in answer
    no_repetition_collapse = unique_ratio >= 0.65
    heuristic_passed = (
        valid_utf8
        and len(words) >= 40
        and no_repetition_collapse
        and factual_markers >= 2
    )
    return {
        "valid_utf8": valid_utf8,
        "word_count": len(words),
        "trigram_unique_ratio": round(unique_ratio, 4),
        "no_repetition_collapse": no_repetition_collapse,
        "factual_marker_count": factual_markers,
        "heuristic_passed": heuristic_passed,
        "manual_coherence_review_required": True,
    }


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_release_gates(result: dict) -> dict:
    decode = next(
        (item for item in result.get("performance", []) if item.get("id") == "decode_64"),
        {},
    )
    timings = decode.get("timings") or {}
    usage = decode.get("usage") or {}
    generated = int(
        timings.get("predicted_n")
        or usage.get("completion_tokens")
        or 0
    )
    decode_tps = timings.get("predicted_per_second")
    prompt_tps = timings.get("prompt_per_second")
    speed_passed = bool(
        generated >= 64 and decode_tps is not None and float(decode_tps) >= 10
    )
    accuracy = result.get("accuracy_summary") or {}
    quality_passed = int(accuracy.get("passed") or 0) >= 7
    stability = (result.get("stability") or {}).get("assessment") or {}
    stability_passed = bool(stability.get("heuristic_passed"))
    return {
        "generated_tokens_measured": generated,
        "prompt_tokens_per_second": prompt_tps,
        "decode_tokens_per_second": decode_tps,
        "speed_floor_passed": speed_passed,
        "accuracy_passed": quality_passed,
        "stability_heuristic_passed": stability_passed,
        "manual_coherence_review_required": True,
        "automated_gates_passed": speed_passed and quality_passed and stability_passed,
    }


def server_command(
    args,
    mode: str,
    caps: ServerCapabilities = ALL_CAPS,
) -> list[str]:
    """Build the llama-server command, emitting only supported flags.

    Flags the binary does not advertise are skipped and reported to stderr.
    Modes that structurally require an unavailable flag raise instead of
    launching a server that cannot perform the requested benchmark.
    """
    unsupported: list[str] = []

    def warn(feature: str, flag: str):
        unsupported.append(f"{feature} (requires {flag})")

    command = [
        args.server,
        "-m", args.model,
        "-ngl", args.gpu_layers,
    ]
    if args.fit == "on" and caps.supports("--fit"):
        command += ["--fit", "on"]
        if caps.supports("--fit-target"):
            command += ["--fit-target", str(args.fit_target)]
    else:
        warn("fit", "--fit")
    if caps.supports("--cpu-moe"):
        command.append("--cpu-moe")
    else:
        warn("CPU MoE", "--cpu-moe")
    if caps.supports("--moe-cache"):
        command += ["--moe-cache", args.moe_cache]
    else:
        warn("MoE cache", "--moe-cache")
    if caps.supports("--mmap"):
        command.append("--mmap")
    if caps.supports("--no-warmup"):
        command.append("--no-warmup")
    command += [
        "-c", str(args.ctx_size),
        "-np", str(args.parallel),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
    ]
    if caps.supports("--threads"):
        command += ["-t", str(args.threads)]
        if caps.supports("--threads-batch"):
            command += ["-tb", str(args.threads)]
    else:
        warn("thread counts", "--threads")
    if caps.supports("--cache-type-k"):
        command += [
            "--cache-type-k", args.kv_cache_type,
            "--cache-type-v", args.kv_cache_type,
        ]
    else:
        warn("KV cache type", "--cache-type-k")
    if caps.supports("--flash-attn"):
        command += ["--flash-attn", "auto"]
    command += ["--host", "127.0.0.1", "--port", str(args.port)]
    if caps.supports("--metrics"):
        command.append("--metrics")
    else:
        warn("server metrics", "--metrics")
    if caps.supports("--reasoning"):
        command += ["--reasoning", "off"]
    else:
        warn("reasoning off", "--reasoning")

    spec_type = {
        "mtp": "draft-mtp",
        "draft": "draft-simple",
        "baseline": "none",
    }[mode]
    if mode in ("mtp", "draft") and not caps.supports("--spec-type"):
        raise ValueError(
            f"mode '{mode}' requires --spec-type support but this llama-server "
            "build does not advertise it"
        )
    if spec_type != "none" and spec_type not in caps.spec_types:
        raise ValueError(
            f"mode '{mode}' requires spec type '{spec_type}', which this "
            f"llama-server does not expose (available: "
            f"{sorted(caps.spec_types) or 'none'})"
        )
    if caps.supports("--spec-type"):
        command += ["--spec-type", spec_type]

    if args.expert_used_count is not None:
        if caps.supports("--override-kv"):
            command += [
                "--override-kv",
                f"qwen35moe.expert_used_count=int:{args.expert_used_count}",
            ]
        else:
            warn("expert_used_count override", "--override-kv")
    if mode in ("mtp", "draft"):
        if caps.supports("--spec-draft-n-max"):
            command += ["--spec-draft-n-max", str(args.mtp_tokens)]
        else:
            warn("draft length", "--spec-draft-n-max")
        if caps.supports("--spec-draft-ngl"):
            command += ["--spec-draft-ngl", str(args.mtp_gpu_layers)]
    if mode == "mtp":
        if caps.supports("--spec-draft-cpu-moe"):
            command.append("--spec-draft-cpu-moe")
        else:
            warn("MTP draft on CPU", "--spec-draft-cpu-moe")
    elif mode == "draft":
        if args.draft_model and caps.supports("--spec-draft-model"):
            command += ["--spec-draft-model", args.draft_model]
        elif args.draft_hf and caps.supports("--spec-draft-hf"):
            command += ["--spec-draft-hf", args.draft_hf]
        elif args.draft_model or args.draft_hf:
            raise ValueError(
                "draft mode requires a server that supports "
                "--spec-draft-model or --spec-draft-hf"
            )
        else:
            raise ValueError("draft mode requires --draft-model or --draft-hf")

    if unsupported:
        print(
            "WARNING: requested features skipped because this llama-server build "
            f"does not advertise them: {', '.join(unsupported)}",
            file=sys.stderr,
        )
    return command


def checkpoint(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, path)


def run_mode(
    args,
    mode: str,
    output_path: Path,
    caps: ServerCapabilities = ALL_CAPS,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_name(f"{output_path.stem}-server.log")
    command = server_command(args, mode, caps)
    log_file = log_path.open("w")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    monitor = ResourceMonitor(process)
    monitor.start()
    base_url = f"http://127.0.0.1:{args.port}"
    result = {
        "mode": mode,
        "model_sha256": args.model_sha256,
        "command": command,
        "server_log": str(log_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "health": None,
        "performance": [],
        "accuracy": [],
        "stability": None,
        "resources": {},
    }
    try:
        result["health"] = wait_for_health(base_url, process)
        checkpoint(output_path, {"active_mode": result})

        perf_cases = [
            (
                "decode_64",
                "Explain how virtual memory and memory mapping work in at least 150 words.",
                64,
            ),
            (
                "long_prefill",
                ("The quick brown fox jumps over the lazy dog. " * 80)
                + "\nSummarize that repeated sentence in five words.",
                12,
            ),
        ]
        for case_id, prompt, tokens in perf_cases:
            print(f"[{mode}] performance: {case_id}", flush=True)
            item = chat_request(base_url, prompt, tokens)
            item["id"] = case_id
            result["performance"].append(item)
            checkpoint(output_path, {"active_mode": result})

        for case in ACCURACY_CASES:
            print(f"[{mode}] accuracy: {case['id']}", flush=True)
            item = chat_request(base_url, case["prompt"], 24)
            item["id"] = case["id"]
            item["passed"] = score_case(case, item["content"])
            result["accuracy"].append(item)
            checkpoint(output_path, {"active_mode": result})

        print(f"[{mode}] stability: 128 tokens", flush=True)
        result["stability"] = chat_request(
            base_url,
            "Write a coherent, factual explanation of why the sky appears blue.",
            128,
        )
        result["stability"]["assessment"] = assess_stability(
            result["stability"]["content"]
        )
    finally:
        monitor.stop()
        result["resources"] = monitor.summary()
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        result["server_returncode"] = process.returncode
        log_file.close()
        scores = [item["passed"] for item in result["accuracy"]]
        result["accuracy_summary"] = {
            "passed": sum(scores),
            "total": len(scores),
            "accuracy": round(sum(scores) / len(scores), 4) if scores else None,
        }
        result["release_gate_summary"] = evaluate_release_gates(result)
        checkpoint(output_path, {"active_mode": result})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to the exact GGUF artifact under test")
    parser.add_argument(
        "--server",
        default=default_server_binary(),
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "mtp", "draft", "both"),
        default="baseline",
        help="MTP modes are explicit because most corrected GGUFs are target-only",
    )
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--fit", choices=("on", "off"), default="on")
    parser.add_argument("--fit-target", type=int, default=1800)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="server slots; one gives a single request the full context",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--kv-cache-type", default="q8_0")
    parser.add_argument(
        "--moe-cache",
        default="auto",
        help="llama.cpp MoE cache mode: auto, on, off, or a MiB budget",
    )
    parser.add_argument(
        "--expert-used-count",
        type=int,
        choices=range(1, 9),
        help="R&D override for Qwen3.5 routed experts selected per token",
    )
    parser.add_argument("--mtp-tokens", type=int, default=3)
    parser.add_argument("--mtp-gpu-layers", default="0")
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-hf")
    parser.add_argument(
        "--output",
        default=str(ROOT / "benchmark_results" / "full_model_suite.json"),
    )
    args = parser.parse_args()
    args.model = str(Path(args.model).expanduser().resolve())
    args.server = str(Path(args.server).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()

    if not Path(args.model).is_file():
        raise SystemExit(f"Model does not exist: {args.model}")
    if not Path(args.server).is_file():
        raise SystemExit(f"llama-server does not exist: {args.server}")
    print("Hashing exact model artifact...", flush=True)
    args.model_sha256 = file_sha256(args.model)

    caps = detect_server_capabilities(args.server)
    print(
        f"llama-server flags: {len(caps.flags)} supported; spec types: "
        f"{sorted(caps.spec_types) or 'none'}",
        flush=True,
    )

    modes = ["baseline", "mtp"] if args.mode == "both" else [args.mode]
    suite = {
        "schema_version": 1,
        "model": args.model,
        "model_size_bytes": os.path.getsize(args.model),
        "arguments": vars(args),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    for mode in modes:
        mode_path = output_path.with_name(f"{output_path.stem}-{mode}.json")
        run = run_mode(args, mode, mode_path, caps)
        suite["runs"].append(run)
        checkpoint(output_path, suite)
    suite["finished_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint(output_path, suite)
    print(json.dumps({"output": str(output_path), "runs": len(suite["runs"])}, indent=2))


if __name__ == "__main__":
    main()
