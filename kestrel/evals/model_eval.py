"""Bounded, deterministic evaluation against an OpenAI-compatible server.

The evaluator deliberately does not launch a model.  Kestrel (or the user)
owns the llama-server lifecycle; this module sends a small fixed manifest to
an already-running endpoint and emits an auditable JSON report.  This keeps
quality tests separate from capacity/startup tests and makes the same test
set usable with a real 122B model or a tiny fake HTTP transport in CI.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
DEFAULT_SEED = 42


@dataclass(frozen=True)
class EvalCase:
    """One deterministic prompt and its stable output contract."""

    case_id: str
    category: str
    prompt: str
    mode: str = "off"
    max_tokens: int = 128
    validator: str = "nonempty"
    expected: Any = None

    def request(self, model: str, seed: int) -> dict[str, Any]:
        """Build a Qwen-compatible chat-completions request."""
        return {
            "model": model,
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "max_tokens": self.max_tokens,
            "stream": False,
            # Qwen3.5 uses this through the llama.cpp chat template.  Keeping
            # it explicit makes thinking-on/off runs comparable and visible.
            "chat_template_kwargs": {"enable_thinking": self.mode == "on"},
        }


@dataclass(frozen=True)
class EvalManifest:
    """Named evaluation set with explicit capacity and quality gates."""

    name: str
    version: str
    cases: tuple[EvalCase, ...]
    capacity_gate: str = "server_ready_and_all_cases_completed"
    quality_gate: str = "every_case_passes_and_no_repetition_or_malformed_output"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capacity_gate": self.capacity_gate,
            "quality_gate": self.quality_gate,
            "cases": [asdict(case) for case in self.cases],
        }


DEFAULT_MANIFEST = EvalManifest(
    name="qwen-frontier-edge",
    version="1",
    cases=(
        EvalCase(
            "arithmetic_vat",
            "arithmetic",
            "A product costs £250. Apply a 20% discount, then add 20% VAT to the discounted price. Reply with only the final number and no currency symbol.",
            expected="240",
            validator="exact_token",
        ),
        EvalCase(
            "instruction_reverse",
            "instruction",
            "Reverse these words and join them with |. Reply with only the result: red green blue",
            expected="blue|green|red",
            validator="exact_token",
        ),
        EvalCase(
            "strict_json",
            "structured_output",
            'Return exactly one JSON object and no markdown. It must have the boolean key "ok" set to true and numeric key "answer" set to 42.',
            expected={"ok": True, "answer": 42},
            validator="strict_json",
        ),
        EvalCase(
            "python_odd_squares",
            "code",
            "Write a Python expression that sums the squares of odd numbers below 8. Reply with only executable code.",
            expected=84,
            validator="code_result",
        ),
        EvalCase(
            "syllogism",
            "reasoning",
            "All glims are blue. No blue things are red. Mira is a glim. Is Mira red? Answer only yes or no.",
            expected="no",
            validator="exact_token",
        ),
        EvalCase(
            "thinking_off_short",
            "qwen_mode",
            "What is 17 + 25? Give only the integer answer.",
            mode="off",
            max_tokens=64,
            expected="42",
            validator="exact_token",
        ),
        EvalCase(
            "thinking_on_reasoning",
            "qwen_mode",
            "A train travels 120 km at 60 km/h and then 180 km at 90 km/h. How many hours is the total travel time? Give the final answer as a number.",
            mode="on",
            max_tokens=256,
            expected="4",
            validator="numeric_final",
        ),
        EvalCase(
            "bounded_list",
            "stability",
            "List the three additive primary colors separated by commas. Do not repeat any item and do not add commentary.",
            max_tokens=64,
            validator="three_unique_items",
            expected=("red", "green", "blue"),
        ),
        EvalCase(
            "well_formed_json",
            "stability",
            'Return exactly {"items":[1,2,3]} as JSON, with no code fence or explanation.',
            max_tokens=64,
            expected={"items": [1, 2, 3]},
            validator="strict_json",
        ),
    ),
)


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def _has_repetition(text: str) -> bool:
    """Detect obvious runaway repetition, not legitimate repeated words."""
    tokens = re.findall(r"\S+", text.lower())
    if len(tokens) >= 12 and len(set(tokens)) / len(tokens) < 0.35:
        return True
    for width in (3, 4, 5):
        if (
            len(tokens) >= width * 3
            and tokens[-width:] == tokens[-2 * width : -width] == tokens[-3 * width : -2 * width]
        ):
            return True
    return False


def _safe_code_result(expression: str) -> int | float:
    """Evaluate a tiny arithmetic generator expression after strict AST checks."""

    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    allowed = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.GeneratorExp,
        ast.comprehension,
        ast.BinOp,
        ast.Mult,
        ast.Pow,
        ast.Mod,
        ast.Compare,
        ast.Eq,
        ast.Constant,
    )
    if len(nodes) > 40 or any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("unsupported code expression")
    generators = [node for node in nodes if isinstance(node, ast.GeneratorExp)]
    if len(generators) != 1 or len(generators[0].generators) != 1:
        raise ValueError("expected one bounded generator")
    for node in nodes:
        if isinstance(node, ast.Name) and node.id not in {"x", "sum", "range"}:
            raise ValueError("unsupported name")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in {"sum", "range"} or node.keywords:
                raise ValueError("unsupported call")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int) or abs(node.value) > 100:
                raise ValueError("unsupported constant")
    value = eval(compile(tree, "<model-eval>", "eval"), {"__builtins__": {}, "sum": sum, "range": range}, {})
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expression did not produce a number")
    return value


def evaluate_case_output(case: EvalCase, output: str) -> dict[str, Any]:
    """Evaluate one response without an LLM judge or network dependency."""
    had_code_fence = "```" in output
    cleaned = _clean_text(output)
    result: dict[str, Any] = {"passed": False, "validator": case.validator, "normalized": cleaned}
    if not cleaned:
        result["reason"] = "empty_output"
        return result
    if _has_repetition(cleaned):
        result["reason"] = "repetition_detected"
        return result
    try:
        if case.validator == "exact_token":
            result["passed"] = cleaned.casefold() == str(case.expected).casefold()
        elif case.validator == "numeric_final":
            numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", cleaned)
            result["passed"] = bool(numbers) and numbers[-1] == str(case.expected)
        elif case.validator == "strict_json":
            if had_code_fence:
                result["reason"] = "markdown_code_fence"
                return result
            result["parsed"] = json.loads(cleaned)
            result["passed"] = result["parsed"] == case.expected
        elif case.validator == "code_result":
            result["actual_result"] = _safe_code_result(cleaned)
            result["passed"] = result["actual_result"] == case.expected
            result["expected_result"] = case.expected
        elif case.validator == "three_unique_items":
            items = [item.strip().casefold() for item in cleaned.split(",")]
            expected = {str(item).casefold() for item in case.expected or ()}
            result["passed"] = len(items) == 3 and len(set(items)) == 3 and set(items) == expected
            result["items"] = items
        else:
            result["reason"] = f"unknown_validator:{case.validator}"
            return result
    except (SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["reason"] = f"malformed_output:{type(exc).__name__}"
    if not result["passed"] and "reason" not in result:
        result["reason"] = "expectation_mismatch"
    return result


def _request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="POST" if payload is not None else "GET")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
        body = json.dumps(payload).encode("utf-8")
    else:
        body = None
    try:
        with urllib.request.urlopen(request, data=body, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc


def _artifact_metadata(artifact: str | Path | None, include_sha256: bool = False) -> dict[str, Any] | None:
    if not artifact:
        return None
    path = Path(artifact).expanduser().resolve()
    metadata: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        metadata.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if include_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            metadata["sha256"] = digest.hexdigest()
    return metadata


def _hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count() or 0,
    }


def _content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("missing choices[0]")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("missing choices[0].message.content")
    return message["content"]


def run_model_evaluation(
    endpoint: str = "http://127.0.0.1:8080",
    model: str | None = None,
    *,
    manifest: EvalManifest = DEFAULT_MANIFEST,
    seed: int = DEFAULT_SEED,
    timeout: float = 120.0,
    max_cases: int | None = None,
    artifact: str | Path | None = None,
    include_sha256: bool = False,
    request_json: Callable[[str, dict[str, Any] | None, float], dict[str, Any]] = _request_json,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the fixed manifest against a running llama-server endpoint."""
    base = endpoint.rstrip("/")
    cases = manifest.cases[:max_cases] if max_cases else manifest.cases
    selection_complete = len(cases) == len(manifest.cases)
    started = clock()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "started_at_unix": time.time(),
        "endpoint": base,
        "seed": seed,
        "manifest": manifest.as_dict(),
        "selection": {
            "complete": selection_complete,
            "selected_case_ids": [case.case_id for case in cases],
            "selected_count": len(cases),
            "manifest_count": len(manifest.cases),
        },
        "artifact": _artifact_metadata(artifact, include_sha256),
        "hardware": _hardware_metadata(),
        "engine": {"name": "llama.cpp", "model": model},
        "capacity_gate": {"passed": False, "checks": []},
        "quality_gate": {"passed": False, "checks": []},
        "cases": [],
    }
    try:
        health = request_json(f"{base}/health", None, timeout)
        report["capacity_gate"]["checks"].append({"name": "health", "passed": health.get("status") == "ok"})
        models = request_json(f"{base}/v1/models", None, timeout)
        data = models.get("data") if isinstance(models, dict) else None
        model_ids = {
            item.get("id") for item in data or () if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if model is None and isinstance(data, list) and data and isinstance(data[0], dict):
            model = data[0].get("id")
            report["engine"]["model"] = model
        report["capacity_gate"]["checks"].append(
            {"name": "models_endpoint", "passed": isinstance(data, list) and bool(data)}
        )
        report["capacity_gate"]["checks"].append(
            {"name": "requested_model", "passed": isinstance(model, str) and model in model_ids}
        )
    except RuntimeError as exc:
        report["capacity_gate"]["checks"].append({"name": "server_request", "passed": False, "reason": str(exc)})
        report["error"] = str(exc)
        report["elapsed_seconds"] = round(clock() - started, 6)
        return report

    for case in cases:
        case_started = clock()
        row: dict[str, Any] = {"id": case.case_id, "category": case.category, "mode": case.mode, "seed": seed}
        try:
            response = request_json(f"{base}/v1/chat/completions", case.request(model or "unknown", seed), timeout)
            output = _content(response)
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            row["output"] = output
            row["evaluation"] = evaluate_case_output(case, output)
            row["finish_reason"] = finish_reason
            if finish_reason == "length":
                row["evaluation"] = {
                    "passed": False,
                    "validator": case.validator,
                    "normalized": _clean_text(output),
                    "reason": "truncated_at_token_limit",
                }
            row["usage"] = response.get("usage") if isinstance(response, dict) else None
            row["request_ok"] = True
        except (RuntimeError, ValueError, TypeError) as exc:
            row.update(
                {
                    "request_ok": False,
                    "error": str(exc),
                    "evaluation": {"passed": False, "reason": "request_or_schema_error"},
                }
            )
        row["elapsed_seconds"] = round(clock() - case_started, 6)
        report["cases"].append(row)

    capacity_passed = (
        all(check.get("passed") for check in report["capacity_gate"]["checks"])
        and len(report["cases"]) == len(cases)
        and all(row.get("request_ok") for row in report["cases"])
    )
    selected_quality_passed = bool(cases) and all(row.get("evaluation", {}).get("passed") for row in report["cases"])
    quality_passed = capacity_passed and selection_complete and selected_quality_passed
    report["capacity_gate"]["passed"] = capacity_passed
    report["quality_gate"]["selected_cases_passed"] = selected_quality_passed
    report["quality_gate"]["complete_manifest"] = selection_complete
    report["quality_gate"]["passed"] = quality_passed
    report["elapsed_seconds"] = round(clock() - started, 6)
    report["status"] = "passed" if quality_passed else "failed"
    return report
