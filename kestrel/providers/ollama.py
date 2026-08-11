from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class OllamaError(RuntimeError):
    """An Ollama configuration, transport, or response error."""


@dataclass(frozen=True)
class OllamaGeneration:
    response: str
    thinking: str
    prompt_tokens: int
    prompt_duration_ns: int
    generated_tokens: int
    generation_duration_ns: int
    total_duration_ns: int

    @property
    def prompt_tps(self) -> float | None:
        if not self.prompt_duration_ns:
            return None
        return self.prompt_tokens * 1e9 / self.prompt_duration_ns

    @property
    def decode_tps(self) -> float | None:
        if not self.generation_duration_ns:
            return None
        return self.generated_tokens * 1e9 / self.generation_duration_ns


class OllamaClient:
    _MAX_RESPONSE_BYTES = 16 * 1024 * 1024
    _MAX_METRIC_VALUE = (1 << 63) - 1

    def __init__(self, *, base_url: str | None = None, timeout: float = 600.0, opener=None):
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            self.base_url = "http://" + self.base_url
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        num_predict: int = 64,
        num_ctx: int = 2048,
        seed: int = 42,
    ) -> OllamaGeneration:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "seed": seed,
                "temperature": 0,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "kestrel-cli"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read(self._MAX_RESPONSE_BYTES + 1)
                if len(body) > self._MAX_RESPONSE_BYTES:
                    raise OllamaError("Ollama response exceeded the 16 MiB safety limit")
                decoded = json.loads(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read(1001)[:1000].decode(errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaError(f"Could not reach Ollama: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise OllamaError("Ollama returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise OllamaError("Ollama returned an unexpected JSON response")
        if "error" in decoded:
            error = decoded["error"]
            if not isinstance(error, str):
                raise OllamaError("Ollama returned an invalid error response")
            raise OllamaError(error[:1000])
        text_fields = ("response", "thinking")
        if any(field in decoded and not isinstance(decoded[field], str) for field in text_fields):
            raise OllamaError("Ollama returned invalid generation text")
        numeric_fields = (
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "total_duration",
        )
        metrics = {field: decoded.get(field, 0) for field in numeric_fields}
        if any(type(value) is not int or not 0 <= value <= self._MAX_METRIC_VALUE for value in metrics.values()):
            raise OllamaError("Ollama returned invalid generation metrics")
        return OllamaGeneration(
            response=decoded.get("response") or "",
            thinking=decoded.get("thinking") or "",
            prompt_tokens=metrics["prompt_eval_count"],
            prompt_duration_ns=metrics["prompt_eval_duration"],
            generated_tokens=metrics["eval_count"],
            generation_duration_ns=metrics["eval_duration"],
            total_duration_ns=metrics["total_duration"],
        )
