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
                decoded = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaError(f"Could not reach Ollama: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON") from exc
        if "error" in decoded:
            raise OllamaError(str(decoded["error"]))
        return OllamaGeneration(
            response=decoded.get("response") or "",
            thinking=decoded.get("thinking") or "",
            prompt_tokens=int(decoded.get("prompt_eval_count") or 0),
            prompt_duration_ns=int(decoded.get("prompt_eval_duration") or 0),
            generated_tokens=int(decoded.get("eval_count") or 0),
            generation_duration_ns=int(decoded.get("eval_duration") or 0),
            total_duration_ns=int(decoded.get("total_duration") or 0),
        )
