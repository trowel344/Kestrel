from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable

DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k3"


class KimiError(RuntimeError):
    """A configuration, transport, or API error from Moonshot/Kimi."""


@dataclass(frozen=True)
class KimiResponse:
    content: str
    reasoning_content: str
    usage: dict
    raw_message: dict


class KimiClient:
    """Dependency-free client for Moonshot's OpenAI-compatible Kimi API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
        opener: Callable[..., object] | None = None,
    ):
        self.api_key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get(
            "MOONSHOT_API_KEY"
        )
        self.base_url = (
            base_url or os.environ.get("KIMI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("KIMI_MODEL_NAME") or DEFAULT_MODEL
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        messages: Iterable[dict],
        *,
        reasoning_effort: str = "high",
        max_tokens: int = 4096,
    ) -> KimiResponse:
        if not self.api_key:
            raise KimiError(
                "Kimi API key is missing. Set KIMI_API_KEY or MOONSHOT_API_KEY."
            )
        payload = {
            "model": self.model,
            "messages": list(messages),
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "kestrel-cli",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise KimiError(f"Kimi API returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise KimiError(f"Could not reach Kimi API: {exc}") from exc
        try:
            decoded = json.loads(body)
            message = decoded["choices"][0]["message"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise KimiError("Kimi API returned an invalid chat-completion response") from exc
        return KimiResponse(
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or "",
            usage=decoded.get("usage") or {},
            raw_message=dict(message),
        )
