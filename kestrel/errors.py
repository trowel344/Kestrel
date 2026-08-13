"""Structured, machine-readable failure taxonomy for Kestrel.

Every operation that can fail surfaces a :class:`KestrelError` (or subclass)
carrying a stable ``code``, a human ``message``, and an optional actionable
``hint``. The CLI renders these the same way everywhere and, under ``--json``,
as a stable ``{"error": {"code", "message", "hint"}}`` document. Downstream
tools and the bracket-format prompts can key off ``code`` without parsing prose.
"""

from __future__ import annotations


class KestrelError(Exception):
    """Base class for all Kestrel failures.

    Subclasses set ``exit_code`` and ``code``; failures can also provide a
    per-instance ``hint`` recommending the next action.
    """

    exit_code: int = 1
    code: str = "error"

    def __init__(
        self,
        message: str | None = None,
        *,
        hint: str | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message if message is not None else (self.__doc__ or self.code)
        self.hint = hint
        if code is not None:
            self.code = code
        super().__init__(self.message)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }


class ConfigError(KestrelError):
    code = "config_error"
    exit_code = 2


class InputError(KestrelError):
    """A command argument combination is invalid or incomplete."""

    code = "invalid_input"
    exit_code = 2


class ConversionError(KestrelError):
    """A model conversion could not be completed safely."""

    code = "conversion_error"


class ModelError(KestrelError):
    """A model could not be located, parsed, or produced."""

    code = "model_error"


class MissingModelError(ModelError):
    """The requested model is not present locally and cannot be streamed."""

    code = "model_not_found"


class CorruptModelError(ModelError):
    """A model file is truncated or its GGUF headers are invalid."""

    code = "model_corrupt"


class IntegrityError(KestrelError):
    """A checksum/verification step failed; the artifact must not be used."""

    code = "integrity_error"


class BackendError(KestrelError, RuntimeError):
    """The llama.cpp backend could not be located or launched.

    Also a :class:`RuntimeError` so existing ``except RuntimeError`` callers
    keep working while new code can catch the structured subtype.
    """

    code = "backend_error"


class EngineError(KestrelError):
    """An engine checkout could not be updated or rebuilt safely."""

    code = "engine_error"


class ServiceError(KestrelError):
    """The inference server failed to start or became unhealthy."""

    code = "service_error"


class IntegrationError(KestrelError):
    """A coding-agent integration could not be configured or verified."""

    code = "integration_error"


__all__ = [
    "KestrelError",
    "ConfigError",
    "InputError",
    "ConversionError",
    "ModelError",
    "MissingModelError",
    "CorruptModelError",
    "IntegrityError",
    "BackendError",
    "EngineError",
    "ServiceError",
    "IntegrationError",
]
