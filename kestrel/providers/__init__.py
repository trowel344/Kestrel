"""Remote model providers used by the Kestrel CLI."""

from .ollama import OllamaClient, OllamaError

__all__ = ["OllamaClient", "OllamaError"]
