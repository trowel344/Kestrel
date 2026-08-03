"""Optional remote model providers used by the Kestrel CLI."""

from .kimi import KimiClient, KimiError

__all__ = ["KimiClient", "KimiError"]
