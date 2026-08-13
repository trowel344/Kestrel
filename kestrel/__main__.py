"""Allow python -m kestrel to use the installed CLI entry point."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
