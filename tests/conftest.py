"""Set up test environment: keep kestrel tests hermetic and hermetic-safe."""

import os
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    # Never touch a real user config or home during tests.
    os.environ.setdefault("KESTREL_CONFIG", str(ROOT / "tests" / ".testconfig.toml"))
    os.environ.setdefault("KESTREL_MODELS_DIR", "/tmp/kestrel-test-models")


def pytest_unconfigure(config):
    path = ROOT / "tests" / ".testconfig.toml"
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path):
    """Route every on-disk cache (metadata planner cache, capability probes)
    to a throwaway directory so tests never write to the real user cache and
    cannot observe stale entries across test runs."""
    os.environ["KESTREL_CACHE_DIR"] = str(tmp_path / "cache")
    yield
    os.environ.pop("KESTREL_CACHE_DIR", None)
