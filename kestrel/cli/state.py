"""Process-lifetime CLI configuration snapshot.

``USER_CONFIG`` is captured when the module first loads. Interactive flows that
mutate the on-disk config (``setup``, the menu's launch paths) call
:func:`reload_state` so a same-process session sees freshly-saved settings
instead of the stale import-time snapshot.
"""

from __future__ import annotations

from ..backends.llama_cpp import default_llama_cpp_dir
from ..config import KestrelConfig, load_config
from ..errors import ConfigError

USER_CONFIG = None
CONFIG_ERROR: str | None = None
LLAMA_CPP_DIR = ""
MODEL_ALIASES = {
    "qwen3.5:122b-10a": (
        "KESTREL_QWEN35_122B_GGUF",
        "~/.local/share/kestrel/models/qwen3.5-122b-a10b-nvfp4.gguf",
    ),
}


def reload_state() -> None:
    """Re-read the on-disk config and refresh every derived module global.

    ``setup`` and the interactive menu re-enter the dispatch loop in the same
    process; without a reload they would keep planning with the config that
    was on disk at import time.
    """
    global USER_CONFIG, CONFIG_ERROR, LLAMA_CPP_DIR
    try:
        config = load_config()
    except ConfigError as exc:
        USER_CONFIG = KestrelConfig()
        CONFIG_ERROR = str(exc)
    else:
        USER_CONFIG = config
        CONFIG_ERROR = None
    LLAMA_CPP_DIR = USER_CONFIG.llama_cpp_dir or default_llama_cpp_dir()


reload_state()
