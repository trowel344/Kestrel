"""Kestrel's supported, lightweight public API.

Only the llama.cpp runtime wrapper and the validated converter are exported.
Historical Python inference, expert-cache, router, and speculation prototypes
were removed from the repository before release and remain recoverable via git
history.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "InferencePipeline": (".core.pipeline", "InferencePipeline"),
    "LlamaCppBackend": (".backends.llama_cpp", "LlamaCppBackend"),
    "NVFP4Converter": (".gguf.converter", "NVFP4Converter"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__version__ = "1.6.0"
