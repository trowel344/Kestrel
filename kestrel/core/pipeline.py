from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from ..backends.llama_cpp import LlamaCppBackend
from ..model_store import hf_snapshot_dir


def _resolve_model_dir(path: str) -> str:
    root = Path(path).expanduser()
    snapshot = hf_snapshot_dir(root, marker="model.safetensors.index.json")
    # The pipeline falls back to the raw directory name so a plain model dir
    # (not an HF cache tree) still resolves as-is.
    return str(snapshot) if snapshot is not None else str(root)


class InferencePipeline:
    """A small, truthful wrapper around one llama.cpp generation process.

    Experimental Python router/cache components are intentionally not created
    here: they cannot alter llama.cpp's internal tensor residency and only add
    CUDA allocations that increase OOM risk.
    """

    def __init__(
        self,
        model_dir: str = "",
        gguf_path: str = "",
        n_gpu_layers: int | str = "auto",
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_ubatch: int = 128,
        spec_type: str = "none",
        spec_draft_n: int = 3,
        cpu_moe: bool = False,
        fit_target_mib: int = 1024,
        cache_type: str = "q8_0",
        n_threads: int = 0,
        llama_cpp_dir: str | None = None,
    ):
        self.model_dir = _resolve_model_dir(model_dir) if model_dir else ""
        self.gguf_path = gguf_path or self._discover_gguf()
        self.backend = LlamaCppBackend(
            model_path=self.gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            spec_type=spec_type,
            spec_draft_n=spec_draft_n,
            cpu_moe=cpu_moe,
            fit=True,
            fit_target_mib=fit_target_mib,
            cache_type_k=cache_type,
            cache_type_v=cache_type,
            use_mmap=True,
            n_threads=n_threads,
            llama_cpp_dir=llama_cpp_dir,
        )
        self._strategy = "mtp" if self.backend._resolved_spec_type(self.backend.capabilities()) else "none"

    def _discover_gguf(self) -> str:
        if not self.model_dir:
            return ""
        directory = self.model_dir.rstrip(os.sep)
        beside_directory = directory + ".gguf"
        inside_directory = os.path.join(directory, "model.gguf")
        for candidate in (inside_directory, beside_directory):
            if os.path.isfile(candidate):
                return candidate
        # Fall back to any GGUF in the directory (supported by llama.cpp,
        # including the first shard of a split model, whose sibling shards the
        # loader resolves automatically).
        try:
            candidates = sorted(Path(directory).glob("*.gguf"))
        except OSError:
            candidates = []
        if candidates:
            return str(candidates[0])
        return beside_directory

    def generate(self, prompt: str, max_tokens: int = 256) -> tuple[str, dict]:
        text = self.backend.generate(prompt, max_tokens=max_tokens)
        summary = self.backend.last_metrics.as_dict()
        summary["strategy"] = self._strategy
        summary["model"] = self.gguf_path
        return text, summary

    def generate_stream(self, prompt: str, max_tokens: int = 256) -> Iterator[str]:
        yield from self.backend.generate_stream(prompt, max_tokens=max_tokens)

    def close(self):
        self.backend.close()
