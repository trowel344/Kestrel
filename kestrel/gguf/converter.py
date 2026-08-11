import json
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import util
from .quants import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_Q1_0,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGUF_SCALAR_FORMATS,
    ImportanceMatrix,
    _quantize_q1_0_buffer,
    _quantize_q4_0_buffer,
    dequantize_nvfp4,
    dequantize_q4_0,
    ggml_type_block_size,
    ggml_type_size,
    quantize_iq1_s,
    quantize_q2_k,
    quantize_q3_k,
    quantize_q4_0,
    round_up,
)

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# torch and safetensors are the optional [convert] extra. A bare ``pip
# install kestrel`` must still be able to ``import kestrel.gguf.converter``
# (cli.py imports this module for the generic/doctor paths), so the two heavy
# imports live inside a guarded module-level block. Anything that actually
# opens a shard fails with a clear message via ``_require_convert_deps``.
try:
    import torch
    from safetensors import safe_open
except ImportError:  # pragma: no cover - exercised only on a bare install
    torch = None
    safe_open = None


def _require_convert_deps() -> None:
    """Raise a clear error when the optional torch/safetensors imports are gone."""
    if torch is None or safe_open is None:
        raise ImportError("NVFP4 model conversion requires torch and safetensors; install the 'kestrel[convert]' extra")


MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Qwen3.5-122B-A10B-NVFP4")


def write_gguf_string(f, s):
    data = s.encode("utf-8")
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def write_gguf_kv(f, key, vtype, val):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", vtype))
    fmt = GGUF_SCALAR_FORMATS.get(vtype)
    if fmt is not None:
        # The shared scalar table is byte-order-free; the writer is fixed little-endian.
        f.write(struct.pack("<" + fmt, val))
        return
    if vtype == 8:  # string
        write_gguf_string(f, val)
        return
    if vtype == 9:  # array
        f.write(struct.pack("<I", val[0]))
        f.write(struct.pack("<Q", len(val[1])))
        for item in val[1]:
            if val[0] == 8:
                write_gguf_string(f, item)
            elif val[0] == 4:
                f.write(struct.pack("<I", item))


def write_tensor_info(f, name, n_dims, dims, dtype, offset):
    write_gguf_string(f, name)
    f.write(struct.pack("<I", n_dims))
    for d in dims:
        f.write(struct.pack("<Q", d))
    f.write(struct.pack("<I", dtype))
    f.write(struct.pack("<Q", offset))


def tensor_data_layout(tensor_infos, alignment=32):
    """Return validated ``(tensor_info, data_offset)`` entries for a GGUF catalog.

    Tensor-info records carry their payload offset relative to the start of the
    data section.  Keeping that calculation in a pure helper makes the on-disk
    layout easy to verify without opening a model, and rejects malformed
    catalogs before a partially written artifact can be produced.
    """
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("tensor alignment must be a positive power of two")
    offset = 0
    layout = []
    for name, n_dims, dims, dtype, nbytes in tensor_infos:
        if n_dims != len(dims):
            raise ValueError(f"tensor {name} declares {n_dims} dimensions but has {len(dims)}")
        if any(dim < 0 for dim in dims):
            raise ValueError(f"tensor {name} has a negative dimension")
        if nbytes < 0:
            raise ValueError(f"tensor {name} has a negative payload size")
        element_count = math.prod(dims)
        try:
            block_size = ggml_type_block_size(dtype)
            block_bytes = ggml_type_size(dtype)
        except KeyError as exc:
            raise ValueError(f"tensor {name} has unsupported GGML type {dtype}") from exc
        if dims and dims[0] % block_size:
            raise ValueError(
                f"tensor {name} has row width {dims[0]}, which is not divisible by "
                f"the {block_size}-element GGML type-{dtype} block"
            )
        expected_nbytes = element_count // block_size * block_bytes
        if nbytes != expected_nbytes:
            raise ValueError(
                f"tensor {name} has payload size {nbytes}, expected {expected_nbytes} for its shape and type"
            )
        data_offset = round_up(offset, alignment)
        layout.append((name, n_dims, dims, dtype, data_offset))
        offset = data_offset + nbytes
    return layout, offset


LAYER_MAP_FULL = {
    "input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "post_attention_layernorm.weight": "blk.{i}.post_attention_norm.weight",
    "self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "self_attn.q_norm.weight": "blk.{i}.attn_q_norm.weight",
    "self_attn.k_norm.weight": "blk.{i}.attn_k_norm.weight",
    "mlp.gate.weight": "blk.{i}.ffn_gate_inp.weight",
    "mlp.shared_expert_gate.weight": "blk.{i}.ffn_gate_inp_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "blk.{i}.ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "blk.{i}.ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "blk.{i}.ffn_down_shexp.weight",
}

LAYER_MAP_LINEAR = {
    "input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "post_attention_layernorm.weight": "blk.{i}.post_attention_norm.weight",
    "linear_attn.in_proj_qkv.weight": "blk.{i}.attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "blk.{i}.attn_gate.weight",
    "linear_attn.conv1d.weight": "blk.{i}.ssm_conv1d.weight",
    "linear_attn.A_log": "blk.{i}.ssm_a",
    "linear_attn.dt_bias": "blk.{i}.ssm_dt.bias",
    "linear_attn.norm.weight": "blk.{i}.ssm_norm.weight",
    "linear_attn.out_proj.weight": "blk.{i}.ssm_out.weight",
    "linear_attn.in_proj_a.weight": "blk.{i}.ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "blk.{i}.ssm_beta.weight",
    "mlp.gate.weight": "blk.{i}.ffn_gate_inp.weight",
    "mlp.shared_expert_gate.weight": "blk.{i}.ffn_gate_inp_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "blk.{i}.ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "blk.{i}.ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "blk.{i}.ffn_down_shexp.weight",
}

MTP_MAP = {
    "input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "post_attention_layernorm.weight": "blk.{i}.post_attention_norm.weight",
    "self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "self_attn.q_norm.weight": "blk.{i}.attn_q_norm.weight",
    "self_attn.k_norm.weight": "blk.{i}.attn_k_norm.weight",
    "mlp.gate.weight": "blk.{i}.ffn_gate_inp.weight",
    "mlp.shared_expert_gate.weight": "blk.{i}.ffn_gate_inp_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "blk.{i}.ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "blk.{i}.ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "blk.{i}.ffn_down_shexp.weight",
}

MTP_NEXTN_MAP = {
    "eh_proj.weight": "blk.{i}.nextn.eh_proj.weight",
    "pre_fc_norm_embedding.weight": "blk.{i}.nextn.enorm.weight",
    "pre_fc_norm_hidden.weight": "blk.{i}.nextn.hnorm.weight",
}

MTP_NEXTN_HF = {
    "eh_proj.weight": "mtp.fc.weight",
    "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
}


_ROUTER_SUFFIX = "mlp.gate.weight"


def _select_kept_experts(n_exp: int, keep: int, importance: object = None) -> list[int]:
    """Return the source expert indices to emit when pruning to ``keep``.

    With ``importance`` (a length-``n_exp`` sequence of per-expert values, e.g.
    measured router frequency) the highest-valued experts are kept, ties broken
    by index for deterministic output. Without it the first ``keep`` experts are
    kept. The result is sorted ascending to keep emission order stable.
    """
    if importance is None:
        return list(range(keep))
    if len(importance) != n_exp:
        raise ValueError(f"expert importance has {len(importance)} entries, expected {n_exp}")
    picked = sorted(range(n_exp), key=lambda e: (-float(importance[e]), e))[:keep]
    return sorted(picked)


def _resolve_pruning(
    n_exp: int, n_used: int, experts_keep: int | None, importance: object = None
) -> tuple[list[int] | None, int]:
    """Validate and resolve expert pruning into ``(kept_indices, emitted_count)``.

    ``None`` kept means no pruning. Raises ``ValueError`` for counts that would
    either underflow the routing width or not actually prune anything.
    """
    if experts_keep is None:
        return None, n_exp
    if experts_keep < n_used:
        raise ValueError(f"experts_keep ({experts_keep}) cannot be smaller than experts used per token ({n_used})")
    if experts_keep >= n_exp:
        raise ValueError(f"experts_keep ({experts_keep}) must be smaller than the model's expert count ({n_exp})")
    kept = _select_kept_experts(n_exp, experts_keep, importance)
    return kept, len(kept)


class NVFP4Converter:
    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        *,
        include_mtp: bool = False,
        dense_q4: bool = False,
        cold_tier: str = "off",
        q4_sidecar_source: str | None = None,
        experts_only: bool = False,
        q2_edge_layers: int = 0,
        compact_expert_type: str = "q1_0",
        all_q2: bool = False,
        imatrix_path: str | None = None,
        conversion_workers: int | None = None,
        experts_keep: int | None = None,
        expert_importance: str | None = None,
    ):
        self.model_dir = model_dir
        # dense_q4: quantize dense matrices (attention/FFN/embedding) to Q4_0.
        # cold_tier: per-expert low-precision twin emitted alongside the Q4_0
        # expert ("q1_0" emits ffn_*_exps_lp Q1_0 tensors alongside Q4_0;
        # "q1_only" emits only the low-precision tensors for compact CUDA
        # scaling/streaming experiments; "off" emits only Q4_0 experts).
        # experts_keep: experimental expert pruning. When set (< num_experts),
        # only that many experts are emitted per layer and expert_count is
        # rewritten, producing a smaller GGUF that fits more in VRAM. Dropping
        # experts changes quality; this is an explicit opt-in, never default.
        # expert_importance: optional JSON list of n_experts per-expert
        # importance values used to choose which experts are kept (highest
        # kept). Absent, the first experts_keep are kept.
        self.dense_q4 = dense_q4
        self.q4_sidecar_source = q4_sidecar_source
        self.experts_only = experts_only
        self.q2_edge_layers = q2_edge_layers
        if all_q2:
            if q2_edge_layers:
                raise ValueError("all_q2 cannot be combined with q2_edge_layers")
            if compact_expert_type not in ("q1_0", "q2_k"):
                raise ValueError("all_q2 cannot be combined with another compact_expert_type")
            if cold_tier not in ("off", "q1_only"):
                raise ValueError("all_q2 requires cold_tier='q1_only'")
            cold_tier = "q1_only"
            compact_expert_type = "q2_k"
        self.cold_tier = cold_tier
        compact_types = {
            "q1_0": GGML_TYPE_Q1_0,
            "q2_k": GGML_TYPE_Q2_K,
            "q3_k": GGML_TYPE_Q3_K,
            "iq1_s": GGML_TYPE_IQ1_S,
        }
        if compact_expert_type not in compact_types:
            raise ValueError(f"unsupported compact_expert_type: {compact_expert_type}")
        self.compact_expert_type = compact_expert_type
        self.compact_expert_ggml_type = compact_types[compact_expert_type]
        self.imatrix_path = imatrix_path
        self.experts_keep = experts_keep
        self.expert_importance = expert_importance
        if experts_keep is not None and experts_keep < 1:
            raise ValueError("experts_keep must be at least 1")
        if experts_keep is not None and experts_only:
            raise ValueError("expert pruning cannot be combined with experts_only")
        if expert_importance and experts_keep is None:
            raise ValueError("expert_importance requires experts_keep")
        if expert_importance and not os.path.isfile(expert_importance):
            raise FileNotFoundError(f"expert importance file does not exist: {expert_importance}")
        self.conversion_workers = conversion_workers or min(8, os.cpu_count() or 1)
        if self.conversion_workers < 1:
            raise ValueError("conversion_workers must be at least 1")
        self.imatrix = None
        if compact_expert_type == "iq1_s" and not imatrix_path:
            raise ValueError("IQ1_S compact experts require an activation --imatrix")
        if compact_expert_type not in {"iq1_s", "q2_k", "q3_k"} and not q2_edge_layers and imatrix_path:
            raise ValueError("imatrix_path is supported for IQ1_S, Q2_K, and Q3_K experts")
        if compact_expert_type != "q1_0" and cold_tier != "q1_only":
            raise ValueError("non-Q1 compact experts require cold_tier='q1_only'")
        if compact_expert_type != "q1_0" and q4_sidecar_source:
            raise ValueError("non-Q1 compact experts require direct NVFP4 source conversion")
        if compact_expert_type != "q1_0" and experts_only:
            raise ValueError("non-Q1 compact experts cannot be emitted as a cold sidecar")
        if q2_edge_layers < 0:
            raise ValueError("q2_edge_layers cannot be negative")
        if q2_edge_layers and cold_tier != "q1_only":
            raise ValueError("q2_edge_layers requires cold_tier='q1_only'")
        if q2_edge_layers and q4_sidecar_source:
            raise ValueError("q2_edge_layers requires direct NVFP4 source conversion")
        if q4_sidecar_source and cold_tier != "q1_only":
            raise ValueError("q4_sidecar_source requires cold_tier='q1_only'")
        if experts_only and cold_tier != "q1_only":
            raise ValueError("experts_only requires cold_tier='q1_only'")
        if experts_only and include_mtp:
            raise ValueError("experts_only cannot include MTP")
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        self.tcfg = cfg.get("text_config", cfg)
        self.layer_types = self.tcfg["layer_types"]
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.idx = json.load(f)
        self.wm = self.idx["weight_map"]
        self.n_layer = self.tcfg["num_hidden_layers"]
        if self.q2_edge_layers * 2 > self.n_layer:
            raise ValueError("q2_edge_layers cannot cover more than half of the trunk layers")
        self.n_exp = self.tcfg["num_experts"]
        self.n_used = self.tcfg["num_experts_per_tok"]
        self._kept = None
        self._emitted_exp = self.n_exp
        if self.experts_keep is not None:
            importance = None
            if self.expert_importance:
                with open(self.expert_importance) as f:
                    importance = json.load(f)
            self._kept, self._emitted_exp = _resolve_pruning(self.n_exp, self.n_used, self.experts_keep, importance)
        self.hidden = self.tcfg["hidden_size"]
        self.n_ff = self.tcfg["moe_intermediate_size"]
        self.head_dim = self.tcfg["head_dim"]
        self.n_head = self.tcfg["num_attention_heads"]
        self.n_kv = self.tcfg["num_key_value_heads"]
        self.vocab = self.tcfg["vocab_size"]
        self.linear_n_k = self.tcfg.get("linear_num_key_heads", 0)
        self.linear_n_v = self.tcfg.get("linear_num_value_heads", 0)
        self.linear_k_dim = self.tcfg.get("linear_key_head_dim", 0)
        self.linear_v_dim = self.tcfg.get("linear_value_head_dim", 0)
        mtp_val = self.tcfg.get("mtp_num_hidden_layers", 0)
        if mtp_val is None:
            mtp_val = 0
        self.available_mtp_layers = mtp_val
        # MTP is a speculative draft head, not part of target-model quality.
        # It is omitted by default because it adds several GiB and measured
        # slower than target-only decoding on memory-constrained hardware.
        self.mtp_layers = mtp_val if include_mtp else 0
        self.shared_ff = self.tcfg.get("shared_expert_intermediate_size", 1024)
        if self.imatrix_path:
            self.imatrix = ImportanceMatrix(self.imatrix_path)
        # Cached open safetensors shards: a full conversion opens the same few
        # shards tens of thousands of times, and each `safe_open` re-parses the
        # shard's JSON header. Reusing the handle (and its parsed key set)
        # keeps header parsing to once per shard. The lock guards only the
        # handle-cache access; `get_tensor`/`get_slice` materialization happens
        # outside the lock so workers' large memcpys do not serialize behind a
        # single mutex. The handle is cached for the whole conversion, so the
        # mmap stays open while any worker reads from it.
        self._shard_handles: dict[str, object] = {}
        self._shard_keys: dict[str, set[str]] = {}
        self._shard_lock = threading.Lock()

    def _read_torch(self, key):
        shard = self.wm.get(key)
        if not shard:
            return None
        path = os.path.join(self.model_dir, shard)
        if not os.path.exists(path):
            return None
        # Reuse the cached shard handle so the safetensors JSON header is parsed
        # once per shard instead of re-opened for every dense read (a full-model
        # conversion can otherwise re-decode the same headers hundreds of times).
        sf, sf_keys = self._open_shard(shard, path)
        if key not in sf_keys:
            return None
        return sf.get_tensor(key)

    def _read_torch_slice(self, key):
        shard = self.wm.get(key)
        if not shard:
            return None
        path = os.path.join(self.model_dir, shard)
        if not os.path.exists(path):
            return None
        sf, sf_keys = self._open_shard(shard, path)
        if key not in sf_keys:
            return None
        return sf.get_slice(key)

    def _open_shard(self, shard: str, path: str):
        """Return ``(safe_open handle, set of tensor names)`` for a shard.

        Handles are cached per conversion so the shard's header is parsed once
        instead of on every expert read. The lock guards only the handle/key
        cache; callers do the actual ``get_tensor``/``get_slice`` reads outside
        the lock so concurrent workers do not serialize their big memcpys.
        """
        with self._shard_lock:
            _require_convert_deps()
            handle = self._shard_handles.get(shard)
            if handle is None:
                handle = safe_open(path, framework="pt")
                self._shard_handles[shard] = handle
                self._shard_keys[shard] = set(handle.keys())
            return handle, self._shard_keys[shard]

    def _read_nvfp4(self, base):
        """Read an NVFP4 projection, opening each shard file only once.

        The three tensors (``.weight``, ``.weight_scale``, ``.weight_scale_2``)
        usually live in the same shard. Grouping them by shard and opening each
        shard a single time avoids two redundant ``safe_open`` parses per
        projection across tens of thousands of expert reads."""
        keys = (f"{base}.weight", f"{base}.weight_scale", f"{base}.weight_scale_2")
        by_shard: dict[str, list[str]] = {}
        for key in keys:
            shard = self.wm.get(key)
            if not shard:
                return None
            by_shard.setdefault(shard, []).append(key)
        found: dict[str, object] = {}
        for shard, key_list in by_shard.items():
            path = os.path.join(self.model_dir, shard)
            if not os.path.exists(path):
                return None
            sf, sf_keys = self._open_shard(shard, path)
            for key in key_list:
                if key not in sf_keys:
                    return None
                found[key] = sf.get_tensor(key)
        w = found.get(f"{base}.weight")
        ws = found.get(f"{base}.weight_scale")
        ws2 = found.get(f"{base}.weight_scale_2")
        if w is None or ws is None or ws2 is None:
            return None
        return w.numpy(), ws.float().numpy(), float(ws2)

    def _gguf_shape(self, hf_shape):
        shape = list(hf_shape)
        if len(shape) == 2:
            shape.reverse()
        elif len(shape) == 3:
            shape = [shape[2], shape[0]]
        return shape

    @staticmethod
    def _reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
        """Match llama.cpp's tiled V-head order for hybrid linear attention."""
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        expanded = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1 :]
        tensor = tensor.reshape(*expanded)
        permutation = list(range(len(expanded)))
        permutation[dim], permutation[dim + 1] = (
            permutation[dim + 1],
            permutation[dim],
        )
        return tensor.permute(*permutation).contiguous().reshape(*shape)

    def _transform_linear_attention(self, tensor, hf_key):
        """Apply the source-to-llama.cpp V-head layout conversion."""
        if (
            getattr(self, "linear_n_k", 0) <= 0
            or getattr(self, "linear_n_v", 0) <= 0
            or self.linear_n_k == self.linear_n_v
            or ".linear_attn." not in hf_key
        ):
            return tensor

        n_per_k = self.linear_n_v // self.linear_n_k
        if ".in_proj_qkv." in hf_key:
            q_size = self.linear_k_dim * self.linear_n_k
            k_size = q_size
            q = tensor[:q_size]
            k = tensor[q_size : q_size + k_size]
            v = self._reorder_v_heads(
                tensor[q_size + k_size :],
                0,
                self.linear_n_k,
                n_per_k,
                self.linear_v_dim,
            )
            return torch.cat((q, k, v), dim=0)
        if ".in_proj_z." in hf_key:
            return self._reorder_v_heads(tensor, 0, self.linear_n_k, n_per_k, self.linear_v_dim)
        if ".in_proj_a." in hf_key or ".in_proj_b." in hf_key:
            return self._reorder_v_heads(tensor, 0, self.linear_n_k, n_per_k, 1)
        if ".A_log" in hf_key or ".dt_bias" in hf_key or ".dt_proj" in hf_key:
            if tensor.dim() == 1:
                return self._reorder_v_heads(tensor.unsqueeze(-1), 0, self.linear_n_k, n_per_k, 1).squeeze(-1)
            return self._reorder_v_heads(tensor, -1, self.linear_n_k, n_per_k, 1)
        if ".conv1d" in hf_key:
            data = tensor.squeeze()
            qk_channels = self.linear_k_dim * self.linear_n_k * 2
            qk = data[:qk_channels]
            v = self._reorder_v_heads(
                data[qk_channels:],
                0,
                self.linear_n_k,
                n_per_k,
                self.linear_v_dim,
            )
            return torch.cat((qk, v), dim=0)
        if ".out_proj." in hf_key:
            return self._reorder_v_heads(tensor, 1, self.linear_n_k, n_per_k, self.linear_v_dim)
        return tensor

    def _pruned_router(self, tensor, hf_key):
        """Slice a MoE router to the kept experts under expert pruning.

        HF stores the router as ``mlp.gate.weight`` with shape ``(n_exp, n_embd)``:
        row ``e`` holds expert ``e``'s routing logits. Under ``experts_keep`` the
        expert tensors are reduced to ``self._kept`` and ``expert_count`` is
        rewritten, so the router MUST drop the same experts in the same order or
        llama.cpp routes to experts that no longer exist (out-of-bounds reads /
        silently broken mixture). Slicing columns is equivalent to restricting
        top-k to the kept pool; no renormalization is required for top-k routing.
        """
        if self._kept is None or not hf_key.endswith(_ROUTER_SUFFIX):
            return tensor
        if tensor.dim() < 2 or tensor.shape[0] != self.n_exp:
            raise ValueError(f"router {hf_key} has shape {tuple(tensor.shape)}, expected ({self.n_exp}, ...)")
        return tensor[self._kept]

    def _transform_source_tensor(self, tensor, hf_key):
        """Apply all Qwen3.5 source-to-runtime tensor transformations."""
        tensor = self._pruned_router(tensor, hf_key)
        if hf_key.endswith(".A_log"):
            tensor = -torch.exp(tensor)
        elif "conv1d" in hf_key:
            tensor = tensor.squeeze()
        elif hf_key.endswith("norm.weight") and not hf_key.endswith("linear_attn.norm.weight"):
            # Qwen3.5 stores RMSNorm offsets around zero and applies 1 + weight.
            # GGML RMSNorm expects the effective multiplicative weight.
            tensor = tensor + 1
        return self._transform_linear_attention(tensor, hf_key)

    def _emit_experts(self):
        """Iterate the source expert indices this conversion emits.

        ``range(n_exp)`` normally; with expert pruning, only the kept indices.
        The catalog, metadata, and every emission loop share this so the GGUF
        stays self-consistent under pruning.
        """
        return self._kept if self._kept is not None else range(self.n_exp)

    def _write_data_f16_experts(self, f, prefix, i):
        t0 = time.time()
        print(f"  Converting layer {i} F16 experts ({self.n_exp} experts)...")

        # MTP expert reads and fp16 casting are torch/NumPy ops that release
        # the GIL, so overlap them across workers like the Q4/Q1 kernels.
        def read_up_down(e):
            ep = f"{prefix}.mlp.experts.{e}"
            t_g = self._read_torch(f"{ep}.gate_proj.weight")
            t_u = self._read_torch(f"{ep}.up_proj.weight")
            if t_g is None or t_u is None:
                raise KeyError(f"Missing MTP expert gate/up tensors under {ep}")
            payload = t_g.contiguous().to(torch.float16).numpy().tobytes()
            payload += t_u.contiguous().to(torch.float16).numpy().tobytes()
            del t_g, t_u
            return payload

        def read_down(e):
            ep = f"{prefix}.mlp.experts.{e}"
            t = self._read_torch(f"{ep}.down_proj.weight")
            if t is None:
                raise KeyError(f"Missing MTP expert down tensor under {ep}")
            payload = t.contiguous().to(torch.float16).numpy().tobytes()
            del t
            return payload

        with ThreadPoolExecutor(max_workers=self.conversion_workers) as executor:
            self._write_windowed_map(f, executor, read_up_down)
            print(f"    gate/up {self.n_exp}/{self.n_exp} done", flush=True)
            self._pad_to_tensor(f, f"blk.{i}.ffn_down_exps.weight")
            self._write_windowed_map(f, executor, read_down)
            print(f"    down  {self.n_exp}/{self.n_exp} done ({time.time() - t0:.1f}s)", flush=True)

    def _read_bf16_info(self, hf_key):
        sl = self._read_torch_slice(hf_key)
        if sl is None:
            return None
        shape = self._gguf_shape(sl.get_shape())
        if self._kept is not None and hf_key.endswith(_ROUTER_SUFFIX):
            if len(shape) != 2 or shape[1] != self.n_exp:
                raise ValueError(
                    f"router {hf_key} has GGUF shape {shape if hasattr(shape, '__iter__') else list(shape)}, "
                    f"expected (n_embd, {self.n_exp})"
                )
            shape = (shape[0], self._emitted_exp)
        nbytes = int(np.prod(shape)) * 2
        return shape, nbytes

    def _load_tokenizer(self):
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        cfg_path = os.path.join(self.model_dir, "tokenizer_config.json")
        if not os.path.exists(tok_path):
            return None
        with open(tok_path) as f:
            tj = json.load(f)
        vocab = tj["model"]["vocab"]
        merges = tj["model"].get("merges", [])
        added_tokens = {}
        added_token_meta = {}
        for at in tj.get("added_tokens", []):
            added_tokens[at["content"]] = at["id"]
            added_token_meta[at["id"]] = at
        # The model intentionally reserves IDs above the actual tokenizer
        # vocabulary. Empty strings are not safe placeholders: without UNUSED
        # token types llama.cpp can sample them and detokenize them as '?'.
        tokens = [f"[PAD{i}]" for i in range(self.vocab)]
        token_types = [5] * self.vocab  # gguf.TokenType.UNUSED
        for token_str, token_id in vocab.items():
            if token_id < self.vocab:
                tokens[token_id] = token_str
                token_types[token_id] = 1  # gguf.TokenType.NORMAL
        for token_str, token_id in added_tokens.items():
            if token_id < self.vocab:
                tokens[token_id] = token_str
                meta = added_token_meta[token_id]
                looks_special = token_str.startswith("<") and token_str.endswith(">")
                token_types[token_id] = 3 if meta.get("special") or looks_special else 4
        eos_id = None
        bos_id = None
        pad_id = None
        chat_template = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                tc = json.load(f)
            eos_str = tc.get("eos_token")
            bos_str = tc.get("bos_token")
            pad_str = tc.get("pad_token")
            chat_template = tc.get("chat_template")
            if eos_str:
                eos_id = added_tokens.get(eos_str, vocab.get(eos_str))
            if bos_str:
                bos_id = added_tokens.get(bos_str, vocab.get(bos_str))
            if pad_str:
                pad_id = added_tokens.get(pad_str, vocab.get(pad_str))
        if eos_id is None:
            eos_id = added_tokens.get("<|im_end|>")
        return (
            tokens,
            token_types,
            merges,
            eos_id,
            bos_id,
            pad_id,
            chat_template,
        )

    def _init_gguf(self, f):
        cold_tier = getattr(self, "cold_tier", "off")
        emitted_exp = getattr(self, "_emitted_exp", getattr(self, "n_exp", 0))

        rope_params = self.tcfg.get("rope_parameters", {})
        rope_theta = rope_params.get("rope_theta", 10000000)
        partial_r = rope_params.get("partial_rotary_factor", 0.25)
        sections = rope_params.get("mrope_section", [11, 11, 10])
        full_attn_interval = self.tcfg.get("full_attention_interval", 4)

        kvs = {}

        def kv(key, vtype, val):
            kvs[key] = (vtype, val)

        kv("general.architecture", 8, "qwen35moe")
        total_blocks = self.n_layer + self.mtp_layers
        kv("qwen35moe.block_count", 4, total_blocks)
        kv("qwen35moe.context_length", 4, 262144)
        kv("qwen35moe.embedding_length", 4, self.hidden)
        kv("qwen35moe.feed_forward_length", 4, self.n_ff)
        kv("qwen35moe.expert_feed_forward_length", 4, self.n_ff)
        kv("qwen35moe.expert_shared_feed_forward_length", 4, self.shared_ff)
        kv("qwen35moe.attention.head_count", 4, self.n_head)
        kv("qwen35moe.attention.head_count_kv", 4, self.n_kv)
        kv("qwen35moe.expert_count", 4, emitted_exp)
        kv("qwen35moe.expert_used_count", 4, self.n_used)
        kv("qwen35moe.attention.layer_norm_rms_epsilon", 6, 1e-6)
        kv("qwen35moe.file_type", 4, 39)
        kv("qwen35moe.use_parallel_residual", 7, False)
        kv("qwen35moe.full_attention_interval", 4, full_attn_interval)
        kv("qwen35moe.ssm.conv_kernel", 4, 4)
        ssm_inner_size = self.tcfg.get("linear_value_head_dim", 128) * self.tcfg.get("linear_num_value_heads", 64)
        kv("qwen35moe.ssm.inner_size", 4, ssm_inner_size)
        kv("qwen35moe.ssm.state_size", 4, 128)
        kv("qwen35moe.ssm.time_step_rank", 4, 64)
        kv("qwen35moe.ssm.group_count", 4, 16)
        n_rot = int(self.head_dim * partial_r)
        kv("qwen35moe.rope.dimension_count", 4, n_rot)
        kv("qwen35moe.rope.freq_base", 6, float(rope_theta))
        sections_4 = sections + [n_rot // 2 - sum(sections)]
        kv("qwen35moe.rope.dimension_sections", 9, (4, sections_4))
        kv("qwen35moe.vocab_size", 4, self.vocab)
        kv("qwen35moe.attention.key_length", 4, self.head_dim)
        kv("qwen35moe.attention.value_length", 4, self.head_dim)

        tok_data = self._load_tokenizer()
        if tok_data:
            (
                tokens,
                token_types,
                merges,
                eos_id,
                bos_id,
                pad_id,
                chat_template,
            ) = tok_data
            kv("tokenizer.ggml.model", 8, "gpt2")
            kv("tokenizer.ggml.pre", 8, "qwen35")
            kv("tokenizer.ggml.tokens", 9, (8, tokens))
            kv("tokenizer.ggml.token_type", 9, (4, token_types))
            kv("tokenizer.ggml.merges", 9, (8, merges))
            if eos_id is not None:
                kv("tokenizer.ggml.eos_token_id", 4, eos_id)
            if bos_id is not None:
                kv("tokenizer.ggml.bos_token_id", 4, bos_id)
            if pad_id is not None:
                kv("tokenizer.ggml.padding_token_id", 4, pad_id)
            if chat_template:
                kv("tokenizer.chat_template", 8, chat_template)

        tensor_infos = []

        def add_ti(name, dims, dtype, nbytes):
            tensor_infos.append((name, len(dims), dims, dtype, nbytes))

        def add_dense_ti(name, shape, source_nbytes):
            # llama.cpp's CPU elementwise binary ops require their weight
            # vectors (norms, biases, SSM coefficients) as F32. Matrices are
            # supported as BF16 and should remain compact.
            if len(shape) == 1 or name.endswith(".ssm_conv1d.weight"):
                add_ti(name, shape, GGML_TYPE_F32, int(np.prod(shape)) * 4)
            elif self.dense_q4 and shape[0] % 32 == 0:
                n_blocks = shape[0] // 32
                nbytes = n_blocks * 18 * int(np.prod(shape[1:]))
                add_ti(name, shape, GGML_TYPE_Q4_0, nbytes)
            else:
                add_ti(name, shape, GGML_TYPE_BF16, source_nbytes)

        # Dense tensors — read actual shapes from the source model.
        bf16_keys = [
            ("token_embd.weight", "model.language_model.embed_tokens.weight"),
            ("output_norm.weight", "model.language_model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ]

        seen_tis = set()
        for gguf_name, hf_key in bf16_keys:
            info = self._read_bf16_info(hf_key)
            if info:
                shape, nbytes = info
                add_dense_ti(gguf_name, shape, nbytes)
                seen_tis.add(gguf_name)

        for i in range(total_blocks):
            if i >= self.n_layer:
                mtp_idx = i - self.n_layer
                prefix = f"mtp.layers.{mtp_idx}"
                kmap = MTP_MAP
                nextn_map = MTP_NEXTN_MAP
            else:
                lt = self.layer_types[i]
                kmap = LAYER_MAP_FULL if lt == "full_attention" else LAYER_MAP_LINEAR
                prefix = f"model.language_model.layers.{i}"
                nextn_map = {}
            for hf_suffix, gguf_tmpl in kmap.items():
                gguf_name = gguf_tmpl.format(i=i)
                if gguf_name in seen_tis:
                    continue
                info = self._read_bf16_info(f"{prefix}.{hf_suffix}")
                if info:
                    shape, nbytes = info
                    add_dense_ti(gguf_name, shape, nbytes)
                    seen_tis.add(gguf_name)
            for hf_suffix, gguf_tmpl in nextn_map.items():
                gguf_name = gguf_tmpl.format(i=i)
                if gguf_name in seen_tis:
                    continue
                hf_key = MTP_NEXTN_HF.get(hf_suffix, f"{prefix}.{hf_suffix}")
                info = self._read_bf16_info(hf_key)
                if info:
                    shape, nbytes = info
                    add_dense_ti(gguf_name, shape, nbytes)
                    seen_tis.add(gguf_name)

            if i < self.n_layer:
                if cold_tier != "q1_only":
                    q4_up = (self.hidden * self.n_ff * 2 * emitted_exp // 32) * 18
                    add_ti(
                        f"blk.{i}.ffn_gate_up_exps.weight",
                        [self.hidden, self.n_ff * 2, emitted_exp],
                        GGML_TYPE_Q4_0,
                        q4_up,
                    )
                    q4_down = (self.n_ff * self.hidden * emitted_exp // 32) * 18
                    add_ti(
                        f"blk.{i}.ffn_down_exps.weight", [self.n_ff, self.hidden, emitted_exp], GGML_TYPE_Q4_0, q4_down
                    )
                if cold_tier in ("q1_0", "q1_only"):
                    # A compact primary model must use llama.cpp's canonical
                    # routed-expert tensor names.  The ``_lp`` names are only
                    # meaningful for a supplementary experts-only sidecar;
                    # llama.cpp does not discover them when loading a normal
                    # model and otherwise reports the canonical tensors as
                    # missing.
                    compact_suffix = "_lp" if self.experts_only or cold_tier == "q1_0" else ""
                    lp_type = (
                        GGML_TYPE_Q2_K
                        if self.q2_edge_layers and (i < self.q2_edge_layers or i >= self.n_layer - self.q2_edge_layers)
                        else self.compact_expert_ggml_type
                    )
                    lp_block = ggml_type_block_size(lp_type)
                    lp_size = ggml_type_size(lp_type)
                    lp_up = (self.hidden * self.n_ff * 2 * emitted_exp // lp_block) * lp_size
                    add_ti(
                        f"blk.{i}.ffn_gate_up_exps{compact_suffix}.weight",
                        [self.hidden, self.n_ff * 2, emitted_exp],
                        lp_type,
                        lp_up,
                    )
                    lp_down = (self.n_ff * self.hidden * emitted_exp // lp_block) * lp_size
                    add_ti(
                        f"blk.{i}.ffn_down_exps{compact_suffix}.weight",
                        [self.n_ff, self.hidden, emitted_exp],
                        lp_type,
                        lp_down,
                    )
            else:
                n_f16_up = self.hidden * self.n_ff * 2 * emitted_exp * 2
                add_ti(
                    f"blk.{i}.ffn_gate_up_exps.weight",
                    [self.hidden, self.n_ff * 2, emitted_exp],
                    GGML_TYPE_F16,
                    n_f16_up,
                )
                n_f16_down = self.n_ff * self.hidden * emitted_exp * 2
                add_ti(
                    f"blk.{i}.ffn_down_exps.weight", [self.n_ff, self.hidden, emitted_exp], GGML_TYPE_F16, n_f16_down
                )

        if self.mtp_layers > 0:
            kv("qwen35moe.nextn_predict_layers", 4, self.mtp_layers)

        self._kvs = list(kvs.items())
        self._tensor_infos = tensor_infos

    def _write_header(self, f):
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(self._tensor_infos)))
        f.write(struct.pack("<Q", len(self._kvs)))

    def _write_kv(self, f):
        for key, (vt, val) in self._kvs:
            write_gguf_kv(f, key, vt, val)

    def _write_ti(self, f):
        ti_start = f.tell()
        layout, total_data_bytes = tensor_data_layout(self._tensor_infos)
        # Compute total TI section size
        ti_size = sum(8 + len(n.encode("utf-8")) + 4 + nd * 8 + 4 + 8 for n, nd, _, _, _ in layout)
        data_start = round_up(ti_start + ti_size, 32)
        self._data_start = data_start
        self._data_end = data_start + total_data_bytes
        self._tensor_offsets = {name: data_off for name, _nd, _dims, _dtype, data_off in layout}
        for name, n_dims, dims, dtype, data_off in layout:
            write_tensor_info(f, name, n_dims, dims, dtype, data_off)
        return data_start

    def _pad_to_tensor(self, f, name):
        """Pad the payload stream to the catalog offset for ``name``."""
        target = self._data_start + self._tensor_offsets[name]
        current = f.tell()
        if current > target:
            raise ValueError(f"tensor payload for {name} started {current - target} bytes past its catalog offset")
        if current < target:
            f.write(b"\x00" * (target - current))

    def _write_data_nvfp4(self, f, prefix, i):
        t0 = time.time()
        print(f"  Converting layer {i} experts ({self.n_exp} experts)...")

        # Write gate/up expert blocks immediately. The previous implementation
        # retained every converted expert for a layer in Python lists, causing
        # multi-gigabyte peak memory usage, and serialized every dequantize/
        # quantize on one core. NumPy ufuncs release the GIL, so a bounded
        # thread pool overlaps the dequantize/quantize cost across cores while
        # ``executor.map`` preserves the exact output byte order.
        def quantize_gate_up(e):
            ep = f"{prefix}.mlp.experts.{e}"
            chunks = []
            for projection in ("gate_proj", "up_proj"):
                tensors = self._read_nvfp4(f"{ep}.{projection}")
                if tensors is None:
                    raise KeyError(f"Missing NVFP4 tensor: {ep}.{projection}")
                packed, scales, s2 = tensors
                # The dequant buffer is private to this worker; quantize in
                # place so Q4_0 does not re-copy the full f32 matrix.
                chunks.append(_quantize_q4_0_buffer(dequantize_nvfp4(packed, scales, s2)))
                del packed, scales
            return b"".join(chunks)

        def quantize_down(e):
            ep = f"{prefix}.mlp.experts.{e}"
            tensors = self._read_nvfp4(f"{ep}.down_proj")
            if tensors is None:
                raise KeyError(f"Missing NVFP4 tensor: {ep}.down_proj")
            packed, scales, s2 = tensors
            data = _quantize_q4_0_buffer(dequantize_nvfp4(packed, scales, s2))
            del packed, scales
            return data

        with ThreadPoolExecutor(max_workers=self.conversion_workers) as executor:
            gate_start = time.time()
            self._write_windowed_map(f, executor, quantize_gate_up)
            gate_elapsed = time.time() - gate_start
            print(f"    gate/up {self.n_exp}/{self.n_exp} ({gate_elapsed:.1f}s)")
            self._pad_to_tensor(f, f"blk.{i}.ffn_down_exps.weight")
            down_started = time.time()
            self._write_windowed_map(f, executor, quantize_down)
            down_elapsed = time.time() - down_started
            print(f"    down    {self.n_exp}/{self.n_exp} ({down_elapsed:.1f}s); layer total {time.time() - t0:.1f}s")

    def _write_windowed_map(self, f, executor, fn):
        """Apply ``fn`` to every emitted expert, writing results in order.

        Submitting the whole layer's experts to ``executor.map`` at once can
        buffer a layer's worth of quantized payloads in memory; a bounded
        window keeps peak memory proportional to the worker count while
        ``executor.map`` still yields results in input order.
        """
        indices = list(self._emit_experts())
        window = max(1, self.conversion_workers)
        for start in range(0, len(indices), window):
            stop = min(start + window, len(indices))
            for payload in executor.map(fn, indices[start:stop]):
                f.write(payload)

    def _write_data_dual_experts(self, f, prefix, i):
        """Write one layer's experts as Q4_0 hot + Q1_0 cold twins.

        The file's tensor-info order for a layer is:
          ffn_gate_up_exps.weight  (Q4_0)  ffn_down_exps.weight (Q4_0)
          ffn_gate_up_exps_lp.weight (Q1_0) ffn_down_exps_lp.weight (Q1_0)
        Both Q4_0 tensors are streamed to disk as experts finish; only the two
        Q1_0 cold twins are buffered in memory (~335 MiB/layer) and flushed
        after the Q4_0 halves, matching the tensor-info layout.
        """
        print(f"  Converting layer {i} experts (Q4_0 hot + Q1_0 cold, {self.n_exp} experts)...")

        # Two passes over the experts. The file's tensor-info order is
        # gate_up Q4, down Q4, gate_up_lp Q1, down_lp Q1, but each pass below
        # produces one Q4 tensor in the right position so only its Q1 twin is
        # buffered. Previously all four tensors were computed in a single loop
        # and three were retained (~785 MiB/layer); streaming the Q4 tensors
        # directly keeps peak cold-tier memory near the Q1 twins (~335 MiB).
        def quantize_twin(e, projections):
            ep = f"{prefix}.mlp.experts.{e}"
            q4_chunks = []
            q1_chunks = []
            for projection in projections:
                tensors = self._read_nvfp4(f"{ep}.{projection}")
                if tensors is None:
                    raise KeyError(f"Missing NVFP4 tensor: {ep}.{projection}")
                packed, scales, s2 = tensors
                f32 = dequantize_nvfp4(packed, scales, s2)
                # Both twins mutate their input in place, so give Q4_0 one copy
                # and let Q1_0 consume the original buffer. Using quantize_q4_0/
                # quantize_q1_0 here would copy the full f32 matrix twice per
                # projection (~12 MiB each); this keeps it to a single copy.
                q4_chunks.append(_quantize_q4_0_buffer(f32.copy()))
                q1_chunks.append(_quantize_q1_0_buffer(f32))
                del packed, scales, f32
            return b"".join(q4_chunks), b"".join(q1_chunks)

        # Parallelize the dequantize/quantize work (NumPy releases the GIL)
        # while streaming each Q4 half to disk in order; only the Q1 cold twins
        # stay buffered, keeping peak memory near the Q1-twin size.
        with ThreadPoolExecutor(max_workers=self.conversion_workers) as executor:

            def write_twin_pass(fn, projections):
                q1_chunks = []
                indices = list(self._emit_experts())
                window = max(1, self.conversion_workers)
                for start in range(0, len(indices), window):
                    stop = min(start + window, len(indices))
                    for q4b, q1b in executor.map(
                        lambda e, p=projections: fn(e, p),
                        indices[start:stop],
                    ):
                        f.write(q4b)
                        q1_chunks.append(q1b)
                return q1_chunks

            q1_up_chunks = write_twin_pass(quantize_twin, ("gate_proj", "up_proj"))
            print(f"    gate/up {self.n_exp}/{self.n_exp} done", flush=True)
            self._pad_to_tensor(f, f"blk.{i}.ffn_down_exps.weight")
            q1_down_chunks = write_twin_pass(quantize_twin, ("down_proj",))
            print(f"    down  {self.n_exp}/{self.n_exp} done", flush=True)
        self._pad_to_tensor(f, f"blk.{i}.ffn_gate_up_exps_lp.weight")
        f.write(b"".join(q1_up_chunks))
        del q1_up_chunks
        self._pad_to_tensor(f, f"blk.{i}.ffn_down_exps_lp.weight")
        f.write(b"".join(q1_down_chunks))
        del q1_down_chunks

    def _write_data_compact_experts(self, f, prefix, i, tensor_type):
        """Write one layer's routed experts in the requested compact format."""
        t0 = time.time()
        quantizers = {
            GGML_TYPE_Q1_0: ("Q1_0", _quantize_q1_0_buffer),
            GGML_TYPE_Q2_K: ("Q2_K", quantize_q2_k),
            GGML_TYPE_Q3_K: ("Q3_K", quantize_q3_k),
            GGML_TYPE_IQ1_S: ("IQ1_S", quantize_iq1_s),
        }
        label, quantizer = quantizers[tensor_type]
        print(f"  Converting layer {i + 1}/{self.n_layer} experts ({label} only, {self.n_exp} experts)...")

        def quantize_gate_up(e):
            ep = f"{prefix}.mlp.experts.{e}"
            chunks = []
            for projection in ("gate_proj", "up_proj"):
                tensors = self._read_nvfp4(f"{ep}.{projection}")
                if tensors is None:
                    raise KeyError(f"Missing NVFP4 tensor: {ep}.{projection}")
                packed, scales, s2 = tensors
                f32 = dequantize_nvfp4(packed, scales, s2)
                importance = None
                if tensor_type == GGML_TYPE_IQ1_S:
                    if self.imatrix is None:
                        raise ValueError(f"{label} expert conversion requires an importance matrix")
                    importance = self.imatrix.expert(f"blk.{i}.ffn_gate_up_exps.weight", e, f32.shape[1])
                elif tensor_type in (GGML_TYPE_Q2_K, GGML_TYPE_Q3_K) and self.imatrix is not None:
                    importance = self.imatrix.expert(f"blk.{i}.ffn_gate_up_exps.weight", e, f32.shape[1])
                chunks.append(quantizer(f32, importance) if importance is not None else quantizer(f32))
                del packed, scales, f32
            return b"".join(chunks)

        def quantize_down(e):
            ep = f"{prefix}.mlp.experts.{e}"
            tensors = self._read_nvfp4(f"{ep}.down_proj")
            if tensors is None:
                raise KeyError(f"Missing NVFP4 tensor: {ep}.down_proj")
            packed, scales, s2 = tensors
            f32 = dequantize_nvfp4(packed, scales, s2)
            importance = None
            if tensor_type == GGML_TYPE_IQ1_S:
                if self.imatrix is None:
                    raise ValueError(f"{label} expert conversion requires an importance matrix")
                importance = self.imatrix.expert(f"blk.{i}.ffn_down_exps.weight", e, f32.shape[1])
            elif tensor_type in (GGML_TYPE_Q2_K, GGML_TYPE_Q3_K) and self.imatrix is not None:
                importance = self.imatrix.expert(f"blk.{i}.ffn_down_exps.weight", e, f32.shape[1])
            return quantizer(f32, importance) if importance is not None else quantizer(f32)

        # Bound in-flight expert work to the worker pool size. Submitting the
        # whole layer's experts to executor.map at once can buffer a layer's
        # worth of quantized payloads in memory; windowed submission keeps
        # peak usage proportional to the worker count while preserving order.
        with ThreadPoolExecutor(max_workers=self.conversion_workers) as executor:
            compact_suffix = "_lp" if self.experts_only else ""
            self._write_windowed_map(f, executor, quantize_gate_up)
            gate_elapsed = time.time() - t0
            print(f"    gate/up {self.n_exp}/{self.n_exp} ({gate_elapsed:.1f}s)")
            self._pad_to_tensor(f, f"blk.{i}.ffn_down_exps{compact_suffix}.weight")
            down_started = time.time()
            self._write_windowed_map(f, executor, quantize_down)
            down_elapsed = time.time() - down_started
            print(f"    down    {self.n_exp}/{self.n_exp} ({down_elapsed:.1f}s); layer total {time.time() - t0:.1f}s")

    def _write_data_q1_experts_from_q4(self, f, reader, i):
        """Derive compact experts directly from an existing Q4 GGUF mmap."""

        # Build the tensor-name index once per sidecar and reuse it across
        # layers instead of rescanning the entire tensor list for every layer.
        cache_key = id(reader)
        cache = getattr(self, "_q4_experts_index", None)
        if cache is None or cache[0] != cache_key:
            cache = (cache_key, {tensor.name: tensor for tensor in reader.tensors})
            self._q4_experts_index = cache
        tensors = cache[1]
        names = (
            f"blk.{i}.ffn_gate_up_exps.weight",
            f"blk.{i}.ffn_down_exps.weight",
        )
        print(f"  Compacting layer {i} Q4_0 experts to Q1_0 ({self.n_exp} experts)...")

        def quantize_from_q4(e):
            # The Q4 decode buffer is private to this thread; consume it in
            # place rather than copying a fresh f32 matrix for the Q1 pass.
            return _quantize_q1_0_buffer(dequantize_q4_0(data[e]))

        for name in names:
            tensor = tensors.get(name)
            if tensor is None:
                raise KeyError(f"Q4 sidecar is missing tensor: {name}")
            if int(tensor.tensor_type) != GGML_TYPE_Q4_0:
                raise ValueError(f"Q4 sidecar tensor {name} has type {tensor.tensor_type}, expected Q4_0")
            data = np.asarray(tensor.data)
            if data.shape[0] != self.n_exp:
                raise ValueError(f"Q4 sidecar tensor {name} has {data.shape[0]} experts, expected {self.n_exp}")
            # The mmap'd sidecar rows are read-only and shared across worker
            # threads; only the decoded/quantized payloads are per-thread.
            self._pad_to_tensor(f, name.replace(".weight", "_lp.weight"))
            with ThreadPoolExecutor(max_workers=self.conversion_workers) as executor:
                gate_start = time.time()
                self._write_windowed_map(f, executor, quantize_from_q4)
                elapsed = time.time() - gate_start
            print(f"    {name}: {self.n_exp}/{self.n_exp} experts ({elapsed:.1f}s)")

    def _write_q4_dense(self, f, name, nbytes, row_chunk=1024):
        hf_key = self._bf16_hf_key(name)
        if hf_key is None:
            raise KeyError(f"No Hugging Face mapping exists for GGUF tensor {name}")

        # Embedding and LM-head matrices are several GiB in F32. Quantizing
        # them as one NumPy array creates multiple equally large temporaries
        # and can exceed system RAM. Safetensors slices keep peak conversion
        # memory proportional to row_chunk instead.
        if name in ("token_embd.weight", "output.weight"):
            source = self._read_torch_slice(hf_key)
            if source is None:
                raise KeyError(f"Missing source tensor {hf_key} for GGUF tensor {name}")
            shape = source.get_shape() if hasattr(source, "get_shape") else source.shape
            written = 0
            for start in range(0, shape[0], row_chunk):
                rows = source[start : min(start + row_chunk, shape[0])]
                data = quantize_q4_0(rows.to(torch.float32).numpy())
                f.write(data)
                written += len(data)
                del rows, data
            if written != nbytes:
                raise ValueError(f"Tensor {name} wrote {written} bytes; expected {nbytes}")
            return

        t = self._read_torch(hf_key)
        if t is None:
            raise KeyError(f"Missing source tensor {hf_key} for GGUF tensor {name}")
        t = self._transform_source_tensor(t, hf_key)
        t = t.contiguous()
        data = quantize_q4_0(t.to(torch.float32).numpy())
        if len(data) != nbytes:
            raise ValueError(f"Tensor {name} wrote {len(data)} bytes; expected {nbytes}")
        f.write(data)

    def _bf16_hf_key(self, gguf_name):
        if gguf_name == "token_embd.weight":
            return "model.language_model.embed_tokens.weight"
        if gguf_name == "output_norm.weight":
            return "model.language_model.norm.weight"
        if gguf_name == "output.weight":
            return "lm_head.weight"
        if "blk." in gguf_name:
            i = int(gguf_name.split(".")[1])
            if i >= self.n_layer:
                mtp_idx = i - self.n_layer
                for hf_suffix, gguf_tmpl in MTP_MAP.items():
                    if gguf_tmpl.format(i=i) == gguf_name:
                        return f"mtp.layers.{mtp_idx}.{hf_suffix}"
                for hf_suffix, gguf_tmpl in MTP_NEXTN_MAP.items():
                    if gguf_tmpl.format(i=i) == gguf_name:
                        return MTP_NEXTN_HF.get(hf_suffix, f"mtp.layers.{mtp_idx}.{hf_suffix}")
                return None
            lt = self.layer_types[i]
            kmap = LAYER_MAP_FULL if lt == "full_attention" else LAYER_MAP_LINEAR
            prefix = f"model.language_model.layers.{i}"
            for hf_suffix, gguf_tmpl in kmap.items():
                if gguf_tmpl.format(i=i) == gguf_name:
                    return f"{prefix}.{hf_suffix}"
            return None
        return None

    def _load_bf16_tensor_bytes(self, name, dtype, nbytes):
        hf_key = self._bf16_hf_key(name)
        if hf_key is None:
            raise KeyError(f"No Hugging Face mapping exists for GGUF tensor {name}")
        t = self._read_torch(hf_key)
        if t is None:
            raise KeyError(f"Missing source tensor {hf_key} for GGUF tensor {name}")
        t = self._transform_source_tensor(t, hf_key)
        # GGUF dimensions are stored in ne[] order while payloads remain
        # source row-major. Reversing metadata dimensions and transposing the
        # bytes as well corrupts every matrix.
        t = t.contiguous()
        if dtype == GGML_TYPE_F32:
            return t.to(torch.float32).numpy().tobytes()
        elif dtype == GGML_TYPE_BF16:
            # NumPy has no portable bfloat16 dtype; preserve the raw 16-bit
            # representation expected by GGML_TYPE_BF16.
            return t.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()
        else:
            return t.to(torch.float16).numpy().tobytes()

    def convert(self, output_path: str):
        import gc

        print(f"Converting {self.model_dir} \u2192 {output_path}")
        self._init_gguf(None)
        if self._kept is not None:
            # The catalog must match the pruned count on every expert tensor
            # and on the router; fail loudly here instead of emitting a
            # silently misaligned GGUF.
            for name, _nd, dims, _dtype, _nbytes in self._tensor_infos:
                if ".ffn_gate_up_exps" in name or ".ffn_down_exps" in name:
                    if len(dims) < 3 or dims[2] != self._emitted_exp:
                        raise ValueError(
                            f"pruned conversion produced expert tensor {name} "
                            f"with expert dim {dims[2] if len(dims) >= 3 else '?'}, "
                            f"expected {self._emitted_exp}"
                        )
                if name.endswith("ffn_gate_inp.weight"):
                    if len(dims) < 2 or dims[1] != self._emitted_exp:
                        raise ValueError(
                            f"pruned conversion produced router tensor {name} "
                            f"with expert dim {dims[1] if len(dims) >= 2 else '?'}, "
                            f"expected {self._emitted_exp}"
                        )
            print(
                f"  EXPERIMENTAL expert pruning: emitting {self._emitted_exp}/"
                f"{self.n_exp} experts per layer (quality may drop)"
            )
        if self.experts_only:
            self._tensor_infos = [info for info in self._tensor_infos if info[0].endswith("_exps_lp.weight")]
            sidecar_keys = {
                "general.architecture",
                "qwen35moe.block_count",
                "qwen35moe.embedding_length",
                "qwen35moe.expert_feed_forward_length",
                "qwen35moe.expert_count",
                "qwen35moe.expert_used_count",
            }
            self._kvs = [item for item in self._kvs if item[0] in sidecar_keys]
            expected = self.n_layer * 2
            if len(self._tensor_infos) != expected:
                raise ValueError(
                    f"experts-only conversion expected {expected} compact tensors, found {len(self._tensor_infos)}"
                )
        output = os.path.abspath(output_path)
        output_dir = os.path.dirname(output)
        os.makedirs(output_dir, exist_ok=True)
        estimated_bytes = sum(item[4] for item in self._tensor_infos) + 16 * 1024**2
        free_bytes = shutil.disk_usage(output_dir).free
        if free_bytes < estimated_bytes:
            raise OSError(
                f"Insufficient disk space for conversion: need approximately "
                f"{estimated_bytes / 1024**3:.1f} GiB, have "
                f"{free_bytes / 1024**3:.1f} GiB"
            )
        q4_reader = None
        if self.q4_sidecar_source:
            import gguf

            sidecar = os.path.abspath(os.path.expanduser(self.q4_sidecar_source))
            if not os.path.isfile(sidecar):
                raise FileNotFoundError(f"Q4 sidecar source does not exist: {sidecar}")
            try:
                q4_reader = gguf.GGUFReader(sidecar)
            except Exception as exc:
                raise RuntimeError(
                    "Cannot parse the Q4 sidecar with the installed gguf reader. "
                    "Kestrel writes expert tensors in llama.cpp ne-order; the "
                    f"reader reversed them and failed: {exc}"
                ) from exc

        partial_fd, partial = tempfile.mkstemp(
            dir=output_dir,
            prefix=f".{os.path.basename(output)}.",
            suffix=".partial",
        )
        try:
            with os.fdopen(partial_fd, "wb") as f:
                self._write_header(f)
                self._write_kv(f)
                data_start = self._write_ti(f)

                pad = data_start - f.tell()
                if pad > 0:
                    f.write(b"\x00" * pad)

                expert_done = set()
                f16_exp_done = set()
                for name, _n_dims, _dims, dtype, nbytes in self._tensor_infos:
                    is_exps = ("ffn_gate_up_exps" in name or "ffn_down_exps" in name) and "_lp" not in name
                    if is_exps and dtype == GGML_TYPE_Q4_0:
                        i = int(name.split(".")[1])
                        if i not in expert_done:
                            self._pad_to_tensor(f, name)
                            prefix = f"model.language_model.layers.{i}"
                            if self.cold_tier == "q1_0":
                                self._write_data_dual_experts(f, prefix, i)
                            else:
                                self._write_data_nvfp4(f, prefix, i)
                            expert_done.add(i)
                    elif dtype in (GGML_TYPE_Q1_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_IQ1_S) and (
                        "_exps_lp" in name
                        or (self.cold_tier == "q1_only" and ("ffn_gate_up_exps" in name or "ffn_down_exps" in name))
                    ):
                        if self.cold_tier == "q1_only":
                            i = int(name.split(".")[1])
                            if i not in expert_done:
                                self._pad_to_tensor(f, name)
                                if q4_reader is not None:
                                    self._write_data_q1_experts_from_q4(f, q4_reader, i)
                                else:
                                    prefix = f"model.language_model.layers.{i}"
                                    self._write_data_compact_experts(f, prefix, i, dtype)
                                expert_done.add(i)
                        # Dual-tier twins are emitted by the Q4-triggered pass.
                        continue
                    elif dtype == GGML_TYPE_F16 and "ffn_gate_up_exps" in name:
                        if name not in f16_exp_done:
                            i = int(name.split(".")[1])
                            mtp_idx = i - self.n_layer
                            prefix = f"mtp.layers.{mtp_idx}"
                            self._pad_to_tensor(f, name)
                            self._write_data_f16_experts(f, prefix, i)
                            f16_exp_done.add(name)
                    elif dtype == GGML_TYPE_Q4_0:
                        self._pad_to_tensor(f, name)
                        self._write_q4_dense(f, name, nbytes)
                        if nbytes >= 256 * 1024**2:
                            gc.collect()
                    elif dtype in (GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32):
                        if "ffn_gate_up_exps" in name or "ffn_down_exps" in name:
                            continue
                        self._pad_to_tensor(f, name)
                        data = self._load_bf16_tensor_bytes(name, dtype, nbytes)
                        if len(data) != nbytes:
                            raise ValueError(f"Tensor {name} wrote {len(data)} bytes; expected {nbytes}")
                        f.write(data)
                        del data
                        # Freed arrays return to the allocator by refcount; only
                        # the large weights justify a cycle-collector sweep.
                        if nbytes >= 256 * 1024**2:
                            gc.collect()
                expected_end = getattr(self, "_data_end", f.tell())
                if f.tell() != expected_end:
                    raise ValueError(
                        f"GGUF payload ended at byte {f.tell()}, expected {expected_end} from the tensor catalog"
                    )
                f.flush()
                os.fsync(f.fileno())
            os.replace(partial, output)
            util.sync_directory(output_dir)
        except BaseException:
            if os.path.exists(partial):
                os.unlink(partial)
            raise

        print(f"Done! Wrote {output} ({os.path.getsize(output)} bytes)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert Qwen3.5-122B-A10B NVFP4 to GGUF")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/.local/share/kestrel/models/qwen3.5-122b-a10b-nvfp4.gguf"),
    )
    parser.add_argument("--test-single-layer", action="store_true")
    parser.add_argument(
        "--test-layers",
        type=int,
        help=(
            "emit only the first N trunk layers for smoke testing; Qwen3.5 "
            "hybrid models require at least 4 layers to include one complete "
            "recurrent/full-attention cycle"
        ),
    )
    parser.add_argument(
        "--include-mtp",
        action="store_true",
        help="include the optional speculative MTP draft block",
    )
    parser.add_argument(
        "--dense-q4",
        action="store_true",
        help="quantize dense matrices (attention/FFN/embedding) to Q4_0 instead of BF16",
    )
    parser.add_argument(
        "--cold-tier",
        choices=["off", "q1_0", "q1_only"],
        default="off",
        help="emit Q1_0 expert twins alongside Q4_0, or Q1_0 experts only for compact CUDA experiments",
    )
    parser.add_argument(
        "--q4-sidecar-source",
        help="derive q1_only experts directly from an existing canonical Q4 GGUF",
    )
    parser.add_argument(
        "--experts-only",
        action="store_true",
        help="emit only compact routed experts for use as a cold sidecar",
    )
    parser.add_argument(
        "--q2-edge-layers",
        type=int,
        default=0,
        metavar="N",
        help="use Q2_K experts for the first and last N layers of a direct q1_only conversion",
    )
    parser.add_argument(
        "--all-q2",
        action="store_true",
        help="use Q2_K for every routed expert in a direct compact conversion",
    )
    parser.add_argument(
        "--compact-expert-type",
        choices=["q1_0", "q2_k", "q3_k", "iq1_s"],
        default="q1_0",
        help="expert format used by a direct q1_only compact-primary conversion; q2_k is the all-Q2 mode",
    )
    parser.add_argument(
        "--imatrix",
        help="llama-imatrix GGUF used to calibrate IQ1_S, Q2_K, or Q3_K experts",
    )
    parser.add_argument(
        "--conversion-workers",
        type=int,
        help="parallel expert conversion workers (default: up to 4)",
    )
    args = parser.parse_args()

    conv = NVFP4Converter(
        args.model_dir,
        include_mtp=args.include_mtp,
        dense_q4=args.dense_q4,
        cold_tier=args.cold_tier,
        q4_sidecar_source=args.q4_sidecar_source,
        experts_only=args.experts_only,
        q2_edge_layers=args.q2_edge_layers,
        compact_expert_type=args.compact_expert_type,
        all_q2=args.all_q2,
        imatrix_path=args.imatrix,
        conversion_workers=args.conversion_workers,
    )
    if args.test_single_layer and args.test_layers is not None:
        parser.error("--test-single-layer and --test-layers are mutually exclusive")
    if args.test_single_layer:
        # A one-layer Qwen3.5 hybrid model has no full-attention layer. The
        # hybrid KV cache then creates an input tensor without a backing
        # buffer and inference aborts before reaching the experts. Preserve
        # the old flag as a smoke-test alias, but emit one complete cycle.
        conv.n_layer = 4
        conv.mtp_layers = 0
    elif args.test_layers is not None:
        if args.test_layers < 4 or args.test_layers > conv.n_layer:
            parser.error(f"--test-layers must be between 4 and {conv.n_layer}")
        if args.q2_edge_layers * 2 > args.test_layers:
            parser.error("--q2-edge-layers cannot cover more than half of --test-layers")
        conv.n_layer = args.test_layers
        conv.mtp_layers = 0
    print(f"Converting {args.model_dir} → {args.output}")
    conv.convert(args.output)


def find_convert_script(llama_cpp_dir: str | None = None) -> str | None:
    """Locate llama.cpp's ``convert_hf_to_gguf.py``.

    Searches the selected llama.cpp directory and, when it is a ``<checkout>/build``
    path, its family root too, plus the known local checkouts.
    """
    candidates: list[str] = []
    if llama_cpp_dir:
        candidates.append(llama_cpp_dir)
    candidates += [
        os.path.expanduser("~/llama.cpp"),
        os.path.expanduser("~/llama.cpp-moe-cache"),
    ]
    resolved: list[str] = []
    for directory in candidates:
        resolved.append(directory)
        if directory.rstrip(os.sep).endswith(os.sep + "build"):
            resolved.append(os.path.dirname(directory.rstrip(os.sep)))
    for directory in resolved:
        script = os.path.join(directory, "convert_hf_to_gguf.py")
        if os.path.isfile(script):
            return script
    return None


def generic_convert_hf_to_gguf(
    src_path: str,
    output_path: str,
    *,
    outtype: str = "bf16",
    llama_cpp_dir: str | None = None,
) -> tuple[str, int]:
    """Convert any Hugging Face safetensors dir to GGUF via llama.cpp's script.

    Broadens supported inputs beyond the NVFP4 Qwen3.5 layout: any model with a
    ``config.json`` upstream can be emitted as GGUF without reimplementing the
    per-architecture conversion. Returns ``(command_line, returncode)`` and
    raises ``FileNotFoundError`` if ``convert_hf_to_gguf.py`` is absent.
    """
    script = find_convert_script(llama_cpp_dir)
    if not script:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found; clone llama.cpp and (optional) set KESTREL_LLAMA_CPP_DIR to its checkout"
        )
    output = Path(output_path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the staging name GGUF-shaped. Some third-party converter versions
    # infer or replace an output suffix; an explicit final `.gguf` suffix and
    # no `{ftype}` placeholder makes the requested path unambiguous.
    partial_fd, partial_name = tempfile.mkstemp(
        dir=output.parent,
        # Do not include the caller's filename: upstream treats placeholders
        # such as ``{ftype}`` specially and could derive an unowned sibling.
        prefix=".kestrel-convert-",
        suffix=".partial.gguf",
    )
    os.close(partial_fd)
    partial = Path(partial_name)
    command = [
        sys.executable,
        script,
        src_path,
        "--outfile",
        str(partial),
        "--outtype",
        outtype,
    ]
    display_command = [*command]
    display_command[display_command.index("--outfile") + 1] = str(output)
    try:
        # A converter can run for hours and emit unbounded output. Do not let
        # its logs consume coordinator RAM or disk; the exit status remains a
        # stable diagnostic and users can run the returned command directly
        # when they need upstream's verbose output.
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=24 * 60 * 60,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("convert_hf_to_gguf.py exceeded its 24-hour time budget") from exc
        except OSError as exc:
            raise RuntimeError(f"could not start convert_hf_to_gguf.py: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(f"convert_hf_to_gguf.py failed (exit {result.returncode})")

        # Success means the exact path we supplied was populated. Do not guess
        # at an alternate derived filename: adopting a nearby file could move
        # unrelated data over the requested model.
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(partial, flags)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
                    raise RuntimeError("convert_hf_to_gguf.py did not produce a non-empty regular GGUF staging file")
                if os.read(fd, 4) != b"GGUF":
                    raise RuntimeError("convert_hf_to_gguf.py produced a file without GGUF magic")
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise RuntimeError("convert_hf_to_gguf.py did not produce its exact requested output path") from exc
        os.replace(partial, output)
        util.sync_directory(output.parent)
        return " ".join(display_command), result.returncode
    finally:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
