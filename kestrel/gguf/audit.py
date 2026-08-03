from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def audit_snapshot(
    fields: dict[str, Any],
    tensors: list[tuple[str, int]],
    *,
    source_config: dict[str, Any] | None = None,
    source_tokenizer_config: dict[str, Any] | None = None,
) -> list[AuditFinding]:
    """Audit already-decoded GGUF metadata and tensor records.

    This pure-data layer is intentionally independent of the GGUF reader so the
    structural failure rules can be tested with small fixtures.
    """

    findings: list[AuditFinding] = []

    def error(code: str, message: str):
        findings.append(AuditFinding("error", code, message))

    def warning(code: str, message: str):
        findings.append(AuditFinding("warning", code, message))

    arch = str(fields.get("general.architecture", "unknown"))
    block_count = _integer(fields.get(f"{arch}.block_count"))
    mtp_layers = _integer(fields.get(f"{arch}.nextn_predict_layers"))
    names = [name for name, _ in tensors]
    block_ids = sorted(
        {
            int(match.group(1))
            for name in names
            if (match := re.match(r"blk\.(\d+)\.", name))
        }
    )
    nextn_blocks = sorted(
        {
            int(match.group(1))
            for name in names
            if ".nextn." in name
            and (match := re.match(r"blk\.(\d+)\.", name))
        }
    )

    source_text = None
    if source_config:
        source_text = source_config.get("text_config", source_config)
        target_layers = _integer(source_text.get("num_hidden_layers"))
        source_mtp = _integer(source_text.get("mtp_num_hidden_layers"))
        if target_layers:
            missing = [index for index in range(target_layers) if index not in block_ids]
            if missing:
                error(
                    "TARGET_BLOCKS_MISSING",
                    f"Missing target block IDs: {missing[:8]}"
                    + ("..." if len(missing) > 8 else ""),
                )
            replaced = [index for index in nextn_blocks if index < target_layers]
            if replaced:
                error(
                    "MTP_REPLACES_TARGET",
                    "MTP next-token tensors occur inside target block range at "
                    f"{replaced}; MTP must be appended after all {target_layers} target blocks.",
                )
            if mtp_layers:
                expected = target_layers + mtp_layers
                if block_count != expected:
                    error(
                        "BLOCK_COUNT_MISMATCH",
                        f"GGUF declares {block_count} blocks and {mtp_layers} MTP layer(s); "
                        f"source requires {target_layers} target + {mtp_layers} MTP = {expected}.",
                    )
            elif source_mtp and nextn_blocks:
                error(
                    "MTP_METADATA_MISSING",
                    "MTP tensors exist but nextn_predict_layers is missing or zero.",
                )

    token_types = fields.get("tokenizer.ggml.token_type")
    tokens = fields.get("tokenizer.ggml.tokens") or []
    if token_types is None:
        error(
            "TOKEN_TYPES_MISSING",
            "tokenizer.ggml.token_type is absent; special and reserved tokens cannot be classified safely.",
        )
    if "tokenizer.ggml.pre" not in fields:
        error(
            "PRETOKENIZER_MISSING",
            "tokenizer.ggml.pre is absent, so llama.cpp cannot select the source tokenizer behavior reliably.",
        )

    empty_ids = [
        index
        for index, token in enumerate(tokens)
        if token == "" or token == b""
    ]
    if empty_ids:
        unsafe = token_types is None or any(
            index >= len(token_types) or _integer(token_types[index], -1) != 5
            for index in empty_ids
        )
        if unsafe:
            error(
                "EMPTY_NORMAL_TOKENS",
                f"{len(empty_ids)} empty vocabulary entries are not explicitly UNUSED "
                f"(first IDs: {empty_ids[:5]}).",
            )

    if source_tokenizer_config:
        if source_tokenizer_config.get("chat_template") and "tokenizer.chat_template" not in fields:
            error(
                "CHAT_TEMPLATE_MISSING",
                "The source chat template exists but is absent from the GGUF.",
            )
        if source_tokenizer_config.get("bos_token") is None and "tokenizer.ggml.bos_token_id" in fields:
            error(
                "FABRICATED_BOS",
                f"GGUF declares BOS ID {fields['tokenizer.ggml.bos_token_id']} "
                "but the source tokenizer has no BOS token.",
            )
        if source_tokenizer_config.get("pad_token") and "tokenizer.ggml.padding_token_id" not in fields:
            error(
                "PAD_TOKEN_MISSING",
                "The source pad token exists but tokenizer.ggml.padding_token_id is absent.",
            )

    f32_count = sum(dtype == 0 for _, dtype in tensors)
    if f32_count > len(tensors) // 2:
        warning(
            "F32_EXPANSION",
            f"{f32_count} of {len(tensors)} tensors are F32. Elementwise vectors "
            "need F32 for llama.cpp CPU kernels, but a majority-F32 artifact likely "
            "expanded supported BF16 matrices and increases memory and I/O.",
        )

    return findings


def audit_cold_sidecar_snapshot(
    fields: dict[str, Any],
    tensors: list[tuple[str, int]],
    *,
    source_config: dict[str, Any] | None = None,
) -> list[AuditFinding]:
    """Validate the intentionally minimal experts-only cold-sidecar layout."""

    findings: list[AuditFinding] = []

    def error(code: str, message: str):
        findings.append(AuditFinding("error", code, message))

    arch = str(fields.get("general.architecture", "unknown"))
    if arch != "qwen35moe":
        error("SIDECAR_ARCH", f"Cold sidecar architecture is {arch}, expected qwen35moe.")

    declared_layers = _integer(fields.get(f"{arch}.block_count"))
    expected_layers = declared_layers
    if source_config:
        source_text = source_config.get("text_config", source_config)
        expected_layers = _integer(source_text.get("num_hidden_layers"), declared_layers)
        if declared_layers != expected_layers:
            error(
                "SIDECAR_BLOCK_COUNT",
                f"Cold sidecar declares {declared_layers} layers; source requires {expected_layers}.",
            )

    found: dict[int, set[str]] = {}
    for name, dtype in tensors:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate_up|down)_exps_lp\.weight", name)
        if not match:
            error("SIDECAR_EXTRA_TENSOR", f"Unexpected cold-sidecar tensor: {name}")
            continue
        layer = int(match.group(1))
        found.setdefault(layer, set()).add(match.group(2))
        if dtype != 41:
            error("SIDECAR_TYPE", f"{name} has GGML type {dtype}, expected Q1_0 (41).")

    for layer in range(expected_layers):
        missing = {"gate_up", "down"} - found.get(layer, set())
        if missing:
            error(
                "SIDECAR_EXPERTS_MISSING",
                f"Layer {layer} is missing compact tensor(s): {sorted(missing)}.",
            )
    extra_layers = sorted(layer for layer in found if layer >= expected_layers)
    if extra_layers:
        error("SIDECAR_EXTRA_LAYERS", f"Unexpected cold-sidecar layer IDs: {extra_layers}.")
    return findings


def _field_contents(reader, key: str):
    field = reader.fields.get(key)
    if field is None:
        return None
    try:
        value = field.contents()
    except (AttributeError, TypeError, ValueError):
        value = field.parts[-1]
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _compare_last_target_norm(
    reader,
    source_dir: Path,
    source_config: dict[str, Any],
) -> list[AuditFinding]:
    """Prove the known target/MTP substitution when optional deps are present."""

    try:
        import numpy as np
        import torch
        from safetensors import safe_open
    except ImportError:
        return []

    text_config = source_config.get("text_config", source_config)
    target_layers = _integer(text_config.get("num_hidden_layers"))
    if not target_layers:
        return []
    block = target_layers - 1
    gguf_name = f"blk.{block}.attn_norm.weight"
    tensor = next((item for item in reader.tensors if item.name == gguf_name), None)
    if tensor is None:
        return []
    tensor_type = _integer(tensor.tensor_type, -1)
    if tensor_type not in (0, 1, 30):
        return []

    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    weight_map = json.loads(index_path.read_text())["weight_map"]

    def source_tensor(key: str):
        shard = weight_map.get(key)
        if not shard or not (source_dir / shard).is_file():
            return None
        with safe_open(str(source_dir / shard), framework="pt") as handle:
            if key not in handle.keys():
                return None
            value = handle.get_tensor(key)
            if tensor_type == 30:
                # GGUFReader exposes BF16 payloads as raw bytes. Compare the
                # exact source BF16 representation rather than round-tripping
                # through float32.
                return (
                    value.to(torch.bfloat16)
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .reshape(-1)
                )
            dtype = torch.float16 if tensor_type == 1 else torch.float32
            return value.to(dtype).numpy().reshape(-1)

    target = source_tensor(
        f"model.language_model.layers.{block}.input_layernorm.weight"
    )
    mtp = source_tensor("mtp.layers.0.input_layernorm.weight")
    if target is None or mtp is None:
        return []
    actual = np.asarray(tensor.data).reshape(-1)
    if actual.shape == mtp.shape and np.array_equal(actual, mtp) and not np.array_equal(actual, target):
        return [
            AuditFinding(
                "error",
                "MTP_WEIGHTS_IN_TARGET_BLOCK",
                f"{gguf_name} is byte-for-byte equal to source MTP layer 0, "
                f"not source target layer {block}.",
            )
        ]
    return []


def audit_gguf(
    path: str,
    source_dir: str | None = None,
    *,
    cold_sidecar: bool = False,
) -> dict[str, Any]:
    import gguf

    model_path = Path(path).expanduser().resolve()
    reader = gguf.GGUFReader(str(model_path))
    arch = _field_contents(reader, "general.architecture") or "unknown"
    keys = {
        "general.architecture",
        f"{arch}.block_count",
        f"{arch}.nextn_predict_layers",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.pre",
        "tokenizer.chat_template",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.padding_token_id",
    }
    fields = {
        key: value
        for key in keys
        if (value := _field_contents(reader, key)) is not None
    }
    tensors = [(item.name, _integer(item.tensor_type, -1)) for item in reader.tensors]

    source_config = None
    source_tokenizer_config = None
    source_path = None
    if source_dir:
        source_path = Path(source_dir).expanduser().resolve()
        config_path = source_path / "config.json"
        tokenizer_path = source_path / "tokenizer_config.json"
        if config_path.is_file():
            source_config = json.loads(config_path.read_text())
        if tokenizer_path.is_file():
            source_tokenizer_config = json.loads(tokenizer_path.read_text())

    if cold_sidecar:
        findings = audit_cold_sidecar_snapshot(
            fields,
            tensors,
            source_config=source_config,
        )
    else:
        findings = audit_snapshot(
            fields,
            tensors,
            source_config=source_config,
            source_tokenizer_config=source_tokenizer_config,
        )
    if not cold_sidecar and source_path and source_config:
        findings.extend(_compare_last_target_norm(reader, source_path, source_config))

    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    return {
        "model": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "architecture": arch,
        "artifact_kind": "cold_sidecar" if cold_sidecar else "model",
        "tensor_count": len(tensors),
        "errors": errors,
        "warnings": warnings,
        "valid": errors == 0,
        "findings": [item.as_dict() for item in findings],
    }
