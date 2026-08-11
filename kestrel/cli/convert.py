"""Model conversion (``convert``) and GGUF validation (``audit``)."""

from __future__ import annotations

import json

from .. import ui
from ..errors import BackendError, ConversionError, ModelError
from . import model_source, state


def cmd_convert(args):
    model_info = model_source.detect_model(args.model)
    if not model_info or model_info["type"] != "safetensors" or not model_info["path"]:
        raise ModelError(
            "conversion input must be a downloaded safetensors model",
            hint="download the Hugging Face snapshot first or pass its local directory",
        )
    output = args.output or model_source._cached_gguf_path(model_info["path"])
    if args.generic:
        from ..gguf.converter import generic_convert_hf_to_gguf

        try:
            _command, _code = generic_convert_hf_to_gguf(
                model_info["path"],
                output,
                outtype=args.outtype,
                llama_cpp_dir=state.LLAMA_CPP_DIR,
            )
        except FileNotFoundError as exc:
            raise BackendError(str(exc)) from exc
        except RuntimeError as exc:
            raise ConversionError(str(exc)) from exc
        print(f"Converted {model_info['path']} → {output}")
        return
    from ..gguf.converter import NVFP4Converter

    NVFP4Converter(
        model_info["path"],
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
        experts_keep=args.experts_keep,
        expert_importance=args.expert_importance,
    ).convert(output)


def cmd_audit(args):
    from ..gguf.audit import audit_gguf

    report = audit_gguf(args.model, args.source, cold_sidecar=args.cold_sidecar)
    if args.json:
        print(json.dumps(report))
    else:
        verdict = "PASS" if report["valid"] else "FAIL"
        severity_color = {
            "error": ui.red,
            "warning": ui.yellow,
            "info": ui.cyan,
        }
        findings = []
        for item in report["findings"]:
            label = item["severity"].upper()
            colorize = severity_color.get(item["severity"], ui.dim)
            findings.append(f"  {colorize(f'[{label}]')} {item['code']}: {item['message']}")
        print(
            ui.box(
                "Kestrel GGUF audit",
                "\n".join(
                    [
                        ui.kv("Verdict", verdict, value_color=ui.green if report["valid"] else ui.red),
                        ui.kv("Model", report["model"], value_color=ui.bold),
                        ui.kv("Tensors", str(report["tensor_count"])),
                        ui.kv(
                            "Errors / warnings",
                            f"{report['errors']} / {report['warnings']}",
                            value_color=ui.yellow if report["warnings"] else None,
                        ),
                        "",
                        *findings,
                    ]
                ),
                title_color=ui.green if report["valid"] else ui.red,
            )
        )
    if not report["valid"]:
        return 1
