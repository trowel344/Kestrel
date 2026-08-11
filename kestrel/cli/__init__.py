"""Post-split ``kestrel.cli`` package.

The former ``kestrel/cli.py`` monolith was split into the modules in this
package. This module re-exports every legacy ``cli.*`` symbol so that
``import kestrel.cli as cli`` continues to resolve names at the package level
(patch sites in tests that touch a module owner still need to go through the
owner module, e.g. ``cli.probes.detect_gpu``).
"""

from __future__ import annotations

import subprocess  # noqa: F401  (exposed as ``cli.subprocess`` for monkeypatching)

from .. import __version__ as __version__  # noqa: F401
from . import (  # module references (``cli.probes`` etc.) resolve here
    bench,
    convert,
    engine,
    evaluate,
    health,
    menu,
    model_source,
    models,
    nodes,
    parser,
    planning,
    probes,
    run,
    runtime,
    state,
    updater,
)
from . import main as _main  # noqa: F401
from .bench import (
    _build_optimize_profile,
    _print_benchmark_summary,
    _run_optimize_benchmark,
    _summarize_benchmark_rows,
    cmd_benchmark,
    cmd_optimize,
)
from .convert import cmd_audit, cmd_convert
from .engine import _engine_dir, cmd_build, cmd_engine
from .evaluate import cmd_evaluate
from .health import (
    _checks_status,
    _doctor_checks,
    _fmt_bytes,
    _save_default_model,
    _writable_probe,
    cmd_doctor,
    cmd_setup,
    cmd_status,
)
from .main import _run_dispatched, main
from .menu import _menu_status_compact, cmd_menu
from .model_source import (
    _cached_gguf_path,
    _ensure_local_gguf,
    _model_profile,
    _resolve_hf_model_dir,
    _resolve_model_alias,
    _resolve_model_source,
    _safetensors_info,
    _safetensors_size,
    detect_model,
    read_gguf_config,
)
from .models import (
    _models_files,
    _models_import,
    _models_info,
    _models_list,
    _models_pull,
    _models_recommend,
    _models_search,
    cmd_models,
)
from .nodes import cmd_nodes
from .parser import (
    _add_json_flag,
    _add_local_run_options,
    _default_model,
    _kestrel_version,
    build_parser,
)
from .planning import (
    _context_size_arg,
    _cpu_moe_thread_sweep,
    _kv_cache_bytes_per_token,
    _plan_mode,
    _select_context_size,
    estimate_config,
)
from .probes import (
    _aggregate_gpu,
    _available_ram_mib,
    _cpu_power_policy,
    _memory_snapshot,
    _warm_page_cache,
    detect_gpu,
    detect_gpus,
)
from .run import _prepare_run_model, _print_run_plan, cmd_run, cmd_serve
from .runtime import (
    _build_server_cmd,
    _configure_backend,
    _finish_json,
    _flatten_extra,
    _human_stream,
    _oneshot_run,
    _print_failure,
    _resolve_ollama_native,
    _run_with_oom_retries,
    _tensor_split_arg,
    _wait_ready,
)
from .state import reload_state
from .updater import (
    _materialize_wheel,
    _post_install_check,
    _restore_install,
    _snapshot_installed,
    cmd_self_update,
)

__all__ = [
    "bench",
    "convert",
    "engine",
    "evaluate",
    "health",
    "menu",
    "model_source",
    "models",
    "nodes",
    "parser",
    "planning",
    "probes",
    "run",
    "runtime",
    "state",
    "updater",
    "subprocess",
    "_build_optimize_profile",
    "_print_benchmark_summary",
    "_run_optimize_benchmark",
    "_summarize_benchmark_rows",
    "cmd_benchmark",
    "cmd_optimize",
    "cmd_audit",
    "cmd_convert",
    "_engine_dir",
    "cmd_build",
    "cmd_engine",
    "cmd_evaluate",
    "_checks_status",
    "_doctor_checks",
    "_fmt_bytes",
    "_save_default_model",
    "_writable_probe",
    "cmd_doctor",
    "cmd_setup",
    "cmd_status",
    "_run_dispatched",
    "main",
    "_menu_status_compact",
    "cmd_menu",
    "_cached_gguf_path",
    "_ensure_local_gguf",
    "_model_profile",
    "_resolve_hf_model_dir",
    "_resolve_model_alias",
    "_resolve_model_source",
    "_safetensors_info",
    "_safetensors_size",
    "detect_model",
    "read_gguf_config",
    "_models_files",
    "_models_import",
    "_models_info",
    "_models_list",
    "_models_pull",
    "_models_recommend",
    "_models_search",
    "cmd_models",
    "cmd_nodes",
    "_add_json_flag",
    "_add_local_run_options",
    "_default_model",
    "_kestrel_version",
    "build_parser",
    "_context_size_arg",
    "_cpu_moe_thread_sweep",
    "_kv_cache_bytes_per_token",
    "_plan_mode",
    "_select_context_size",
    "estimate_config",
    "_aggregate_gpu",
    "_available_ram_mib",
    "_cpu_power_policy",
    "_memory_snapshot",
    "_warm_page_cache",
    "detect_gpu",
    "detect_gpus",
    "_prepare_run_model",
    "_print_run_plan",
    "cmd_run",
    "cmd_serve",
    "_build_server_cmd",
    "_configure_backend",
    "_finish_json",
    "_flatten_extra",
    "_human_stream",
    "_oneshot_run",
    "_print_failure",
    "_resolve_ollama_native",
    "_run_with_oom_retries",
    "_tensor_split_arg",
    "_wait_ready",
    "reload_state",
    "_materialize_wheel",
    "_post_install_check",
    "_restore_install",
    "_snapshot_installed",
    "cmd_self_update",
]
