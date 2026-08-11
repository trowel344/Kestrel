"""Composition root: dispatch from parsed arguments to command handlers.

Handlers return an exit code (or ``None`` for zero); :func:`_run_dispatched`
returns it so ``main`` (and thus the ``kestrel`` entry point) exits with the
handlers' real status instead of always 0.
"""

from __future__ import annotations

import sys

from ..errors import ConfigError, KestrelError
from . import runtime, state
from .bench import cmd_benchmark, cmd_optimize
from .convert import cmd_audit, cmd_convert
from .engine import cmd_build, cmd_engine
from .evaluate import cmd_evaluate
from .health import cmd_doctor, cmd_settings, cmd_setup, cmd_status
from .menu import cmd_menu
from .models import cmd_models
from .nodes import cmd_nodes
from .parser import build_parser
from .run import cmd_run, cmd_serve
from .updater import cmd_self_update

_COMMAND_HANDLERS = {
    "menu": cmd_menu,
    "status": cmd_status,
    "run": cmd_run,
    "chat": cmd_run,
    "serve": cmd_serve,
    "setup": cmd_setup,
    "settings": cmd_settings,
    "benchmark": cmd_benchmark,
    "evaluate": cmd_evaluate,
    "model-test": cmd_evaluate,
    "optimize": cmd_optimize,
    "models": cmd_models,
    "nodes": cmd_nodes,
    "build": cmd_build,
    "engine": cmd_engine,
    "self-update": cmd_self_update,
    "convert": cmd_convert,
    "audit": cmd_audit,
    "doctor": cmd_doctor,
}


def _dispatch_handler(handler, args) -> int:
    """Run a command handler without leaking rich return values to sys.exit.

    A few handlers return a report for internal callers.  Console entry points
    still have an integer-only contract: only explicit integer statuses become
    process exit codes.
    """
    result = handler(args)
    return result if isinstance(result, int) else 0


def main() -> int:
    parser = build_parser()
    return _run_dispatched(parser, parser.parse_args())


def _run_dispatched(parser, args) -> int:
    try:
        if state.CONFIG_ERROR and not (args.command == "setup" and args.reset):
            raise ConfigError(
                state.CONFIG_ERROR,
                hint="run `kestrel setup --reset` to repair it",
            )
        handler = _COMMAND_HANDLERS.get(args.command)
        if handler is not None:
            return _dispatch_handler(handler, args)
        if sys.stdin.isatty() and sys.stdout.isatty():
            return cmd_menu(args) or 0
        parser.print_help()
        return 0
    except KestrelError as exc:
        return runtime._print_failure(exc, json_output=bool(getattr(args, "json", False)))


if __name__ == "__main__":
    sys.exit(main())
