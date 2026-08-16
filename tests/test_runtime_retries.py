import io

import pytest

from kestrel.cli import runtime
from kestrel.cli.runtime import _is_startup_oom, _lower_memory_command
from kestrel.errors import BackendError


def test_startup_oom_requires_specific_memory_failure():
    assert _is_startup_oom("CUDA error: out of memory", 2.0)
    assert _is_startup_oom("CUDA_ERROR_OUT_OF_MEMORY", 2.0)
    assert not _is_startup_oom("CUDA error: invalid argument", 2.0)
    assert not _is_startup_oom("failed to allocate device buffer", 180.0)


def test_lower_memory_command_adjusts_both_supported_controls():
    original = ["llama-cli", "-ub", "128", "--fit-target", "1024"]
    retry = _lower_memory_command(original)

    assert retry == (
        ["llama-cli", "-ub", "64", "--fit-target", "1536"],
        ["micro-batch 64", "a larger VRAM margin"],
    )
    assert original == ["llama-cli", "-ub", "128", "--fit-target", "1024"]


def test_lower_memory_command_stops_when_nothing_can_change():
    assert _lower_memory_command(["llama-cli", "-ub", "16"]) is None
    assert _lower_memory_command(["llama-cli", "-ub", "invalid"]) is None
    assert _lower_memory_command(["llama-cli"]) is None


def test_lower_memory_command_moves_partial_experts_back_to_cpu():
    retry = _lower_memory_command(
        ["llama-cli", "-ngl", "54", "--n-cpu-moe", "45", "-ub", "256", "--fit-target", "1473"]
    )
    assert retry == (
        ["llama-cli", "-ngl", "54", "--n-cpu-moe", "49", "-ub", "128", "--fit-target", "1985"],
        ["micro-batch 128", "CPU experts through layer 49", "a larger VRAM margin"],
    )


def test_lower_memory_command_adjusts_effective_last_duplicate():
    retry = _lower_memory_command(
        ["llama-cli", "-ub", "128", "--fit-target", "1024", "-ub", "64", "--fit-target", "2048"]
    )

    assert retry == (
        ["llama-cli", "-ub", "128", "--fit-target", "1024", "-ub", "32", "--fit-target", "2560"],
        ["micro-batch 32", "a larger VRAM margin"],
    )


def test_oom_runner_does_not_repeat_an_identical_command(monkeypatch, capsys):
    commands = []
    popen_kwargs = []

    class FailedProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            popen_kwargs.append(kwargs)
            self.stderr = io.StringIO("CUDA error: out of memory\n")

        def wait(self, timeout=None):
            return 7

    monkeypatch.setattr(runtime.subprocess, "Popen", FailedProcess)

    child_output = io.StringIO()
    assert runtime._run_with_oom_retries(["llama-cli"], max_retries=2, stdout=child_output) == 7
    assert commands == [["llama-cli"]]
    assert popen_kwargs[0]["stdout"] is child_output
    assert "not repeating the same launch" in capsys.readouterr().err


def test_oom_runner_normalizes_keyboard_interrupt_to_130(monkeypatch):
    class InterruptedProcess:
        def __init__(self, _command, **_kwargs):
            self.stderr = io.StringIO()
            self.waits = 0
            self.terminated = False

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return -15

        def terminate(self):
            self.terminated = True

    process = InterruptedProcess([])
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: process)

    assert runtime._run_with_oom_retries(["llama-cli"]) == 130
    assert process.terminated


def test_oom_runner_types_process_spawn_failure(monkeypatch):
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("not executable")),
    )

    with pytest.raises(BackendError, match="could not start llama.cpp"):
        runtime._run_with_oom_retries(["llama-cli"])
