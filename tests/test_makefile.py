"""Shell-injection regression tests for the Makefile."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run_make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


@pytest.mark.parametrize(
    "target,var",
    [
        ("run", "folder"),
        ("run-single", "file"),
        ("combine-transcripts", "session"),
        ("filter-vtt", "file"),
    ],
)
def test_backtick_injection_no_exec(target: str, var: str, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    payload = f"x`touch {marker}`"
    result = _run_make("CLI=echo", target, f"{var}={payload}")
    assert not marker.exists(), (
        f"{target}: shell injection created {marker}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_run_single_filename_with_space(tmp_path: Path) -> None:
    audio = tmp_path / "a b.wav"
    audio.touch()
    result = _run_make("CLI=echo", "run-single", f"file={audio}")
    assert result.returncode == 0, result.stderr
    assert f"file {audio}" in result.stdout


@pytest.mark.parametrize("payload", ["x`touch {m}`", "x$(touch {m})", "a b"])
def test_env_file_reaches_the_cli_unexpanded(payload: str, tmp_path: Path) -> None:
    # The real CLI line, with a stub combobulator that prints what it was given.
    marker = tmp_path / "marker"
    stub = tmp_path / "combobulator"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$ENV_FILE"\n')
    stub.chmod(0o755)
    value = payload.format(m=marker)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["make", "VENV_CMD=", "combine-transcripts"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**env, "ENV_FILE": value, "session": "s"},
    )
    assert not marker.exists(), result.stderr
    assert result.returncode == 0, result.stderr
    assert value in result.stdout.splitlines()
