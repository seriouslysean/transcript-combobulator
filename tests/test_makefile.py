"""Shell-injection regression tests for the Makefile."""

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


@pytest.mark.parametrize("var", ["ref", "hyp", "json"])
def test_compare_transcripts_injection_no_exec(var: str, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    args = {"ref": "a.txt", "hyp": "b.txt", var: f"x`touch {marker}`$(touch {marker})*"}
    result = _run_make(
        "CLI=echo", "compare-transcripts", *(f"{k}={v}" for k, v in args.items())
    )
    assert not marker.exists(), result.stderr
    assert result.returncode == 0, result.stderr


def test_compare_transcripts_splits_file_lists() -> None:
    result = _run_make(
        "CLI=echo", "compare-transcripts", "ref=a-1.txt a-2.txt", "hyp=b-1.txt b-2.txt"
    )
    assert result.returncode == 0, result.stderr
    assert "compare --ref a-1.txt a-2.txt --hyp b-1.txt b-2.txt" in result.stdout
