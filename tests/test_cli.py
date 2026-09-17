"""Console entry point."""

import subprocess
import sys
from pathlib import Path

import pytest

from transcript_combobulator.cli import COMMANDS, main


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "transcript_combobulator", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_top_level_help_lists_every_command(capsys) -> None:
    main([])
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_command_has_help(command: str, tmp_path: Path) -> None:
    result = _run(command, "--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_unknown_command_exits_2(tmp_path: Path) -> None:
    result = _run("nope", cwd=tmp_path)
    assert result.returncode == 2
    assert "Unknown command" in result.stderr


def test_help_does_not_import_torch() -> None:
    result = _run("--help")
    assert result.returncode == 0
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, transcript_combobulator.cli as c; c.main(['--help']); print('torch' in sys.modules)"],
        capture_output=True, text=True,
    )
    assert probe.stdout.strip().endswith("False"), probe.stdout
