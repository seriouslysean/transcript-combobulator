"""Config path anchoring."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root_dir_seen_from(cwd: Path, extra_env: dict[str, str] | None = None) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    env.pop("PROJECT_ROOT", None)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", "from src.config import ROOT_DIR; print(ROOT_DIR)"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_root_dir_does_not_depend_on_cwd(tmp_path: Path) -> None:
    assert _root_dir_seen_from(tmp_path) == str(REPO_ROOT)


def test_project_root_env_overrides(tmp_path: Path) -> None:
    override = tmp_path / "elsewhere"
    override.mkdir()
    assert _root_dir_seen_from(tmp_path, {"PROJECT_ROOT": str(override)}) == str(
        override.resolve()
    )


def test_relative_env_file_resolves_against_repo_when_absent_from_cwd(
    tmp_path: Path,
) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "ENV_FILE": ".env.jfk-sample"}
    env.pop("PROJECT_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-c", "import os, src.config; print(os.environ['WHISPER_MODEL'])"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = [
        line.split("=", 1)[1].strip()
        for line in (REPO_ROOT / ".env.jfk-sample").read_text().splitlines()
        if line.startswith("WHISPER_MODEL=")
    ][0]
    assert result.stdout.strip() == expected


def test_missing_env_file_is_an_error(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "ENV_FILE": ".env.anihilation"}
    env.pop("PROJECT_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-c", "import src.config"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ENV_FILE='.env.anihilation' not found" in result.stderr


def test_temperature_scalar_or_fallback_tuple() -> None:
    from src.config import _parse_temperature

    assert _parse_temperature("0.0") == 0.0
    assert _parse_temperature("0.0, 0.2,0.4") == (0.0, 0.2, 0.4)
    assert _parse_temperature("") == 0.0
    assert _parse_temperature("warm") == 0.0
