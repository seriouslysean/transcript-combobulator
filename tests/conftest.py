#!/usr/bin/env python3
"""Shared test fixtures."""

import shutil
from pathlib import Path

import pytest

from src.config import INPUT_DIR, OUTPUT_DIR
from tools.create_sample_files import create_sample_files

# Everything the suite creates is named test_jfk*; teardown removes only that.
# tmp/output holds real session transcripts and must never be swept wholesale.
_TEST_STEM_GLOB = "test_jfk*"


def _remove_test_artifacts() -> None:
    for root in (INPUT_DIR, OUTPUT_DIR):
        for path in root.glob(_TEST_STEM_GLOB):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


@pytest.fixture(scope="session", autouse=True)
def setup_test_files():
    """Create tmp/input/test_jfk*.wav before tests; remove every test_jfk* after."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    _remove_test_artifacts()
    create_sample_files(prefix="test_", copies=1, padded_copies=3)

    yield

    _remove_test_artifacts()
