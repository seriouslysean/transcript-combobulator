#!/usr/bin/env python3
"""Shared test fixtures."""

from pathlib import Path

import pytest

from tools.create_sample_files import create_sample_files


@pytest.fixture(scope="session", autouse=True)
def setup_test_files():
    """Create tmp/input/test_jfk*.wav before tests, remove after."""
    input_dir = Path('tmp/input')
    input_dir.mkdir(parents=True, exist_ok=True)

    create_sample_files(prefix="test_", copies=1, padded_copies=3)

    yield

    (input_dir / 'test_jfk.wav').unlink(missing_ok=True)
    (input_dir / 'test_jfk_padded.wav').unlink(missing_ok=True)
