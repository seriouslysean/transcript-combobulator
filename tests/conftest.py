#!/usr/bin/env python3
"""Shared test fixtures."""

from pathlib import Path
import pytest
from tools.create_test_files import create_padded_audio

@pytest.fixture(scope="session", autouse=True)
def setup_test_files():
    """Copy test files to tmp/input and clean up after all tests."""
    # Create tmp/input directory if it doesn't exist
    input_dir = Path('tmp/input')
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create test files
    from tools.create_test_files import main as create_test_files
    create_test_files()

    yield

    # Clean up test files
    (input_dir / 'test_jfk.wav').unlink(missing_ok=True)
    (input_dir / 'test_jfk_padded.wav').unlink(missing_ok=True)
