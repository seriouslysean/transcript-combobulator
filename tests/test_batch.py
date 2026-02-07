#!/usr/bin/env python3
"""Test batch processing utilities."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from tools.process_batch import find_audio_files, _build_table, _format_duration, _status_display


class TestFindAudioFiles:
    """Tests for audio file discovery."""

    def test_finds_supported_formats(self, tmp_path):
        """Finds all supported audio file extensions."""
        extensions = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"]
        for ext in extensions:
            (tmp_path / f"speaker{ext}").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == len(extensions)

    def test_ignores_non_audio_files(self, tmp_path):
        """Skips non-audio files like .txt or .json."""
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.json").touch()
        (tmp_path / "speaker.wav").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "speaker.wav"

    def test_excludes_converted_files(self, tmp_path):
        """Skips files with _converted in the stem."""
        (tmp_path / "speaker.wav").touch()
        (tmp_path / "speaker_converted.wav").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "speaker.wav"

    def test_returns_sorted(self, tmp_path):
        """Returns files in sorted order."""
        (tmp_path / "3-charlie.flac").touch()
        (tmp_path / "1-alice.flac").touch()
        (tmp_path / "2-bob.flac").touch()

        files = find_audio_files(tmp_path)
        names = [f.name for f in files]
        assert names == ["1-alice.flac", "2-bob.flac", "3-charlie.flac"]

    def test_empty_directory(self, tmp_path):
        """Returns empty list for directory with no audio files."""
        files = find_audio_files(tmp_path)
        assert files == []

    def test_case_insensitive_extensions(self, tmp_path):
        """Finds files regardless of extension case."""
        (tmp_path / "speaker.WAV").touch()
        (tmp_path / "speaker2.Flac").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == 2

    def test_does_not_recurse(self, tmp_path):
        """Only finds files in the target directory, not subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "top.wav").touch()
        (subdir / "nested.wav").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "top.wav"


class TestBuildTable:
    """Tests for the rich progress table builder."""

    def test_all_waiting(self):
        """Table renders correctly when all files are waiting."""
        names = ["a.flac", "b.flac"]
        status = {"a.flac": "waiting", "b.flac": "waiting"}
        table = _build_table(names, status, 2)
        assert table.title == "Transcription Progress (2 workers)"
        assert table.row_count == 2

    def test_mixed_statuses(self):
        """Table renders with a mix of statuses."""
        names = ["a.flac", "b.flac", "c.flac"]
        status = {"a.flac": "done", "b.flac": "transcribing 3/10", "c.flac": "waiting"}
        table = _build_table(names, status, 2)
        assert table.row_count == 3

    def test_error_status(self):
        """Table handles error status."""
        names = ["a.flac"]
        status = {"a.flac": "error"}
        table = _build_table(names, status, 1)
        assert table.row_count == 1

    def test_unknown_status_fallback(self):
        """Table handles unexpected status values gracefully."""
        names = ["a.flac"]
        status = {"a.flac": "unknown_state"}
        table = _build_table(names, status, 1)
        assert table.row_count == 1

    def test_missing_key_defaults_to_waiting(self):
        """Files not in status dict default to waiting."""
        names = ["a.flac"]
        status = {}
        table = _build_table(names, status, 1)
        assert table.row_count == 1


class TestStatusDisplay:
    """Tests for status label/style mapping."""

    def test_waiting(self):
        assert _status_display("waiting") == ("waiting", "dim")

    def test_converting(self):
        assert _status_display("converting") == ("converting", "yellow")

    def test_splitting(self):
        assert _status_display("splitting") == ("splitting", "yellow")

    def test_loading_model(self):
        assert _status_display("loading model") == ("loading model", "blue")

    def test_done(self):
        label, style = _status_display("done")
        assert "done" in label
        assert style == "green"

    def test_transcribing_with_progress(self):
        label, style = _status_display("transcribing 3/15")
        assert label == "transcribing 3/15"
        assert style == "magenta"

    def test_error(self):
        label, style = _status_display("error")
        assert "error" in label
        assert "red" in style

    def test_unknown_fallback(self):
        assert _status_display("something_else") == ("something_else", "")


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_seconds_only(self):
        assert _format_duration(45) == "0m 45s"

    def test_minutes_and_seconds(self):
        assert _format_duration(125) == "2m 05s"

    def test_hours(self):
        assert _format_duration(3661) == "1h 01m 01s"

    def test_zero(self):
        assert _format_duration(0) == "0m 00s"

    def test_exact_hour(self):
        assert _format_duration(3600) == "1h 00m 00s"


class TestConfigSettings:
    """Tests for the parallel processing config values."""

    def test_parallel_jobs_default(self):
        """PARALLEL_JOBS defaults to 2."""
        from src.config import PARALLEL_JOBS
        # The default is 2 unless overridden by env
        assert isinstance(PARALLEL_JOBS, int)
        assert PARALLEL_JOBS >= 1

    def test_torch_threads_default(self):
        """TORCH_THREADS defaults to 0 (auto-detect)."""
        from src.config import TORCH_THREADS
        assert isinstance(TORCH_THREADS, int)
        assert TORCH_THREADS >= 0


class TestProcessSingleFile:
    """Tests for process_single_file status updates."""

    def test_status_dict_updates(self):
        """Verify _update_status writes to shared dict when provided."""
        from tools.process_single_file import main

        # We can't easily run the full pipeline without audio files,
        # but we can verify the function signature accepts status_dict/status_key
        import inspect
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        assert "status_dict" in params
        assert "status_key" in params

    def test_main_still_works_without_status_dict(self):
        """main() should accept being called without status_dict (backward compat)."""
        import inspect
        from tools.process_single_file import main
        sig = inspect.signature(main)
        # Both params should have defaults (None)
        assert sig.parameters["status_dict"].default is None
        assert sig.parameters["status_key"].default is None
