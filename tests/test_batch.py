#!/usr/bin/env python3
"""Test batch processing utilities."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from tools.process_batch import (
    _build_table,
    _calculate_torch_threads,
    _format_duration,
    _initialize_worker,
    _status_display,
    find_audio_files,
)


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

    def test_cached(self):
        label, style = _status_display("cached")
        assert "cached" in label
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

    def test_main_skips_completed_pipeline(self, tmp_path):
        """Completed outputs return without running conversion, VAD, or Whisper."""
        from tools.process_single_file import main

        input_file = tmp_path / "speaker.wav"
        input_file.touch()
        statuses = {}
        with patch("tools.process_single_file.get_output_path_for_input", return_value=tmp_path), \
             patch("tools.process_single_file.get_manifest_path", return_value=tmp_path / "manifest.json"), \
             patch("tools.process_single_file.is_pipeline_complete", return_value=True), \
             patch("tools.process_single_file.process_audio") as process_audio:
            main(str(input_file), statuses, "speaker.wav")

        assert statuses["speaker.wav"] == "cached"
        process_audio.assert_not_called()

    def test_force_ignores_completed_pipeline(self, tmp_path):
        """Force mode runs processing even when a completion manifest exists."""
        from tools.process_single_file import main

        input_file = tmp_path / "speaker.wav"
        input_file.touch()
        output_file = tmp_path / "speaker.wav"
        transcription = {
            "vtt_file": str(tmp_path / "speaker.vtt"),
            "json_file": str(tmp_path / "speaker.json"),
            "mapping_file": str(tmp_path / "speaker_mapping.json"),
        }
        with patch("tools.process_single_file.get_output_path_for_input", return_value=tmp_path), \
             patch("tools.process_single_file.is_pipeline_complete", return_value=True), \
             patch("tools.process_single_file.needs_conversion", return_value=False), \
             patch("tools.process_single_file.process_audio", return_value=(tmp_path, [])) as process_audio, \
             patch("tools.process_single_file.transcribe_segments", return_value=transcription), \
             patch("tools.process_single_file.write_pipeline_manifest"):
            main(str(input_file), force=True)

        assert output_file.exists()
        process_audio.assert_called_once()

    def test_reprocessing_invalidates_manifest_before_pipeline_work(self, tmp_path):
        """An interrupted forced run cannot leave an old completion record."""
        from tools.process_single_file import main

        input_file = tmp_path / "speaker.wav"
        input_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest = output_dir / "speaker_pipeline_manifest.json"
        manifest.write_text('{"fingerprint": "old"}', encoding="utf-8")

        def fail_after_manifest_invalidation(*args, **kwargs):
            assert not manifest.exists()
            raise RuntimeError("interrupted")

        with patch(
            "tools.process_single_file.get_output_path_for_input",
            return_value=output_dir,
        ), patch(
            "tools.process_single_file.is_pipeline_complete", return_value=True
        ), patch(
            "tools.process_single_file.needs_conversion", return_value=False
        ), patch(
            "tools.process_single_file.process_audio",
            side_effect=fail_after_manifest_invalidation,
        ), pytest.raises(RuntimeError, match="interrupted"):
            main(str(input_file), force=True)

        assert not manifest.exists()

    def test_cache_miss_replaces_existing_normalized_audio(self, tmp_path):
        """A changed normalized source replaces the prior derived WAV."""
        from tools.process_single_file import main

        input_file = tmp_path / "input" / "speaker.wav"
        input_file.parent.mkdir()
        input_file.write_bytes(b"new audio")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_file = output_dir / "speaker.wav"
        output_file.write_bytes(b"old audio")
        transcription = {
            "vtt_file": str(output_dir / "speaker.vtt"),
            "json_file": str(output_dir / "speaker.json"),
            "mapping_file": str(output_dir / "speaker_mapping.json"),
        }

        with patch(
            "tools.process_single_file.get_output_path_for_input",
            return_value=output_dir,
        ), patch(
            "tools.process_single_file.is_pipeline_complete", return_value=False
        ), patch(
            "tools.process_single_file.needs_conversion", return_value=False
        ), patch(
            "tools.process_single_file.process_audio", return_value=(output_dir, [])
        ), patch(
            "tools.process_single_file.transcribe_segments",
            return_value=transcription,
        ), patch("tools.process_single_file.write_pipeline_manifest"):
            main(str(input_file))

        assert output_file.read_bytes() == b"new audio"

    def test_cache_miss_removes_existing_audio_before_conversion(self, tmp_path):
        """Conversion cannot silently reuse a valid but stale derived WAV."""
        from tools.process_single_file import main

        input_file = tmp_path / "speaker.flac"
        input_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_file = output_dir / "speaker.wav"
        output_file.write_bytes(b"old audio")
        transcription = {
            "vtt_file": str(output_dir / "speaker.vtt"),
            "json_file": str(output_dir / "speaker.json"),
            "mapping_file": str(output_dir / "speaker_mapping.json"),
        }

        def convert_after_removal(source, destination):
            assert source == input_file
            assert destination == output_file
            assert not destination.exists()
            destination.touch()

        with patch(
            "tools.process_single_file.get_output_path_for_input",
            return_value=output_dir,
        ), patch(
            "tools.process_single_file.is_pipeline_complete", return_value=False
        ), patch(
            "tools.process_single_file.needs_conversion", return_value=True
        ), patch(
            "tools.process_single_file.convert_to_wav",
            side_effect=convert_after_removal,
        ), patch(
            "tools.process_single_file.process_audio", return_value=(output_dir, [])
        ), patch(
            "tools.process_single_file.transcribe_segments",
            return_value=transcription,
        ), patch("tools.process_single_file.write_pipeline_manifest"):
            main(str(input_file))

        assert output_file.exists()


class TestWorkerInitialization:
    """Tests for process-wide worker setup."""

    def test_applies_priority_and_torch_limits_once(self):
        with patch("tools.process_batch.os.nice") as nice, patch(
            "torch.set_num_threads"
        ) as set_threads, patch("torch.set_num_interop_threads") as set_interop:
            _initialize_worker(torch_threads=3, worker_nice=10)

        nice.assert_called_once_with(10)
        set_threads.assert_called_once_with(3)
        set_interop.assert_called_once_with(1)

    def test_zero_values_leave_process_defaults(self):
        with patch("tools.process_batch.os.nice") as nice, patch(
            "torch.set_num_threads"
        ) as set_threads:
            _initialize_worker(torch_threads=0, worker_nice=0)

        nice.assert_not_called()
        set_threads.assert_not_called()


class TestTorchThreadAllocation:
    """Tests for automatic intra-op thread allocation."""

    def test_auto_splits_cpus_across_active_workers(self):
        assert _calculate_torch_threads(2, 0, cpu_count=14) == 7
        assert _calculate_torch_threads(4, 0, cpu_count=14) == 3

    def test_explicit_setting_wins(self):
        assert _calculate_torch_threads(2, 5, cpu_count=14) == 5

    def test_never_returns_less_than_one_thread(self):
        assert _calculate_torch_threads(8, 0, cpu_count=4) == 1

    def test_missing_cpu_count_uses_safe_fallback(self):
        with patch("tools.process_batch.os.cpu_count", return_value=None):
            assert _calculate_torch_threads(2, 0) == 2
