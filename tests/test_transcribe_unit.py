"""Fast unit tests for the transcription glue."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from transcript_combobulator.transcribe import TranscriptionError, transcribe_audio


def test_offset_uses_padded_clip_start(tmp_path: Path) -> None:
    audio = tmp_path / "3-nilbits.wav"
    audio.touch()
    clip = tmp_path / "3-nilbits_segment_000.wav"
    clip.touch()
    mapping = [{
        "start_seconds": 10.0,
        "end_seconds": 12.0,
        "clip_start_seconds": 9.7,
        "clip_end_seconds": 12.3,
        "segment_file": str(clip),
    }]
    captured = {}

    def fake_transcribe(segments, output_vtt, progress_callback=None, metrics=None, **kwargs):
        captured["segments"] = segments
        return []

    with patch("transcript_combobulator.transcribe.get_output_path_for_input", return_value=tmp_path), \
         patch("transcript_combobulator.transcribe.transcribe_audio_segments", side_effect=fake_transcribe):
        transcribe_audio(audio, pre_processed_mapping=mapping)

    assert captured["segments"] == [(clip, 9.7)]


def test_offset_falls_back_to_speech_start_for_old_mappings(tmp_path: Path) -> None:
    audio = tmp_path / "3-nilbits.wav"
    audio.touch()
    clip = tmp_path / "seg.wav"
    clip.touch()
    captured = {}

    def fake_transcribe(segments, output_vtt, progress_callback=None, metrics=None, **kwargs):
        captured["segments"] = segments
        return []

    with patch("transcript_combobulator.transcribe.get_output_path_for_input", return_value=tmp_path), \
         patch("transcript_combobulator.transcribe.transcribe_audio_segments", side_effect=fake_transcribe):
        transcribe_audio(
            audio,
            pre_processed_mapping=[{"start_seconds": 10.0, "end_seconds": 12.0, "segment_file": str(clip)}],
        )

    assert captured["segments"] == [(clip, 10.0)]


def test_silent_track_writes_empty_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "5-afk.wav"
    audio.touch()
    metrics = {}
    with patch("transcript_combobulator.transcribe.get_output_path_for_input", return_value=tmp_path), \
         patch("transcript_combobulator.transcribe.transcribe_audio_segments") as whisper_call:
        result = transcribe_audio(audio, pre_processed_mapping=[], metrics=metrics)

    whisper_call.assert_not_called()
    assert result["segments"] == []
    assert (tmp_path / "5-afk.vtt").read_text(encoding="utf-8") == "WEBVTT\n\n"
    assert json.loads((tmp_path / "5-afk_transcription.json").read_text())["segments"] == []
    assert metrics["chunk_count"] == 0
    assert metrics["written_vtt_cue_count"] == 0


def test_mapping_with_only_missing_files_is_still_an_error(tmp_path: Path) -> None:
    audio = tmp_path / "3-nilbits.wav"
    audio.touch()
    with patch("transcript_combobulator.transcribe.get_output_path_for_input", return_value=tmp_path), \
         pytest.raises(TranscriptionError, match="No valid segments"):
        transcribe_audio(
            audio,
            pre_processed_mapping=[{"start_seconds": 0.0, "end_seconds": 1.0, "segment_file": str(tmp_path / "gone.wav")}],
        )
