#!/usr/bin/env python3
"""Test Whisper transcription."""

from pathlib import Path
import pytest
from src.transcribe import transcribe_audio
from src.config import OUTPUT_DIR
import json

def test_basic_transcription():
    """Test transcription of the original JFK audio file."""
    input_file = Path('tmp/input/test_jfk.wav')
    assert input_file.exists(), "Test JFK file not found"

    # Transcribe the file
    result = transcribe_audio(input_file)

    # Verify result structure
    assert 'segments' in result, "No segments in transcription result"
    assert len(result['segments']) > 0, "No segments were transcribed"

    # Verify output files — transcribe_audio writes to OUTPUT_DIR/<stem>/
    output_subdir = OUTPUT_DIR / input_file.stem
    vtt_file = output_subdir / f"{input_file.stem}.vtt"
    json_file = output_subdir / f"{input_file.stem}_transcription.json"

    assert vtt_file.exists(), f"VTT file not created at {vtt_file}"
    assert json_file.exists(), f"JSON file not created at {json_file}"

    # Verify VTT content
    with open(vtt_file) as f:
        vtt_content = f.read()
        assert 'WEBVTT' in vtt_content, "Invalid VTT format"
        assert '-->' in vtt_content, "No timestamps in VTT"

    # Verify JSON content
    with open(json_file) as f:
        json_content = json.load(f)
        assert 'segments' in json_content, "No segments in JSON output"
        assert len(json_content['segments']) > 0, "Empty segments in JSON output"

        # Verify segment structure
        for segment in json_content['segments']:
            assert 'start' in segment, "Missing start time"
            assert 'end' in segment, "Missing end time"
            assert 'text' in segment, "Missing text"
            assert segment['end'] > segment['start'], "Invalid segment timing"
            assert len(segment['text'].strip()) > 0, "Empty segment text"
