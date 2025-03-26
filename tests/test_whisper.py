#!/usr/bin/env python3
"""Test Whisper transcription."""

from pathlib import Path
import pytest
from src.transcribe import transcribe_audio
from src.config import TRANSCRIPTIONS_DIR, OUTPUT_DIR
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

    # Verify output files
    vtt_file = TRANSCRIPTIONS_DIR / f"{input_file.stem}.vtt"
    json_file = OUTPUT_DIR / f"{input_file.stem}_transcription.json"

    assert vtt_file.exists(), "VTT file not created"
    assert json_file.exists(), "JSON file not created"

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
