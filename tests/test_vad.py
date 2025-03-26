#!/usr/bin/env python3
"""Test Voice Activity Detection."""

from pathlib import Path
import pytest
from src.vad import load_vad_model, process_audio, VADError
import json

def test_vad_model_loading():
    """Test that the VAD model loads successfully."""
    model, _ = load_vad_model()
    assert model is not None, "VAD model is None"

def test_vad_detection():
    """Test that VAD detects speech segments in the test JFK file."""
    input_file = Path('tmp/input/test_jfk.wav')
    assert input_file.exists(), "Test JFK file not found"

    # Process the audio
    output_path, segments = process_audio(input_file)

    # Verify output files
    assert output_path.exists(), "Processed audio file not created"
    mapping_file = output_path.parent / f"{input_file.stem}_mapping.json"
    assert mapping_file.exists(), "Mapping file not created"

    # Verify mapping content
    with open(mapping_file) as f:
        mapping = json.load(f)
        assert 'segments' in mapping, "No segments in mapping"
        assert len(mapping['segments']) > 0, "Empty segments in mapping"

        # Verify segment structure
        for segment in mapping['segments']:
            assert 'start_seconds' in segment, "Missing start time"
            assert 'end_seconds' in segment, "Missing end time"
            assert 'segment_file' in segment, "Missing segment file"
            assert segment['end_seconds'] > segment['start_seconds'], "Invalid segment timing"
            assert Path(segment['segment_file']).exists(), "Segment file not created"
