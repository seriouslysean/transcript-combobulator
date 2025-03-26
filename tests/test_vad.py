#!/usr/bin/env python3
import json
from pathlib import Path
import pytest
from src.vad import load_vad_model
from src.process_audio import process_audio

def test_vad_model_loading():
    """Test that the VAD model can be loaded."""
    model, utils = load_vad_model()
    assert model is not None, "VAD model failed to load"
    assert utils is not None, "VAD utilities failed to load"

    # Check that all required utilities are present
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    assert all([get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks]), \
        "Missing required VAD utilities"

def test_vad_detection():
    """Test VAD detection on JFK audio file."""
    input_file = Path('deps/whisper.cpp/samples/jfk.wav')
    assert input_file.exists(), (
        f"Error: {input_file} not found. "
        "Run 'make setup-whisper' to download sample files"
    )

    print(f"\nProcessing {input_file.name}...")
    processed_path = process_audio(input_file)
    assert processed_path.exists(), f"Processed file not created: {processed_path}"

    # Check mapping file
    mapping_file = processed_path.parent / f"{input_file.stem}_mapping.json"
    assert mapping_file.exists(), f"Mapping file not created: {mapping_file}"

    # Verify mapping content
    with open(mapping_file) as f:
        mapping = json.load(f)
        assert 'segments' in mapping, "No segments found in mapping"
        assert len(mapping['segments']) > 0, "No speech segments detected"

        print("\nSpeech segments detected:")
        for segment in mapping['segments']:
            assert 'start_seconds' in segment, "Missing start_seconds in segment"
            assert 'end_seconds' in segment, "Missing end_seconds in segment"
            assert 'segment_file' in segment, "Missing segment_file in segment"
            segment_path = Path(segment['segment_file'])
            assert segment_path.exists(), f"Segment file not found: {segment_path}"
            print(f"  {segment['start_seconds']:.2f}s --> {segment['end_seconds']:.2f}s ({segment_path.name})")
