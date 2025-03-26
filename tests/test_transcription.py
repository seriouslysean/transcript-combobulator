#!/usr/bin/env python3
"""Test transcription functionality."""

from pathlib import Path
import pytest
from src.transcribe import transcribe_audio
from src.config import TRANSCRIPTIONS_DIR, OUTPUT_DIR
import json
from difflib import SequenceMatcher

def similar(a: str, b: str) -> float:
    """Calculate string similarity ratio."""
    return SequenceMatcher(None, a, b).ratio()

def test_segmented_transcription():
    """Test that transcribing a padded file produces similar content to the original."""
    # Get paths to test files
    original_file = Path('tmp/input/test_jfk.wav')
    padded_file = Path('tmp/input/test_jfk_padded.wav')
    assert original_file.exists(), "Original test file not found"
    assert padded_file.exists(), "Padded test file not found"

    # Transcribe both files
    original_result = transcribe_audio(original_file)
    padded_result = transcribe_audio(padded_file)

    # Get the full original transcription
    original_vtt = TRANSCRIPTIONS_DIR / 'test_jfk.vtt'
    with open(original_vtt) as f:
        original_text = f.read()

    # Get the padded transcription
    padded_vtt = TRANSCRIPTIONS_DIR / 'test_jfk_padded.vtt'
    with open(padded_vtt) as f:
        padded_text = f.read()

    # Split padded text into segments (split on double newline which separates VTT entries)
    segments = [s.strip() for s in padded_text.split('\n\n') if s.strip()]
    # Remove the WEBVTT header
    segments = segments[1:]

    # Verify we have exactly 3 segments
    assert len(segments) == 3, \
        f"Expected 3 segments in padded file, got {len(segments)}"

    # Get just the text content from original (remove WEBVTT header and timestamps)
    original_lines = [line for line in original_text.split('\n') if '-->' not in line and 'WEBVTT' not in line]
    original_content = '\n'.join(line for line in original_lines if line.strip())

    # Verify each segment contains similar text to the original
    for i, segment in enumerate(segments):
        # Get just the text content (remove timestamps)
        segment_lines = [line for line in segment.split('\n') if '-->' not in line]
        segment_content = '\n'.join(line for line in segment_lines if line.strip())

        similarity = similar(original_content, segment_content)
        assert similarity >= 0.95, \
            f"Segment {i+1} similarity too low: {similarity:.2%}\n" \
            f"Original:\n{original_content}\n" \
            f"Segment {i+1}:\n{segment_content}"
