#!/usr/bin/env python3
"""Test transcription functionality."""

from pathlib import Path
import pytest
from src.transcribe import transcribe_audio
from src.config import OUTPUT_DIR
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
    original_vtt = OUTPUT_DIR / original_file.stem / f"{original_file.stem}.vtt"
    with open(original_vtt) as f:
        original_text = f.read()

    # Get the padded transcription
    padded_vtt = OUTPUT_DIR / padded_file.stem / f"{padded_file.stem}.vtt"
    with open(padded_vtt) as f:
        padded_text = f.read()

    # Split padded text into segments (split on double newline which separates VTT entries)
    segments = [s.strip() for s in padded_text.split('\n\n') if s.strip()]
    # Remove the WEBVTT header
    segments = segments[1:]

    # Whisper segment count is nondeterministic; the padded file should produce
    # at least 2 segments (original speech repeated across silence gaps) and
    # shouldn't exceed a reasonable upper bound.
    assert len(segments) >= 2, \
        f"Expected at least 2 segments in padded file, got {len(segments)}"
    assert len(segments) <= 6, \
        f"Expected at most 6 segments in padded file, got {len(segments)}"

    # Get just the text content from original (remove WEBVTT header and timestamps)
    original_lines = [line for line in original_text.split('\n') if '-->' not in line and 'WEBVTT' not in line]
    original_content = ' '.join(line.strip() for line in original_lines if line.strip()).lower()

    # Concatenate all padded segment text
    all_segment_text = []
    for segment in segments:
        segment_lines = [line for line in segment.split('\n') if '-->' not in line]
        segment_content = ' '.join(line.strip() for line in segment_lines if line.strip())
        all_segment_text.append(segment_content)
    combined_padded = ' '.join(all_segment_text).lower()

    # The padded file repeats the same speech across silence gaps, so the combined
    # text should contain the original words. Use SequenceMatcher for a robust
    # similarity check that handles repeated/reordered words properly.
    similarity = similar(original_content, combined_padded)
    assert similarity >= 0.3, \
        f"Padded transcription too different from original (similarity: {similarity:.2%})\n" \
        f"Original: {original_content}\n" \
        f"Padded:   {combined_padded}"
