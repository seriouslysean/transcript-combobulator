"""Transcription module for audio files.

This module provides functions for transcribing audio files using Whisper.
It supports both direct transcription and VAD-based segmentation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import os
import datetime

from src.process_audio import process_audio
from src.whisper import transcribe_audio_segments, WhisperError
from src.config import (
    OUTPUT_DIR,
    TRANSCRIPTIONS_DIR,
)

class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass

def transcribe_segments(audio_path: Path) -> Dict:
    """Transcribe existing segments of an audio file that has already been processed with VAD.
    This is a specialized wrapper that handles pre-processed segments, utilizing the core
    transcribe_audio functionality.

    Args:
        audio_path: Path to the original audio file

    Returns:
        Dict containing transcription results and metadata

    Raises:
        TranscriptionError: If transcription fails
    """
    try:
        # Get the mapping file path
        mapping_path = OUTPUT_DIR / audio_path.stem / f"{audio_path.stem}_mapping.json"
        if not mapping_path.exists():
            raise TranscriptionError(f"Mapping file not found for {audio_path.name}")

        # Load the mapping file
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        # Validate mapping structure
        if 'segments' not in mapping_data:
            raise TranscriptionError(f"Invalid mapping file format for {audio_path.name}")

        # Use the core transcribe_audio function with the pre-processed segments
        result = transcribe_audio(audio_path, pre_processed_mapping=mapping_data['segments'])

        return {
            'vtt_file': str(OUTPUT_DIR / audio_path.stem / f"{audio_path.stem}.vtt"),
            'json_file': str(OUTPUT_DIR / audio_path.stem / f"{audio_path.stem}_transcription.json"),
            'mapping_file': str(mapping_path),
            'segments': result['segments']
        }

    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e

def transcribe_audio(audio_path: Path, pre_processed_mapping: List[Dict] = None) -> Dict[str, Any]:
    """Transcribe audio file using Whisper.

    This function handles the complete pipeline:
    1. Process audio with VAD to get segments (unless pre-processed segments are provided)
    2. Transcribe each segment
    3. Combine results into a single output

    Args:
        audio_path: Path to the audio file to transcribe.
        pre_processed_mapping: Optional list of pre-processed segments to use instead of running VAD

    Returns:
        Dict[str, Any]: Transcription results including text and timestamps (without mapping details).

    Raises:
        TranscriptionError: If transcription fails.
    """
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if not audio_path.suffix.lower() == '.wav':
        print(f"Error: Audio file must be a WAV file: {audio_path}")
        raise TranscriptionError(f"Audio file must be a WAV file: {audio_path}")

    try:
        # Create output directory for this audio file
        output_dir = OUTPUT_DIR / audio_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get segments either from VAD or pre-processed mapping
        if pre_processed_mapping is None:
            print("Processing audio with VAD...")
            _, mapping = process_audio(audio_path)
            mapping_file = output_dir / f"{audio_path.stem}_mapping.json"

            # Save mapping file (contains segmentation details)
            mapping_data = {
                'original_file': str(audio_path),
                'sample_rate': 16000,  # TODO: Get this from the audio file dynamically
                'segments': mapping,
                'created_at': datetime.datetime.now().isoformat()
            }
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
        else:
            print("Using pre-processed segments...")
            mapping = pre_processed_mapping
            mapping_file = output_dir / f"{audio_path.stem}_mapping.json"

        # Create output path for VTT (if still needed)
        output_vtt = output_dir / f"{audio_path.stem}.vtt"

        # Prepare segments for transcription
        segments_to_transcribe = []
        for segment in mapping:
            segment_path = Path(segment['segment_file'])
            if not segment_path.exists():
                print(f"Warning: Segment file not found: {segment_path}")
                continue
            segments_to_transcribe.append((segment_path, segment['start_seconds']))

        if not segments_to_transcribe:
            raise TranscriptionError(f"No valid segments found for {audio_path.name}")

        print(f"Found {len(segments_to_transcribe)} segments to transcribe")
        print("Transcribing segments...")
        segments = transcribe_audio_segments(segments_to_transcribe, output_vtt)

        # Save combined transcription results as a single JSON file
        print("Saving transcription results...")
        output_json = output_dir / f"{audio_path.stem}_transcription.json"
        # Note: The mapping details are omitted from this combined JSON.
        result = {
            'audio_path': str(audio_path),
            'segments': segments,
            'mapping_file': str(mapping_file)
        }

        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        print("Transcription complete")
        return result

    except WhisperError as e:
        print(f"Error: Failed to transcribe audio: {e}")
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
    except Exception as e:
        print(f"Error: Failed to transcribe audio: {e}")
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
