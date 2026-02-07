"""Transcription module for audio files.

This module provides functions for transcribing audio files using Whisper.
It supports both direct transcription and VAD-based segmentation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import os
import datetime
import re

from src.logging_config import get_logger
logger = get_logger(__name__)

from src.process_audio import process_audio
from src.whisper import transcribe_audio_segments, WhisperError
from src.config import (
    OUTPUT_DIR,
    TRANSCRIPTIONS_DIR,
)

class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass

def transcribe_segments(audio_path: Path, original_input_path: Path = None, progress_callback=None) -> Dict:
    """Transcribe existing segments of an audio file that has already been processed with VAD.
    This is a specialized wrapper that handles pre-processed segments, utilizing the core
    transcribe_audio functionality.

    Args:
        audio_path: Path to the converted/processed audio file
        original_input_path: Optional path to the original input file for proper directory structure

    Returns:
        Dict containing transcription results and metadata

    Raises:
        TranscriptionError: If transcription fails
    """
    try:
        # We should look for the mapping file in the output directory
        # The output_file is the WAV file that's been processed by VAD
        # The mapping file should be in the same directory
        if original_input_path:
            # Use original input path for proper directory structure
            from src.config import get_output_path_for_input
            output_dir = get_output_path_for_input(original_input_path)
        else:
            # Fallback to using the converted file's parent directory
            output_dir = audio_path.parent

        # First try with the exact stem name
        mapping_path = output_dir / f"{audio_path.stem}_mapping.json"
        logger.info(f"Looking for mapping file at: {mapping_path}")

        if not mapping_path.exists():
            # Check if the mapping file was saved in the output directory
            logger.error(f"Mapping file not found at {mapping_path}")
            raise TranscriptionError(f"Mapping file not found for {audio_path.name}")

        # Load the mapping file
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        # Validate mapping structure
        if 'segments' not in mapping_data:
            raise TranscriptionError(f"Invalid mapping file format for {audio_path.name}")

        logger.info(f"Processing user: {audio_path.name}")
        # Use the core transcribe_audio function with the pre-processed segments
        result = transcribe_audio(audio_path, pre_processed_mapping=mapping_data['segments'], original_input_path=original_input_path, progress_callback=progress_callback)

        return {
            'vtt_file': str(output_dir / f"{audio_path.stem}.vtt"),
            'json_file': str(output_dir / f"{audio_path.stem}_transcription.json"),
            'mapping_file': str(mapping_path),
            'segments': result['segments']
        }

    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e

def transcribe_audio(audio_path: Path, pre_processed_mapping: List[Dict] = None, original_input_path: Path = None, progress_callback=None) -> Dict[str, Any]:
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
        logger.error(f"Audio file not found: {audio_path}")
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if not audio_path.suffix.lower() == '.wav':
        logger.error(f"Expected WAV file but got: {audio_path.suffix}")
        raise TranscriptionError(f"Expected WAV file but got: {audio_path.suffix}")

    try:
        # Create output directory for this audio file using the same logic as VAD
        from src.config import get_output_path_for_input
        # Use original input path if provided for proper directory structure
        path_for_output = original_input_path if original_input_path else audio_path
        output_dir = get_output_path_for_input(path_for_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get segments either from VAD or pre-processed mapping
        if pre_processed_mapping is None:
            logger.info("Processing audio with VAD...")
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
            logger.info("Using pre-processed segments...")
            mapping = pre_processed_mapping
            mapping_file = output_dir / f"{audio_path.stem}_mapping.json"

        # Create output path for user-specific combined VTT
    # Extract username from path like "3-username_16khz" -> "username"
        username_match = re.match(r'\d+-(.+)_16khz', audio_path.stem)
        if username_match:
            username = username_match.group(1)
            output_vtt = output_dir / f"{username}_combined.vtt"
        else:
            # Fallback to original name if pattern doesn't match
            output_vtt = output_dir / f"{audio_path.stem}.vtt"

        # Prepare segments for transcription
        segments_to_transcribe = []
        for segment in mapping:
            segment_path = Path(segment['segment_file'])
            if not segment_path.exists():
                logger.warning(f"Segment file not found: {segment_path}")
                continue
            segments_to_transcribe.append((segment_path, segment['start_seconds']))

        if not segments_to_transcribe:
            raise TranscriptionError(f"No valid segments found for {audio_path.name}")

        logger.info(f"Found {len(segments_to_transcribe)} segments to transcribe")
        logger.info("Transcribing segments...")
        segments = transcribe_audio_segments(segments_to_transcribe, output_vtt, progress_callback=progress_callback)

        # Save combined transcription results as a single JSON file
        logger.info("Saving transcription results...")
        output_json = output_dir / f"{audio_path.stem}_transcription.json"
        # Note: The mapping details are omitted from this combined JSON.
        result = {
            'audio_path': str(audio_path),
            'segments': segments,
            'mapping_file': str(mapping_file)
        }

        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("Transcription complete")
        return result

    except WhisperError as e:
        logger.error(f"Failed to transcribe audio: {e}")
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
