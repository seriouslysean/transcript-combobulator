"""Whisper transcription utilities."""

from pathlib import Path
import json
from typing import Dict, Any, List

from src.config import OUTPUT_DIR, TRANSCRIPTIONS_DIR
from src.whisper_cpp import transcribe_audio_segments, WhisperCppError
from src.process_audio import process_audio

class TranscriptionError(Exception):
    """Base exception for transcription-related errors."""
    pass

def transcribe_audio(audio_path: Path) -> Dict[str, Any]:
    """Transcribe audio file using whisper.cpp.

    Args:
        audio_path: Path to the audio file to transcribe.

    Returns:
        Dict[str, Any]: Transcription results including text and timestamps.

    Raises:
        TranscriptionError: If transcription fails.
    """
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if not audio_path.suffix.lower() == '.wav':
        raise TranscriptionError(f"Audio file must be a WAV file: {audio_path}")

    try:
        # First, process audio with VAD to get segments
        processed_path = process_audio(audio_path)
        mapping_file = processed_path.parent / f"{audio_path.stem}_mapping.json"

        with open(mapping_file) as f:
            mapping = json.load(f)

        # Create output paths - preserve test_ prefix if present
        stem = audio_path.stem
        output_vtt = TRANSCRIPTIONS_DIR / f"{stem}.vtt"
        output_json = OUTPUT_DIR / f"{stem}_transcription.json"

        # Prepare segments for transcription
        segments_to_transcribe = [
            {
                'audio_path': segment['segment_file'],
                'start': segment['start_seconds']
            }
            for segment in mapping['segments']
        ]

        print(f"Found {len(segments_to_transcribe)} segments to transcribe")

        # Transcribe the audio segments
        segments = transcribe_audio_segments(segments_to_transcribe, output_vtt)

        # Save transcription results as JSON
        result = {
            'audio_path': str(audio_path),
            'segments': segments
        }

        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    except WhisperCppError as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
