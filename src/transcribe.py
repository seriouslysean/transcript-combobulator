"""Whisper transcription utilities."""

import whisper
from pathlib import Path
from typing import Dict, Any
import json

from src.config import (
    WHISPER_MODEL,
    WHISPER_PROMPT,
    WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURE,
    WHISPER_LANGUAGE,
    OUTPUT_DIR
)

class TranscriptionError(Exception):
    """Base exception for transcription-related errors."""
    pass

def transcribe_audio(audio_path: Path) -> Dict[str, Any]:
    """Transcribe audio file using Whisper.

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
        # Load Whisper model
        model = whisper.load_model(WHISPER_MODEL)

        # Transcribe audio
        result = model.transcribe(
            str(audio_path),
            prompt=WHISPER_PROMPT,
            beam_size=WHISPER_BEAM_SIZE,
            temperature=WHISPER_TEMPERATURE,
            language=WHISPER_LANGUAGE,
            fp16=False  # Ensure we use FP32 for CPU
        )

        # Save transcription
        output_path = OUTPUT_DIR / f"{audio_path.stem}_transcription.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
