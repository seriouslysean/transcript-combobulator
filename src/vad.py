"""Voice Activity Detection using Silero VAD."""

import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

from src.logging_config import get_logger
logger = get_logger(__name__)

from src.config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_DURATION,
    VAD_MIN_SILENCE_DURATION,
    get_output_path_for_input
)
from src.audio_utils import validate_audio_file, needs_conversion, convert_to_wav, AudioValidationError

class VADError(Exception):
    """Base exception for VAD-related errors."""
    pass

def load_vad_model() -> Tuple[Any, Any]:
    """Load the Silero VAD model and utilities.

    Returns:
        Tuple[Any, Any]: The VAD model and utilities.

    Raises:
        VADError: If model loading fails.
    """
    try:
        model = load_silero_vad()
        return model, None  # We don't need utils since we imported them directly
    except Exception as e:
        raise VADError(f"Failed to load VAD model: {e}") from e

def process_audio(input_path: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    """Process audio file using Silero VAD to remove silence.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Tuple[Path, List[Dict[str, Any]]]: Path to output directory and list of speech segments.

    Raises:
        VADError: If audio processing fails.
    """
    if not input_path.exists():
        raise VADError(f"Input file not found: {input_path}")

    try:
        # Validate audio file (should already be converted WAV)
        audio_info = validate_audio_file(input_path)
        logger.info(f"Processing audio: {audio_info['sample_rate']}Hz, {audio_info['channels']} channel(s), {audio_info['duration']:.2f}s")
        
        # Expect WAV file at this point
        if input_path.suffix.lower() != '.wav':
            raise VADError(f"Expected WAV file but got: {input_path.suffix}")
        
        working_path = input_path

    except AudioValidationError as e:
        raise VADError(f"Audio validation failed: {e}") from e

    try:
        # Create output directory for this audio file, preserving input structure
        output_dir = get_output_path_for_input(input_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load VAD model
        model, _ = load_vad_model()

        # Load audio (using the working path which may be converted)
        wav, sr = torchaudio.load(working_path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)

        # Convert stereo to mono by averaging channels if needed
        if wav.shape[0] == 2:
            wav = wav.mean(dim=0, keepdim=True)

        # Normalize audio to [-1, 1] range
        wav = wav / (wav.abs().max() + 1e-8)

        # Get speech timestamps with more sensitive settings
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            return_seconds=True,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
        )

        if not speech_timestamps:
            raise VADError("No speech segments detected in audio")

        # Extract speech segments and save them
        processed_segments = []
        padding_samples = int(0.5 * SAMPLE_RATE)  # 500ms padding on each side
        
        logger.info(f"Found {len(speech_timestamps)} speech segments in {input_path.name}")

        for i, ts in enumerate(speech_timestamps):
            # Add padding to segment boundaries
            start = max(0, int(ts['start'] * SAMPLE_RATE) - padding_samples)
            end = min(wav.shape[1], int(ts['end'] * SAMPLE_RATE) + padding_samples)
            segment = wav[:, start:end]

            # Use the input file stem (e.g., player name) in the segment filename
            segment_path = output_dir / f"{input_path.stem}_segment_{i:03d}.wav"
            sf.write(segment_path.absolute(), segment.T.numpy(), SAMPLE_RATE)

            processed_segments.append({
                'start_seconds': ts['start'],
                'end_seconds': ts['end'],
                'segment_file': str(segment_path)
            })

        # Return the output directory and segments
        return output_dir, processed_segments

    except Exception as e:
        raise VADError(f"Failed to process audio: {e}") from e
