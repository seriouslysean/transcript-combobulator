"""Voice Activity Detection using Silero VAD."""

import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict, Any

from src.config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_DURATION,
    VAD_MIN_SILENCE_DURATION,
    OUTPUT_DIR
)

class VADError(Exception):
    """Base exception for VAD-related errors."""
    pass

def load_vad_model() -> Tuple[torch.nn.Module, Any]:
    """Load the Silero VAD model and utilities.

    Returns:
        Tuple[torch.nn.Module, Any]: The VAD model and utilities.

    Raises:
        VADError: If model loading fails.
    """
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        return model, utils
    except Exception as e:
        raise VADError(f"Failed to load VAD model: {e}") from e

def process_audio(input_path: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    """Process audio file using Silero VAD to remove silence.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Tuple[Path, List[Dict[str, Any]]]: Path to processed audio and list of speech segments.

    Raises:
        VADError: If audio processing fails.
    """
    if not input_path.exists():
        raise VADError(f"Input file not found: {input_path}")

    if not input_path.suffix.lower() == '.wav':
        raise VADError(f"Input file must be a WAV file: {input_path}")

    try:
        # Load VAD model
        model, utils = load_vad_model()
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils

        # Load audio
        wav, sr = torchaudio.load(input_path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            return_seconds=True,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=VAD_MIN_SPEECH_DURATION,
            min_silence_duration_ms=VAD_MIN_SILENCE_DURATION,
        )

        if not speech_timestamps:
            raise VADError("No speech segments detected in audio")

        # Extract speech segments and save them
        processed_segments = []
        for i, ts in enumerate(speech_timestamps):
            start = int(ts['start'] * SAMPLE_RATE)
            end = int(ts['end'] * SAMPLE_RATE)
            segment = wav[:, start:end]

            # Save individual segment
            segment_path = OUTPUT_DIR / f"{input_path.stem}_segment_{i}.wav"
            sf.write(segment_path, segment.T.numpy(), SAMPLE_RATE)

            processed_segments.append({
                'start_seconds': ts['start'],
                'end_seconds': ts['end'],
                'segment_file': str(segment_path)
            })

        # Save processed audio with all segments concatenated
        processed_wav = torch.cat([wav[:, int(ts['start'] * SAMPLE_RATE):int(ts['end'] * SAMPLE_RATE)] for ts in speech_timestamps], dim=1)
        output_path = OUTPUT_DIR / f"{input_path.stem}_processed.wav"
        sf.write(output_path, processed_wav.T.numpy(), SAMPLE_RATE)

        return output_path, processed_segments

    except Exception as e:
        raise VADError(f"Failed to process audio: {e}") from e
