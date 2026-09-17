"""Voice Activity Detection using Silero VAD.

Takes a 16kHz mono WAV (produced by src.audio_utils.convert_to_wav), runs
Silero VAD to find speech regions, writes each region as its own segment
WAV, and writes a mapping JSON describing all segments.
"""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from src.audio_utils import AudioValidationError, validate_audio_file
from src.config import (
    ALLOW_SILENT_TRACKS,
    PADDING_SECONDS,
    SAMPLE_RATE,
    VAD_MIN_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION,
    VAD_THREADS,
    VAD_THRESHOLD,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class VADError(Exception):
    """Base exception for VAD-related errors."""


@lru_cache(maxsize=1)
def load_vad_model() -> Any:
    """Load Silero VAD once per long-lived worker process."""
    try:
        return load_silero_vad()
    except Exception as e:
        raise VADError(f"Failed to load VAD model: {e}") from e


def process_audio(input_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Run VAD on a 16kHz mono WAV, write segment WAVs + mapping JSON.

    Returns (output_dir, segments). The segments list is also persisted to
    ``<stem>_mapping.json`` in output_dir.
    """
    if not input_path.exists():
        raise VADError(f"Input file not found: {input_path}")

    try:
        audio_info = validate_audio_file(input_path)
    except AudioValidationError as e:
        raise VADError(f"Audio validation failed: {e}") from e

    if input_path.suffix.lower() != '.wav':
        raise VADError(f"Expected WAV file but got: {input_path.suffix}")

    logger.info(
        f"VAD: {input_path.name} ({audio_info['sample_rate']}Hz, "
        f"{audio_info['channels']}ch, {audio_info['duration']:.2f}s)"
    )

    output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = load_vad_model()
        # The pipeline converts upstream; anything else is a wiring bug, and
        # silently resampling here would hide it from the cache fingerprint.
        if audio_info['sample_rate'] != SAMPLE_RATE or audio_info['channels'] != 1:
            raise VADError(
                f"Expected {SAMPLE_RATE}Hz mono, got {audio_info['sample_rate']}Hz "
                f"{audio_info['channels']}ch: run convert_to_wav first"
            )
        samples, _ = sf.read(str(input_path), dtype='float32', always_2d=False)
        wav = torch.from_numpy(samples).unsqueeze(0)

        # Silero runs frame by frame; intra-op threading only adds overhead.
        # Restore the worker's whisper thread count afterwards, even on error.
        previous_threads = torch.get_num_threads()
        if VAD_THREADS > 0:
            torch.set_num_threads(VAD_THREADS)
        try:
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                return_seconds=True,
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
            )
        finally:
            if VAD_THREADS > 0:
                torch.set_num_threads(previous_threads)

        if not speech_timestamps:
            if not ALLOW_SILENT_TRACKS:
                raise VADError("No speech segments detected in audio")
            logger.warning(
                f"No speech detected in {input_path.name}; writing an empty mapping "
                "(ALLOW_SILENT_TRACKS=false to treat this as an error)"
            )

        padding_samples = int(PADDING_SECONDS * SAMPLE_RATE)
        logger.info(f"Found {len(speech_timestamps)} speech segments in {input_path.name}")

        segments: list[dict[str, Any]] = []
        for i, ts in enumerate(speech_timestamps):
            start = max(0, int(ts['start'] * SAMPLE_RATE) - padding_samples)
            end = min(wav.shape[1], int(ts['end'] * SAMPLE_RATE) + padding_samples)
            segment = wav[:, start:end]

            segment_path = output_dir / f"{input_path.stem}_segment_{i:03d}.wav"
            sf.write(str(segment_path), segment.T.numpy(), SAMPLE_RATE)

            # start/end are the detected speech bounds; clip_* are the bounds
            # of the WAV actually written, which include the padding. Whisper
            # timestamps are relative to the clip, so the clip start is the
            # offset to add back.
            segments.append({
                'start_seconds': ts['start'],
                'end_seconds': ts['end'],
                'clip_start_seconds': start / SAMPLE_RATE,
                'clip_end_seconds': end / SAMPLE_RATE,
                'segment_file': str(segment_path),
            })

        mapping_path = output_dir / f"{input_path.stem}_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(
                {
                    'original_file': str(input_path),
                    'sample_rate': SAMPLE_RATE,
                    'segments': segments,
                    'created_at': datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        return output_dir, segments

    except VADError:
        raise
    except Exception as e:
        raise VADError(f"Failed to process audio: {e}") from e
