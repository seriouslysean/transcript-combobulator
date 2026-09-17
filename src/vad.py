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
    VAD_PACK_GAP_SECONDS,
    VAD_PACK_ISLANDS,
    VAD_PACK_MAX_SECONDS,
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


def pack_islands(
    islands: list[tuple[int, int]],
    max_samples: int,
    gap_samples: int,
) -> list[list[int]]:
    """Greedily group consecutive islands so each clip stays within max_samples.

    ``islands`` are (start, end) sample bounds already including padding. An
    island longer than max_samples gets its own clip (whisper seeks through
    it as it does today). Returns index groups in order.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    current_len = 0
    for idx, (start, end) in enumerate(islands):
        length = end - start
        extra = length if not current else gap_samples + length
        if current and current_len + extra > max_samples:
            groups.append(current)
            current, current_len = [], 0
            extra = length
        current.append(idx)
        current_len += extra
    if current:
        groups.append(current)
    return groups


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

        islands: list[dict[str, Any]] = [
            {
                'speech_start': float(ts['start']),
                'speech_end': float(ts['end']),
                'start': max(0, int(ts['start'] * SAMPLE_RATE) - padding_samples),
                'end': min(wav.shape[1], int(ts['end'] * SAMPLE_RATE) + padding_samples),
            }
            for ts in speech_timestamps
        ]
        groups = pack_islands(
            [(isl['start'], isl['end']) for isl in islands],
            max_samples=int(VAD_PACK_MAX_SECONDS * SAMPLE_RATE),
            gap_samples=int(VAD_PACK_GAP_SECONDS * SAMPLE_RATE),
        ) if VAD_PACK_ISLANDS else [[i] for i in range(len(islands))]
        if VAD_PACK_ISLANDS:
            logger.info(f"Packed {len(islands)} islands into {len(groups)} clips")

        gap = torch.zeros((1, int(VAD_PACK_GAP_SECONDS * SAMPLE_RATE)), dtype=wav.dtype)
        segments: list[dict[str, Any]] = []
        for i, group in enumerate(groups):
            first, last = islands[group[0]], islands[group[-1]]
            segment_path = output_dir / f"{input_path.stem}_segment_{i:03d}.wav"
            # start/end are the detected speech bounds; clip_* are the bounds
            # of the audio actually written, which include the padding. Whisper
            # timestamps are relative to the clip, so the clip start is the
            # offset to add back for a single island. Packed clips carry a
            # piece list instead (see src.timemap).
            entry: dict[str, Any] = {
                'start_seconds': first['speech_start'],
                'end_seconds': last['speech_end'],
                'clip_start_seconds': first['start'] / SAMPLE_RATE,
                'clip_end_seconds': last['end'] / SAMPLE_RATE,
                'segment_file': str(segment_path),
            }
            if len(group) == 1 and not VAD_PACK_ISLANDS:
                sf.write(str(segment_path), wav[:, first['start']:first['end']].T.numpy(), SAMPLE_RATE)
            else:
                parts: list[torch.Tensor] = []
                pieces: list[dict[str, float]] = []
                cursor = 0
                for n, idx in enumerate(group):
                    isl = islands[idx]
                    if n:
                        parts.append(gap)
                        cursor += gap.shape[1]
                    length = isl['end'] - isl['start']
                    pieces.append({
                        'clip_offset_seconds': cursor / SAMPLE_RATE,
                        'source_start_seconds': isl['start'] / SAMPLE_RATE,
                        'duration_seconds': length / SAMPLE_RATE,
                    })
                    parts.append(wav[:, isl['start']:isl['end']])
                    cursor += length
                sf.write(str(segment_path), torch.cat(parts, dim=1).T.numpy(), SAMPLE_RATE)
                entry['pieces'] = pieces
            segments.append(entry)

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
