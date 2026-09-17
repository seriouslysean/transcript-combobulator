"""Voice Activity Detection using Silero VAD.

Takes a 16kHz mono WAV (produced by transcript_combobulator.audio_utils.convert_to_wav), runs
Silero VAD to find speech regions, writes each region as its own segment
WAV, and writes a mapping JSON describing all segments.

Memory is bounded by the block size, not the track length: the model's
frame probabilities are collected while streaming the WAV in blocks with the
detector's state carried across block boundaries, and each speech region is
written by seeking into the WAV. The region logic is a verbatim port of
silero-vad 6.2.1 ``get_speech_timestamps`` (which only accepts a whole-file
tensor); ``tests/test_vad_streaming.py`` asserts identical output against
the installed library, so a silero upgrade that changes it fails loudly.
"""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from silero_vad import load_silero_vad

from transcript_combobulator.audio_utils import AudioValidationError, validate_audio_file
from transcript_combobulator.config import (
    ALLOW_SILENT_TRACKS,
    PADDING_SECONDS,
    VAD_BACKEND,
    SAMPLE_RATE,
    VAD_MIN_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION,
    VAD_THREADS,
    VAD_THRESHOLD,
)
from transcript_combobulator.logging_config import get_logger

logger = get_logger(__name__)


class VADError(Exception):
    """Base exception for VAD-related errors."""


@lru_cache(maxsize=1)
def load_vad_model() -> Any:
    """Load Silero VAD once per long-lived worker process (see VAD_BACKEND)."""
    try:
        return load_silero_vad(onnx=VAD_BACKEND == 'onnx')
    except Exception as e:
        raise VADError(f"Failed to load VAD model ({VAD_BACKEND}): {e}") from e


_WINDOW_SAMPLES = 512          # silero's frame at 16 kHz
_BLOCK_SAMPLES = 60 * 16000     # streaming read size; a multiple of the frame


def _stream_speech_probs(
    audio_path: Path, model: Any, block_samples: int = _BLOCK_SAMPLES
) -> tuple[list[float], int]:
    """One speech probability per 512-sample frame, reading the WAV in blocks.

    Matches silero's whole-file loop exactly: frames in order, the model's
    recurrent state carried across blocks, and a final short frame zero
    padded. Returns (probs, total_samples).
    """
    model.reset_states()
    probs: list[float] = []
    carry: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)
    with sf.SoundFile(str(audio_path)) as f:
        total = f.frames
        for block in f.blocks(blocksize=block_samples, dtype='float32', always_2d=False):
            data = block if carry.size == 0 else np.concatenate((carry, block))
            full = (len(data) // _WINDOW_SAMPLES) * _WINDOW_SAMPLES
            for start in range(0, full, _WINDOW_SAMPLES):
                frame = torch.from_numpy(np.ascontiguousarray(data[start:start + _WINDOW_SAMPLES]))
                probs.append(float(model(frame, SAMPLE_RATE).item()))
            carry = np.asarray(data[full:], dtype=np.float32).reshape(-1)
    if carry.size:
        frame = torch.nn.functional.pad(
            torch.from_numpy(carry), (0, _WINDOW_SAMPLES - len(carry))
        )
        probs.append(float(model(frame, SAMPLE_RATE).item()))
    return probs, total


def _speech_regions_from_probs(
    speech_probs: list[float],
    audio_length_samples: int,
    *,
    sampling_rate: int,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int = 30,
    max_speech_duration_s: float = float('inf'),
    neg_threshold: float | None = None,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
    return_seconds: bool = True,
    time_resolution: int = 1,
) -> list[dict[str, float]]:
    """Verbatim port of silero-vad 6.2.1 get_speech_timestamps, post-probability part."""
    window_size_samples = _WINDOW_SAMPLES
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    triggered = False
    speeches: list[dict[str, Any]] = []
    current_speech: dict[str, Any] = {}

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0
    prev_end = next_start = 0
    possible_ends: list[tuple[int, int]] = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = cur_sample
            continue

        if triggered and (cur_sample - current_speech['start'] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur
                if next_start < prev_end + cur_sample:
                    current_speech['start'] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech['end'] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end
            if not use_max_poss_sil_at_max_speech and sil_dur_now > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for speech_dict in speeches:
            speech_dict['start'] = max(round(speech_dict['start'] / sampling_rate, time_resolution), 0)
            speech_dict['end'] = min(round(speech_dict['end'] / sampling_rate, time_resolution), audio_length_seconds)

    return speeches


def detect_speech(
    audio_path: Path, model: Any, block_samples: int = _BLOCK_SAMPLES
) -> list[dict[str, float]]:
    """Speech regions in seconds for a 16 kHz mono WAV, streamed from disk.

    Same result as ``silero_vad.get_speech_timestamps`` on the whole file
    with this project's settings, without loading the whole file.
    """
    probs, total = _stream_speech_probs(audio_path, model, block_samples)
    return _speech_regions_from_probs(
        probs,
        total,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
        min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
    )


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
        total_samples = int(audio_info['frames'])

        # Silero runs frame by frame; intra-op threading only adds overhead.
        # Restore the worker's whisper thread count afterwards, even on error.
        previous_threads = torch.get_num_threads()
        if VAD_THREADS > 0:
            torch.set_num_threads(VAD_THREADS)
        try:
            speech_timestamps = detect_speech(input_path, model)
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
        with sf.SoundFile(str(input_path)) as source:
            for i, ts in enumerate(speech_timestamps):
                start = max(0, int(ts['start'] * SAMPLE_RATE) - padding_samples)
                end = min(total_samples, int(ts['end'] * SAMPLE_RATE) + padding_samples)
                source.seek(start)
                segment = source.read(end - start, dtype='float32', always_2d=False)

                segment_path = output_dir / f"{input_path.stem}_segment_{i:03d}.wav"
                sf.write(str(segment_path), segment, SAMPLE_RATE)

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
