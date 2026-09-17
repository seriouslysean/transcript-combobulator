"""Whisper integration for audio transcription."""

import json
import os
import time
import warnings
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import soundfile as sf
import whisper

from transcript_combobulator.config import (
    DEDUPE_STRATEGY,
    DEDUPE_WINDOW_SECONDS,
    WHISPER_CONFIDENCE_THRESHOLD,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WHISPER_MODELS_DIR,
    SAMPLE_RATE,
    get_whisper_options,
)
from transcript_combobulator.logging_config import get_logger
from transcript_combobulator.pipeline_cache import ChunkCheckpoint
from transcript_combobulator.telemetry import elapsed_seconds

logger = get_logger(__name__)

# Suppress FutureWarning from torch.load regarding weights_only=False
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Whisper hallucinates repeated single tokens on long silences / laughs.
# Any string that is the same short word repeated this many times is collapsed.
_REPETITION_THRESHOLD = 6


class WhisperError(Exception):
    """Base exception for whisper-related errors."""


def get_whisper_device() -> str:
    if WHISPER_DEVICE not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f"Invalid WHISPER_DEVICE: {WHISPER_DEVICE}. Must be cpu, cuda, or mps.")
    return WHISPER_DEVICE


def format_timestamp(seconds: float) -> str:
    """Format seconds into VTT timestamp format (HH:MM:SS.mmm)."""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def collapse_repetition(text: str, threshold: int = _REPETITION_THRESHOLD) -> str:
    """Collapse degenerate whisper output like 'laughs laughs laughs …'.

    If the text is a single short word repeated >= threshold times (with only
    whitespace/punctuation between), replace with a single occurrence.
    Otherwise return the input unchanged.
    """
    if not text:
        return text
    tokens = text.split()
    if len(tokens) < threshold:
        return text
    first = tokens[0].strip(".,!?;:").lower()
    if not first or len(first) > 12:
        return text
    if all(t.strip(".,!?;:").lower() == first for t in tokens):
        return tokens[0]
    return text


def load_whisper_model(model_name: Optional[str] = None) -> whisper.Whisper:
    """Load a whisper model from the local models dir, once per worker process.

    The name is resolved here so ``load_whisper_model()`` and
    ``load_whisper_model('large-v3-turbo')`` share one cache entry instead of
    holding two 1.6 GB models. Raises WhisperError if the file is missing;
    run `make setup-whisper`.
    """
    return _load_whisper_model(model_name or WHISPER_MODEL)


@lru_cache(maxsize=None)
def _load_whisper_model(model_name: str) -> whisper.Whisper:
    model_path = WHISPER_MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise WhisperError(
            f"Model file not found: {model_path}. Run `make setup-whisper` first."
        )
    # Load by path: whisper.load_model(name) re-reads and sha256s the whole
    # checkpoint on every call (1.6 GB for large-v3-turbo, once per worker per
    # run). A path skips that but also skips the alignment heads that word
    # timestamps need, so restore them for known model names.
    model = whisper.load_model(str(model_path), device=get_whisper_device())
    alignment_heads = whisper._ALIGNMENT_HEADS.get(model_name)
    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)
    return model


def _segments_from_result(
    result: dict[str, Any], offset: float = 0.0
) -> list[dict[str, Any]]:
    """Extract [{start, end, text, confidence}] from a whisper result."""
    out: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        if not isinstance(segment, dict):
            continue
        start = float(segment.get("start", 0.0)) + offset
        end = float(segment.get("end", 0.0)) + offset
        text = collapse_repetition(str(segment.get("text", "")).strip())
        avg_logprob = float(segment.get('avg_logprob', 0))
        confidence = min(100, max(0, (1 + avg_logprob) * 100))
        # whisper records the temperature it settled on; anything above the
        # first value means its compression/logprob guard tripped and it
        # re-decoded (only possible with a WHISPER_TEMPERATURE list).
        out.append({
            "start": start,
            "end": end,
            "text": text,
            "confidence": confidence,
            "temperature": float(segment.get("temperature", 0.0)),
        })
    return out


def dedupe_segments(
    segments: list[dict[str, Any]],
    strategy: str = DEDUPE_STRATEGY,
    window_seconds: float = DEDUPE_WINDOW_SECONDS,
) -> list[dict[str, Any]]:
    """Drop repeated cues per DEDUPE_STRATEGY; blank cues are always dropped.

    'consecutive' only removes a cue whose text matches the previously kept
    cue and starts within window_seconds of its end. That is the shape of
    whisper's repeated-line hallucination; a genuine "Yeah." ten minutes later
    survives. 'global' is the legacy exact-text set across the whole file.
    """
    ordered = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    kept: list[dict[str, Any]] = []
    if strategy == 'global':
        seen: set[str] = set()
        for seg in ordered:
            line = seg["text"].strip()
            if line and line not in seen:
                kept.append(seg)
                seen.add(line)
        return kept
    prev: Optional[dict[str, Any]] = None
    for seg in ordered:
        line = seg["text"].strip()
        if not line:
            continue
        if (
            strategy == 'consecutive'
            and prev is not None
            and line == prev["text"].strip()
            and float(seg["start"]) - float(prev["end"]) <= window_seconds
        ):
            continue
        kept.append(seg)
        prev = seg
    return kept


def _write_vtt(output_path: Path, segments: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            f.write(
                f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
                f"{text}\n\n"
            )


def _load_segment_audio(audio_path: Path) -> npt.NDArray[np.float32]:
    """Decode a normalized pipeline WAV without spawning FFmpeg."""
    audio, sample_rate = sf.read(str(audio_path), dtype='float32', always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise WhisperError(
            f"Expected {SAMPLE_RATE}Hz segment but got {sample_rate}Hz: {audio_path}"
        )
    if audio.ndim != 1:
        raise WhisperError(f"Expected mono segment but got shape {audio.shape}: {audio_path}")
    return np.asarray(audio, dtype=np.float32)


def transcribe_segment(
    audio_path: Path,
    output_path: Optional[Path] = None,
    offset: float = 0.0,
    model: Optional[whisper.Whisper] = None,
    timing: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Transcribe a single audio file, optionally writing a VTT."""
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    metrics = timing if timing is not None else {}
    started = time.perf_counter()
    metrics['model_was_provided'] = model is not None
    try:
        stage_started = time.perf_counter()
        try:
            model = model or load_whisper_model()
        finally:
            metrics['model_resolve_seconds'] = elapsed_seconds(
                stage_started, time.perf_counter()
            )

        stage_started = time.perf_counter()
        try:
            audio = _load_segment_audio(audio_path)
            metrics['audio_seconds'] = round(len(audio) / SAMPLE_RATE, 6)
        finally:
            metrics['audio_decode_seconds'] = elapsed_seconds(
                stage_started, time.perf_counter()
            )

        stage_started = time.perf_counter()
        try:
            result = model.transcribe(audio, **get_whisper_options())
        finally:
            metrics['inference_seconds'] = elapsed_seconds(
                stage_started, time.perf_counter()
            )

        stage_started = time.perf_counter()
        try:
            segments = _segments_from_result(result, offset=offset)
        finally:
            metrics['result_processing_seconds'] = elapsed_seconds(
                stage_started, time.perf_counter()
            )

        stage_started = time.perf_counter()
        if output_path and segments:
            try:
                _write_vtt(output_path, segments)
            finally:
                metrics['vtt_write_seconds'] = elapsed_seconds(
                    stage_started, time.perf_counter()
                )
        else:
            metrics['vtt_write_seconds'] = 0.0
        metrics['status'] = 'processed' if segments else 'empty'
        metrics['result_segment_count'] = len(segments)
        metrics['text_characters'] = sum(len(s.get('text', '')) for s in segments)
        return segments
    except Exception as e:
        metrics['status'] = 'error'
        metrics['error_type'] = type(e).__name__
        metrics['error'] = str(e)
        raise WhisperError(f"Failed to transcribe segment: {e}") from e
    finally:
        metrics['total_seconds'] = elapsed_seconds(started, time.perf_counter())


def _whisper_model_cache_hits() -> Optional[int]:
    cache_info = getattr(_load_whisper_model, 'cache_info', None)
    if not callable(cache_info):
        return None
    try:
        hits = cache_info().hits
    except (AttributeError, TypeError):
        return None
    return hits if isinstance(hits, int) else None


def transcribe_audio_segments(
    segments: list[tuple[Path, float]],
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    metrics: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Transcribe a list of (segment_path, start_offset) tuples with a shared model.

    Dedupes identical lines across segments in the output VTT.
    """
    transcription_metrics = metrics if metrics is not None else {}
    transcription_metrics.update(
        {
            'chunk_count': len(segments),
            'chunks': [],
            'failed_chunk_count': 0,
            'raw_result_segment_count': 0,
            'written_vtt_cue_count': 0,
        }
    )
    started = time.perf_counter()
    try:
        total = len(segments)
        logger.info("Loading Whisper model...")
        if progress_callback:
            progress_callback("loading", 0, total)

        cache_hits_before = _whisper_model_cache_hits()
        stage_started = time.perf_counter()
        try:
            model = load_whisper_model()
        finally:
            transcription_metrics['model_load_seconds'] = elapsed_seconds(
                stage_started, time.perf_counter()
            )
        cache_hits_after = _whisper_model_cache_hits()
        transcription_metrics['model_cache_hit'] = (
            cache_hits_after > cache_hits_before
            if cache_hits_before is not None and cache_hits_after is not None
            else None
        )
        logger.info(f"VAD found {total} segments")

        all_segments: list[dict[str, Any]] = []
        failed_segments = 0
        checkpoint: Optional[ChunkCheckpoint] = None
        if checkpoint_path is not None and checkpoint_key:
            checkpoint = ChunkCheckpoint(checkpoint_path, checkpoint_key)
            if checkpoint.completed:
                logger.info(
                    f"Resuming: {len(checkpoint.completed)}/{total} chunks already transcribed"
                )
        resumed_segments = 0

        for i, (segment_path, start_time) in enumerate(segments, 1):
            chunk_metrics: dict[str, Any] = {
                'index': i,
                'file': segment_path.name,
                'offset_seconds': round(float(start_time), 6),
                'timings': {},
            }
            if checkpoint is not None and i in checkpoint.completed:
                saved = checkpoint.completed[i]
                chunk_segments = list(saved['segments'])
                all_segments.extend(chunk_segments)
                chunk_metrics['status'] = 'resumed'
                chunk_metrics['timings'] = dict(saved.get('timings') or {})
                chunk_metrics['result_segment_count'] = len(chunk_segments)
                chunk_metrics['audio_seconds'] = chunk_metrics['timings'].get('audio_seconds')
                transcription_metrics['chunks'].append(chunk_metrics)
                resumed_segments += 1
                if progress_callback:
                    progress_callback("transcribing", i, total)
                continue
            logger.info(f"Processing segment {i}/{total}...")
            if progress_callback:
                progress_callback("transcribing", i, total)
            try:
                chunk_segments = transcribe_segment(
                    segment_path,
                    None,
                    start_time,
                    model,
                    timing=chunk_metrics['timings'],
                )
                all_segments.extend(chunk_segments)
                chunk_metrics['status'] = (
                    'processed' if chunk_segments else 'empty'
                )
                if checkpoint is not None:
                    checkpoint.record(i, chunk_segments, chunk_metrics['timings'])
                chunk_metrics['result_segment_count'] = len(chunk_segments)
                chunk_metrics['text_characters'] = sum(
                    len(segment.get('text', '')) for segment in chunk_segments
                )
            except Exception as e:
                failed_segments += 1
                chunk_metrics['status'] = 'error'
                chunk_metrics['error_type'] = type(e).__name__
                chunk_metrics['error'] = str(e)
                logger.warning(f"Failed to transcribe segment {segment_path}: {e}")
            chunk_metrics['audio_seconds'] = chunk_metrics['timings'].get(
                'audio_seconds'
            )
            transcription_metrics['chunks'].append(chunk_metrics)

        transcription_metrics['failed_chunk_count'] = failed_segments
        transcription_metrics['resumed_chunk_count'] = resumed_segments
        transcription_metrics['fallback_segment_count'] = sum(
            1 for s in all_segments if float(s.get('temperature', 0.0)) > 0.0
        )
        transcription_metrics['raw_result_segment_count'] = len(all_segments)
        if total and failed_segments == total:
            if checkpoint is not None:
                checkpoint.close(remove=False)
            raise WhisperError(f"All {total} audio segments failed to transcribe")

        stage_started = time.perf_counter()
        try:
            if output_path:
                deduped = dedupe_segments(all_segments)
                _write_vtt(output_path, deduped)
                transcription_metrics['written_vtt_cue_count'] = len(deduped)
                logger.info(f"User transcript saved: {output_path.name}")
            if checkpoint is not None:
                # Keep the checkpoint while chunks failed so a rerun retries
                # only those; remove it once every chunk succeeded.
                checkpoint.close(remove=failed_segments == 0)
        finally:
            transcription_metrics['vtt_write_seconds'] = elapsed_seconds(
                stage_started, time.perf_counter()
            )

        all_segments.sort(key=lambda s: s["start"])
        return all_segments
    finally:
        transcription_metrics['total_seconds'] = elapsed_seconds(
            started, time.perf_counter()
        )


def filter_by_confidence(
    segments: list[dict[str, Any]],
    confidence_threshold: float = WHISPER_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in segments:
        if float(s.get("confidence", 0.0)) >= confidence_threshold:
            out.append(s)
    return out


def regenerate_vtt_with_confidence(
    json_path: Path,
    output_vtt: Path,
    confidence_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Rewrite a VTT from the pipeline's saved JSON, optionally filtered by confidence.

    Accepts the pipeline's ``<stem>_transcription.json`` (a dict with a
    ``segments`` list) or a bare list of segments. Applies the same dedup rule
    as the pipeline so the rewritten VTT matches what a fresh run would write.
    No inference happens here.
    """
    if not json_path.exists():
        raise WhisperError(f"JSON file not found: {json_path}")

    try:
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)
        segments: list[dict[str, Any]] = (
            loaded["segments"] if isinstance(loaded, dict) else loaded
        )
        if confidence_threshold is not None:
            segments = filter_by_confidence(segments, confidence_threshold)
        deduped = dedupe_segments(segments)
        _write_vtt(output_vtt, deduped)
        return deduped
    except Exception as e:
        raise WhisperError(f"Failed to regenerate VTT: {e}") from e
