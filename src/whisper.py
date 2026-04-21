"""Whisper integration for audio transcription."""

import json
import os
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import whisper

from src.config import (
    WHISPER_CONFIDENCE_THRESHOLD,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WHISPER_MODELS_DIR,
    WHISPER_PROMPT,
    get_whisper_options,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

# Suppress FutureWarning from torch.load regarding weights_only=False
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Whisper hallucinates repeated single tokens on long silences / laughs.
# Any string that is the same short word repeated this many times is collapsed.
_REPETITION_THRESHOLD = 6


class WhisperError(Exception):
    """Base exception for whisper-related errors."""


def get_whisper_device() -> str:
    device = os.getenv('WHISPER_DEVICE', WHISPER_DEVICE)
    if device not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f"Invalid WHISPER_DEVICE: {device}. Must be cpu, cuda, or mps.")
    return device


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
    """Load a whisper model from the local models dir.

    Raises WhisperError if the model file is missing — run `make setup-whisper`.
    """
    model_name = model_name or os.getenv('WHISPER_MODEL', WHISPER_MODEL)
    model_path = WHISPER_MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise WhisperError(
            f"Model file not found: {model_path}. Run `make setup-whisper` first."
        )
    os.environ['WHISPER_MODELS_DIR'] = str(WHISPER_MODELS_DIR)
    return whisper.load_model(
        model_name,
        device=get_whisper_device(),
        download_root=str(WHISPER_MODELS_DIR),
    )


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
        out.append({"start": start, "end": end, "text": text, "confidence": confidence})
    return out


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


def transcribe_segment(
    audio_path: Path,
    output_path: Optional[Path] = None,
    offset: float = 0.0,
    model: Optional[whisper.Whisper] = None,
) -> list[dict[str, Any]]:
    """Transcribe a single audio file, optionally writing a VTT."""
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    try:
        model = model or load_whisper_model()
        result = model.transcribe(str(audio_path), **get_whisper_options())
        segments = _segments_from_result(result, offset=offset)
        if output_path and segments:
            _write_vtt(output_path, segments)
        return segments
    except Exception as e:
        raise WhisperError(f"Failed to transcribe segment: {e}") from e


def transcribe_audio_segments(
    segments: list[tuple[Path, float]],
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[dict[str, Any]]:
    """Transcribe a list of (segment_path, start_offset) tuples with a shared model.

    Dedupes identical lines across segments in the output VTT.
    """
    total = len(segments)
    logger.info("Loading Whisper model...")
    if progress_callback:
        progress_callback("loading", 0, total)

    model = load_whisper_model()
    logger.info(f"VAD found {total} segments")

    all_segments: list[dict[str, Any]] = []
    for i, (segment_path, start_time) in enumerate(segments, 1):
        logger.info(f"Processing segment {i}/{total}...")
        if progress_callback:
            progress_callback("transcribing", i, total)
        try:
            all_segments.extend(transcribe_segment(segment_path, None, start_time, model))
        except Exception as e:
            logger.warning(f"Failed to transcribe segment {segment_path}: {e}")

    if output_path and all_segments:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for seg in all_segments:
            line = seg["text"].strip()
            if line and line not in seen:
                deduped.append(seg)
                seen.add(line)
        _write_vtt(output_path, deduped)
        logger.info(f"User transcript saved: {output_path.name}")

    all_segments.sort(key=lambda s: s["start"])
    return all_segments


def filter_by_confidence(
    segments: list[dict[str, Any]],
    confidence_threshold: float = WHISPER_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in segments:
        if float(s.get("confidence", 0.0)) >= confidence_threshold:
            out.append(s)
    return out


def transcribe_file_direct(
    audio_path: Path,
    output_path: Path,
    prompt: str = "",
) -> list[dict[str, Any]]:
    """Transcribe a whole audio file in one pass (no VAD segmentation).

    Used by ``regenerate_vtt_for_audio``. For the standard pipeline, use
    ``src.transcribe.transcribe_audio`` which goes through VAD.
    """
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    try:
        model = load_whisper_model()
        opts = get_whisper_options()
        if prompt:
            opts['initial_prompt'] = prompt
        result = model.transcribe(str(audio_path), **opts)
        segments = _segments_from_result(result)
        for seg in segments:
            if seg["text"]:
                logger.info(
                    f"  [{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}] {seg['text']}"
                )
        _write_vtt(output_path, segments)
        return segments
    except Exception as e:
        raise WhisperError(f"Failed to transcribe audio: {e}") from e


def regenerate_vtt_with_confidence(
    json_path: Path,
    output_vtt: Path,
    confidence_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Rewrite a VTT from a saved segments JSON, optionally filtering by confidence."""
    if not json_path.exists():
        raise WhisperError(f"JSON file not found: {json_path}")

    try:
        with open(json_path) as f:
            segments: list[dict[str, Any]] = json.load(f)
        if confidence_threshold is not None:
            segments = filter_by_confidence(segments, confidence_threshold)
        _write_vtt(output_vtt, segments)
        return segments
    except Exception as e:
        raise WhisperError(f"Failed to regenerate VTT: {e}") from e


def regenerate_vtt_for_audio(
    audio_path: Path,
    confidence_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Re-transcribe an audio file and rewrite its VTT, with optional confidence filter."""
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    json_path = audio_path.with_suffix(".json")
    output_vtt = audio_path.with_suffix(".vtt")

    segments = transcribe_file_direct(audio_path, output_vtt, prompt=WHISPER_PROMPT)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    if confidence_threshold is not None:
        segments = regenerate_vtt_with_confidence(json_path, output_vtt, confidence_threshold)

    return segments
