"""Transcription pipeline: VAD -> per-segment whisper -> combined user VTT."""

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from src.config import get_output_path_for_input
from src.logging_config import get_logger
from src.vad import process_audio
from src.whisper import WhisperError, transcribe_audio_segments

logger = get_logger(__name__)

_USERNAME_PATTERN = re.compile(r'\d+-(.+)_16khz')


class TranscriptionError(Exception):
    """Base exception for transcription errors."""


def _extract_username_vtt_name(stem: str) -> str:
    """Given a stem like '3-username_16khz', return 'username_combined.vtt'."""
    m = _USERNAME_PATTERN.match(stem)
    return f"{m.group(1)}_combined.vtt" if m else f"{stem}.vtt"


def transcribe_segments(
    audio_path: Path,
    original_input_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict[str, Any]:
    """Transcribe a pre-VAD-processed audio file by reading its mapping JSON.

    Expects <audio_path.stem>_mapping.json to exist in the output directory
    (written by src.vad.process_audio).
    """
    try:
        output_dir = (
            get_output_path_for_input(original_input_path)
            if original_input_path
            else audio_path.parent
        )
        mapping_path = output_dir / f"{audio_path.stem}_mapping.json"
        logger.info(f"Looking for mapping file at: {mapping_path}")

        if not mapping_path.exists():
            raise TranscriptionError(f"Mapping file not found for {audio_path.name}")

        with open(mapping_path) as f:
            mapping_data = json.load(f)

        if 'segments' not in mapping_data:
            raise TranscriptionError(f"Invalid mapping file format for {audio_path.name}")

        logger.info(f"Processing user: {audio_path.name}")
        result = transcribe_audio(
            audio_path,
            pre_processed_mapping=mapping_data['segments'],
            original_input_path=original_input_path,
            progress_callback=progress_callback,
        )

        return {
            'vtt_file': str(output_dir / _extract_username_vtt_name(audio_path.stem)),
            'json_file': str(output_dir / f"{audio_path.stem}_transcription.json"),
            'mapping_file': str(mapping_path),
            'segments': result['segments'],
        }

    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e


def transcribe_audio(
    audio_path: Path,
    pre_processed_mapping: Optional[list[dict[str, Any]]] = None,
    original_input_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict[str, Any]:
    """Run the full transcription pipeline for a single audio file.

    If ``pre_processed_mapping`` is provided, VAD is skipped and the given
    segments are transcribed directly. Otherwise VAD runs first.
    """
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() != '.wav':
        raise TranscriptionError(f"Expected WAV file but got: {audio_path.suffix}")

    try:
        path_for_output = original_input_path or audio_path
        output_dir = get_output_path_for_input(path_for_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if pre_processed_mapping is None:
            logger.info("Processing audio with VAD...")
            _, mapping = process_audio(audio_path)
        else:
            logger.info("Using pre-processed segments...")
            mapping = pre_processed_mapping

        mapping_file = output_dir / f"{audio_path.stem}_mapping.json"
        output_vtt = output_dir / _extract_username_vtt_name(audio_path.stem)

        segments_to_transcribe: list[tuple[Path, float]] = []
        for segment in mapping:
            segment_path = Path(segment['segment_file'])
            if not segment_path.exists():
                logger.warning(f"Segment file not found: {segment_path}")
                continue
            segments_to_transcribe.append((segment_path, segment['start_seconds']))

        if not segments_to_transcribe:
            raise TranscriptionError(f"No valid segments found for {audio_path.name}")

        logger.info(f"Found {len(segments_to_transcribe)} segments to transcribe")
        segments = transcribe_audio_segments(
            segments_to_transcribe, output_vtt, progress_callback=progress_callback
        )

        logger.info("Saving transcription results...")
        output_json = output_dir / f"{audio_path.stem}_transcription.json"
        result = {
            'audio_path': str(audio_path),
            'segments': segments,
            'mapping_file': str(mapping_file),
        }
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("Transcription complete")
        return result

    except WhisperError as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
