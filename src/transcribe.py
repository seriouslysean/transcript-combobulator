"""Transcription pipeline: VAD -> per-segment whisper -> combined user VTT."""

import json
from pathlib import Path
from typing import Any, Callable, Optional

from src.config import (
    ALLOW_SILENT_TRACKS,
    get_output_path_for_input,
    vtt_name_for_stem,
    vtt_path_for_input,
)
from src.logging_config import get_logger
from src.vad import process_audio
from src.whisper import WhisperError, _write_vtt, transcribe_audio_segments

logger = get_logger(__name__)

__all__ = ['TranscriptionError', 'transcribe_segments', 'transcribe_audio', 'vtt_path_for_input']


class TranscriptionError(Exception):
    """Base exception for transcription errors."""



def transcribe_segments(
    audio_path: Path,
    original_input_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    metrics: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_key: Optional[str] = None,
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
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
        )

        return {
            'vtt_file': str(output_dir / vtt_name_for_stem(audio_path.stem)),
            'json_file': str(output_dir / f"{audio_path.stem}_transcription.json"),
            'mapping_file': str(mapping_path),
            'segments': result['segments'],
            'metrics': result.get('metrics', {}),
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
    metrics: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_key: Optional[str] = None,
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
        output_vtt = output_dir / vtt_name_for_stem(audio_path.stem)

        segments_to_transcribe: list[tuple[Path, float]] = []
        for segment in mapping:
            segment_path = Path(segment['segment_file'])
            if not segment_path.exists():
                logger.warning(f"Segment file not found: {segment_path}")
                continue
            # Whisper's timestamps are relative to the clip, which starts
            # PADDING_SECONDS before the detected speech. Older mappings lack
            # clip_start_seconds and fall back to the (late) speech start.
            offset = float(segment.get('clip_start_seconds', segment['start_seconds']))
            segments_to_transcribe.append((segment_path, offset))

        transcription_metrics = metrics if metrics is not None else {}
        output_json = output_dir / f"{audio_path.stem}_transcription.json"

        if not segments_to_transcribe:
            if mapping or not ALLOW_SILENT_TRACKS:
                raise TranscriptionError(f"No valid segments found for {audio_path.name}")
            # A silent track: no speech was detected, so there is nothing to
            # transcribe. Write an empty transcript rather than fail the batch.
            logger.warning(f"No speech segments for {audio_path.name}; writing empty transcript")
            _write_vtt(output_vtt, [])
            transcription_metrics.update({
                'chunk_count': 0,
                'failed_chunk_count': 0,
                'raw_result_segment_count': 0,
                'written_vtt_cue_count': 0,
                'chunks': [],
                'total_seconds': 0.0,
            })
            result: dict[str, Any] = {
                'audio_path': str(audio_path),
                'segments': [],
                'mapping_file': str(mapping_file),
            }
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2)
            return {**result, 'metrics': transcription_metrics}

        logger.info(f"Found {len(segments_to_transcribe)} segments to transcribe")
        segments = transcribe_audio_segments(
            segments_to_transcribe,
            output_vtt,
            progress_callback=progress_callback,
            metrics=transcription_metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
        )

        logger.info("Saving transcription results...")
        result = {
            'audio_path': str(audio_path),
            'segments': segments,
            'mapping_file': str(mapping_file),
        }
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("Transcription complete")
        return {**result, 'metrics': transcription_metrics}

    except WhisperError as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
