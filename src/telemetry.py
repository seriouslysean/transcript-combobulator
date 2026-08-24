"""Structured timing telemetry for transcription pipeline runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TELEMETRY_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp for persisted reports."""
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(started: float, finished: float) -> float:
    """Return a stable, JSON-friendly duration."""
    return round(max(0.0, finished - started), 6)


def _sum_numbers(values: Iterable[Any]) -> float:
    return round(
        sum(float(value) for value in values if isinstance(value, (int, float))),
        6,
    )


def build_run_summary(
    files: list[dict[str, Any]],
    processing_seconds: float,
    wall_seconds: float,
    max_workers: int,
) -> dict[str, Any]:
    """Aggregate per-file metrics into a session-level performance summary."""
    total_audio_seconds = _sum_numbers(
        file_metrics.get('input_audio_seconds') for file_metrics in files
    )
    total_file_seconds = _sum_numbers(
        file_metrics.get('total_seconds') for file_metrics in files
    )
    stage_names = sorted(
        {
            stage
            for file_metrics in files
            for stage in file_metrics.get('stages', {})
        }
    )
    stage_seconds = {
        stage: _sum_numbers(
            file_metrics.get('stages', {}).get(stage) for file_metrics in files
        )
        for stage in stage_names
    }
    chunks = [
        chunk
        for file_metrics in files
        for chunk in file_metrics.get('transcription', {}).get('chunks', [])
    ]
    inference_seconds = _sum_numbers(
        chunk.get('timings', {}).get('inference_seconds') for chunk in chunks
    )
    chunk_audio_seconds = _sum_numbers(chunk.get('audio_seconds') for chunk in chunks)

    return {
        'files_total': len(files),
        'files_processed': sum(f.get('status') == 'processed' for f in files),
        'files_cached': sum(f.get('status') == 'cached' for f in files),
        'files_failed': sum(f.get('status') == 'error' for f in files),
        'input_audio_seconds': total_audio_seconds,
        'processing_seconds': round(processing_seconds, 6),
        'wall_seconds': round(wall_seconds, 6),
        'audio_x_realtime': (
            round(total_audio_seconds / processing_seconds, 3)
            if total_audio_seconds and processing_seconds
            else None
        ),
        'real_time_factor': (
            round(processing_seconds / total_audio_seconds, 4)
            if total_audio_seconds
            else None
        ),
        'worker_utilization': (
            round(total_file_seconds / (processing_seconds * max_workers), 3)
            if processing_seconds and max_workers
            else None
        ),
        'vad_chunks': sum(
            int(f.get('vad', {}).get('chunk_count', 0)) for f in files
        ),
        'transcript_segments': sum(
            int(f.get('transcription', {}).get('written_vtt_cue_count', 0))
            for f in files
        ),
        'raw_result_segments': sum(
            int(f.get('transcription', {}).get('raw_result_segment_count', 0))
            for f in files
        ),
        'failed_chunks': sum(chunk.get('status') == 'error' for chunk in chunks),
        'chunk_audio_seconds': chunk_audio_seconds,
        'inference_seconds': inference_seconds,
        'inference_x_realtime': (
            round(chunk_audio_seconds / inference_seconds, 3)
            if chunk_audio_seconds and inference_seconds
            else None
        ),
        'cumulative_stage_seconds': stage_seconds,
    }


def write_metrics_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically persist a telemetry report without transcript content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f'{path.suffix}.tmp')
    try:
        with open(temporary_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, sort_keys=True)
        temporary_path.replace(path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
