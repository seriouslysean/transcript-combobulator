"""Structured timing telemetry for transcription pipeline runs."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TELEMETRY_SCHEMA_VERSION = 1

_STAGE_KEYS = (
    'input_probe_seconds',
    'setup_seconds',
    'cache_check_seconds',
    'cache_invalidation_seconds',
    'conversion_seconds',
    'vad_seconds',
    'transcription_seconds',
    'manifest_write_seconds',
    'vtt_rewrite_seconds',
)
_RUNTIME_NUMBER_KEYS = (
    'parallel_jobs_configured',
    'active_workers',
    'torch_threads_per_worker',
    'worker_nice',
)
_SUMMARY_NUMBER_KEYS = (
    'files_total',
    'files_processed',
    'files_cached',
    'files_failed',
    'input_audio_seconds',
    'processing_seconds',
    'wall_seconds',
    'audio_x_realtime',
    'real_time_factor',
    'worker_utilization',
    'vad_chunks',
    'transcript_segments',
    'raw_result_segments',
    'failed_chunks',
    'chunk_audio_seconds',
    'inference_seconds',
    'inference_x_realtime',
)
_FILE_NUMBER_KEYS = (
    'input_bytes',
    'input_audio_seconds',
    'artifact_count',
    'total_seconds',
)
_TRANSCRIPTION_NUMBER_KEYS = (
    'chunk_count',
    'failed_chunk_count',
    'raw_result_segment_count',
    'written_vtt_cue_count',
    'model_cache_hit',
    'model_load_seconds',
    'vtt_write_seconds',
    'total_seconds',
)
_CHUNK_NUMBER_KEYS = (
    'index',
    'offset_seconds',
    'result_segment_count',
    'text_characters',
    'audio_seconds',
)
_TIMING_NUMBER_KEYS = (
    'model_resolve_seconds',
    'audio_seconds',
    'audio_decode_seconds',
    'inference_seconds',
    'result_processing_seconds',
    'vtt_write_seconds',
    'result_segment_count',
    'text_characters',
    'total_seconds',
)
_WHISPER_NUMBER_KEYS = (
    'temperature',
    'beam_size',
    'no_speech_threshold',
    'logprob_threshold',
    'compression_ratio_threshold',
)
_WHISPER_BOOL_KEYS = (
    'fp16',
    'word_timestamps',
    'condition_on_previous_text',
    'carry_initial_prompt',
    'prompt_configured',
)
_VAD_NUMBER_KEYS = (
    'threshold',
    'min_speech_duration',
    'min_silence_duration',
    'padding_seconds',
)
_OFFICIAL_MODEL_NAMES = {
    'tiny',
    'tiny.en',
    'base',
    'base.en',
    'small',
    'small.en',
    'medium',
    'medium.en',
    'large',
    'large-v1',
    'large-v2',
    'large-v3',
    'large-v3-turbo',
    'turbo',
}
_VALID_DEVICES = {'cpu', 'cuda', 'mps'}
_RUN_STATUSES = {'completed', 'failed'}
_FILE_STATUSES = {'processed', 'cached', 'error'}
_CHUNK_STATUSES = {'processed', 'empty', 'error', 'resumed'}
_COMBINE_STATUSES = {'completed', 'error', 'skipped'}
_CONVERSION_ACTIONS = {'converted', 'copied', 'cached'}


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


def _copy_numbers(source: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is None and key in source:
            safe[key] = None
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            safe[key] = value
    return safe


def _copy_bools(
    source: Any, keys: tuple[str, ...], allow_none: bool = False
) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool) or (allow_none and value is None and key in source):
            safe[key] = value
    return safe


def _copy_enum(
    source: Any, key: str, allowed: set[str]
) -> dict[str, str]:
    if not isinstance(source, dict):
        return {}
    value = source.get(key)
    return {key: value} if isinstance(value, str) and value in allowed else {}


def _copy_error_type(source: Any) -> dict[str, str]:
    if not isinstance(source, dict):
        return {}
    value = source.get('error_type')
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', value):
        return {'error_type': value}
    return {}


def _copy_timestamps(source: Any) -> dict[str, str]:
    if not isinstance(source, dict):
        return {}
    safe: dict[str, str] = {}
    for key in ('started_at', 'finished_at'):
        value = source.get(key)
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            safe[key] = value
    return safe


def _sanitize_whisper_config(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    safe = {
        **_copy_numbers(source, _WHISPER_NUMBER_KEYS),
        **_copy_bools(source, _WHISPER_BOOL_KEYS),
    }
    model = source.get('model')
    if isinstance(model, str):
        safe['model'] = model if model in _OFFICIAL_MODEL_NAMES else 'custom'
    device = source.get('device')
    if isinstance(device, str):
        safe['device'] = device if device in _VALID_DEVICES else 'custom'
    language = source.get('language')
    if isinstance(language, str) and re.fullmatch(r'[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?', language):
        safe['language'] = language.lower()
    return safe


def sanitize_metrics_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build the persisted, identifier-free telemetry representation."""
    schema_version = report.get('schema_version')
    safe_report: dict[str, Any] = {
        'schema_version': (
            schema_version
            if isinstance(schema_version, int) and not isinstance(schema_version, bool)
            else TELEMETRY_SCHEMA_VERSION
        )
    }

    run = report.get('run')
    if isinstance(run, dict):
        safe_report['run'] = {
            **_copy_enum(run, 'status', _RUN_STATUSES),
            **_copy_timestamps(run),
            **_copy_bools(run, ('force',)),
        }
    runtime = report.get('runtime')
    if isinstance(runtime, dict):
        safe_report['runtime'] = _copy_numbers(runtime, _RUNTIME_NUMBER_KEYS)
    summary = report.get('summary')
    if isinstance(summary, dict):
        safe_summary = _copy_numbers(summary, _SUMMARY_NUMBER_KEYS)
        if 'cumulative_stage_seconds' in summary:
            safe_summary['cumulative_stage_seconds'] = _copy_numbers(
                summary.get('cumulative_stage_seconds'), _STAGE_KEYS
            )
        safe_report['summary'] = safe_summary

    configuration = report.get('configuration')
    if isinstance(configuration, dict):
        safe_report['configuration'] = {
            **_copy_numbers(configuration, ('sample_rate',)),
            'whisper': _sanitize_whisper_config(configuration.get('whisper')),
            'vad': _copy_numbers(configuration.get('vad'), _VAD_NUMBER_KEYS),
        }

    safe_files: list[dict[str, Any]] = []
    for index, file_metrics in enumerate(report.get('files', []), 1):
        if not isinstance(file_metrics, dict):
            continue
        safe_file = {
            'file_id': f'file-{index:03d}',
            **_copy_enum(file_metrics, 'status', _FILE_STATUSES),
            **_copy_numbers(file_metrics, _FILE_NUMBER_KEYS),
            **_copy_bools(file_metrics, ('forced', 'cache_hit')),
            **_copy_timestamps(file_metrics),
            **_copy_error_type(file_metrics),
        }
        if 'stages' in file_metrics:
            safe_file['stages'] = _copy_numbers(
                file_metrics.get('stages'), _STAGE_KEYS
            )
        conversion = file_metrics.get('conversion')
        if isinstance(conversion, dict):
            safe_file['conversion'] = {
                **_copy_bools(conversion, ('required',)),
                **_copy_enum(conversion, 'action', _CONVERSION_ACTIONS),
            }
        vad = file_metrics.get('vad')
        if isinstance(vad, dict):
            safe_file['vad'] = _copy_numbers(
                vad, ('chunk_count', 'speech_seconds')
            )
        transcription = file_metrics.get('transcription')
        if isinstance(transcription, dict):
            safe_transcription = {
                **_copy_numbers(transcription, _TRANSCRIPTION_NUMBER_KEYS),
                **_copy_bools(transcription, ('model_cache_hit',), allow_none=True),
                **_copy_error_type(transcription),
            }
            safe_chunks: list[dict[str, Any]] = []
            for chunk in transcription.get('chunks', []):
                if not isinstance(chunk, dict):
                    continue
                safe_chunk = {
                    **_copy_numbers(chunk, _CHUNK_NUMBER_KEYS),
                    **_copy_enum(chunk, 'status', _CHUNK_STATUSES),
                    **_copy_error_type(chunk),
                }
                timings = chunk.get('timings')
                safe_chunk['timings'] = {
                    **_copy_numbers(timings, _TIMING_NUMBER_KEYS),
                    **_copy_bools(timings, ('model_was_provided',)),
                    **_copy_enum(timings, 'status', _CHUNK_STATUSES),
                    **_copy_error_type(timings),
                }
                safe_chunks.append(safe_chunk)
            safe_transcription['chunks'] = safe_chunks
            safe_file['transcription'] = safe_transcription
        safe_files.append(safe_file)
    if 'files' in report:
        safe_report['files'] = safe_files

    combine = report.get('combine')
    if isinstance(combine, dict):
        output_files = combine.get('output_files', [])
        safe_report['combine'] = {
            **_copy_enum(combine, 'status', _COMBINE_STATUSES),
            **_copy_numbers(combine, ('seconds',)),
            **_copy_error_type(combine),
            'output_file_count': (
                len(output_files) if isinstance(output_files, list) else 0
            ),
        }

    return safe_report


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
            json.dump(sanitize_metrics_report(report), f, indent=2, sort_keys=True)
        temporary_path.replace(path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
