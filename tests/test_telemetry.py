"""Tests for structured pipeline telemetry."""

import json
from pathlib import Path

from src.telemetry import build_run_summary, write_metrics_report


def test_build_run_summary_aggregates_file_and_chunk_metrics() -> None:
    files = [
        {
            'status': 'processed',
            'input_audio_seconds': 120.0,
            'total_seconds': 20.0,
            'stages': {'conversion_seconds': 2.0, 'vad_seconds': 3.0},
            'vad': {'chunk_count': 2},
            'transcription': {
                'raw_result_segment_count': 3,
                'written_vtt_cue_count': 2,
                'chunks': [
                    {
                        'status': 'processed',
                        'audio_seconds': 30.0,
                        'timings': {'inference_seconds': 5.0},
                    },
                    {
                        'status': 'error',
                        'audio_seconds': 10.0,
                        'timings': {'inference_seconds': 2.0},
                    },
                ],
            },
        },
        {
            'status': 'cached',
            'input_audio_seconds': 60.0,
            'total_seconds': 1.0,
            'stages': {'cache_check_seconds': 0.1},
        },
    ]

    summary = build_run_summary(
        files,
        processing_seconds=30.0,
        wall_seconds=32.0,
        max_workers=2,
    )

    assert summary['files_processed'] == 1
    assert summary['files_cached'] == 1
    assert summary['input_audio_seconds'] == 180.0
    assert summary['audio_x_realtime'] == 6.0
    assert summary['real_time_factor'] == 0.1667
    assert summary['worker_utilization'] == 0.35
    assert summary['vad_chunks'] == 2
    assert summary['transcript_segments'] == 2
    assert summary['raw_result_segments'] == 3
    assert summary['failed_chunks'] == 1
    assert summary['inference_x_realtime'] == 5.714
    assert summary['cumulative_stage_seconds'] == {
        'cache_check_seconds': 0.1,
        'conversion_seconds': 2.0,
        'vad_seconds': 3.0,
    }


def test_write_metrics_report_replaces_existing_report(tmp_path: Path) -> None:
    report_path = tmp_path / 'session-metrics.json'
    report_path.write_text('{"old": true}', encoding='utf-8')

    write_metrics_report(report_path, {'schema_version': 1, 'summary': {'files': 2}})

    assert json.loads(report_path.read_text(encoding='utf-8')) == {
        'schema_version': 1,
        'summary': {'files': 2},
    }
    assert not report_path.with_suffix('.json.tmp').exists()
