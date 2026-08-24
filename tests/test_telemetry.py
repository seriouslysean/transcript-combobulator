"""Tests for structured pipeline telemetry."""

import json
from pathlib import Path

from src.telemetry import (
    build_run_summary,
    sanitize_metrics_report,
    write_metrics_report,
)


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

    write_metrics_report(
        report_path, {'schema_version': 1, 'summary': {'files_total': 2}}
    )

    assert json.loads(report_path.read_text(encoding='utf-8')) == {
        'schema_version': 1,
        'summary': {'files_total': 2},
    }
    assert not report_path.with_suffix('.json.tmp').exists()


def test_persisted_report_excludes_identifiers_content_and_raw_errors(
    tmp_path: Path,
) -> None:
    sentinel = 'PRIVATE_SENTINEL'
    report_path = tmp_path / 'session-metrics.json'
    report = {
        'schema_version': 1,
        'run': {
            'session': sentinel,
            'status': 'failed',
            'target_dir': f'/Users/private/{sentinel}',
            'environment_file': f'.env.{sentinel}',
            'force': True,
        },
        'configuration': {
            'sample_rate': 16000,
            'whisper': {
                'model': f'/Users/private/{sentinel}.pt',
                'device': sentinel,
                'language': sentinel,
                'prompt': sentinel,
                'prompt_configured': True,
            },
            'vad': {'threshold': 0.5},
        },
        'summary': {
            'files_total': 1,
            'cumulative_stage_seconds': {
                'conversion_seconds': 2.0,
                sentinel: 1.0,
            },
        },
        'files': [
            {
                'file': f'1-{sentinel}.flac',
                'input_path': f'/Users/private/{sentinel}.flac',
                'output_dir': f'/Users/private/output/{sentinel}',
                'status': 'error',
                'error_type': 'WhisperError',
                'error': f'raw failure at /Users/private/{sentinel}',
                'stages': {'vad_seconds': 2.0, sentinel: 1.0},
                'conversion': {
                    'required': True,
                    'action': 'converted',
                    'detail': sentinel,
                },
                'vad': {
                    'chunk_count': 1,
                    'speech_seconds': 3.0,
                    'source': sentinel,
                },
                'transcription': {
                    'failed_chunk_count': 1,
                    'chunks': [
                        {
                            'index': 1,
                            'file': f'{sentinel}_segment.wav',
                            'status': 'error',
                            'error_type': 'WhisperError',
                            'error': sentinel,
                            'timings': {
                                'total_seconds': 1.0,
                                'error': sentinel,
                            },
                        }
                    ],
                },
            }
        ],
        'combine': {
            'status': 'error',
            'error_type': 'CombineError',
            'error': sentinel,
            'output_files': [f'/Users/private/{sentinel}.txt'],
        },
    }

    write_metrics_report(report_path, report)

    persisted_text = report_path.read_text(encoding='utf-8')
    persisted = json.loads(persisted_text)
    assert sentinel not in persisted_text
    assert '/Users/private' not in persisted_text
    assert persisted['files'][0]['file_id'] == 'file-001'
    assert persisted['files'][0]['error_type'] == 'WhisperError'
    assert persisted['files'][0]['stages'] == {'vad_seconds': 2.0}
    assert persisted['files'][0]['conversion'] == {
        'required': True,
        'action': 'converted',
    }
    assert persisted['files'][0]['vad'] == {
        'chunk_count': 1,
        'speech_seconds': 3.0,
    }
    assert 'file' not in persisted['files'][0]['transcription']['chunks'][0]
    assert persisted['configuration']['whisper'] == {
        'model': 'custom',
        'device': 'custom',
        'prompt_configured': True,
    }
    assert persisted['summary']['cumulative_stage_seconds'] == {
        'conversion_seconds': 2.0
    }
    assert persisted['combine'] == {
        'status': 'error',
        'error_type': 'CombineError',
        'output_file_count': 1,
    }


def test_sanitized_worker_error_retains_type_without_message() -> None:
    report = sanitize_metrics_report(
        {
            'files': [
                {
                    'status': 'error',
                    'error_type': 'TranscriptionError',
                    'error': 'private worker failure',
                }
            ]
        }
    )

    assert report['files'] == [
        {
            'file_id': 'file-001',
            'status': 'error',
            'error_type': 'TranscriptionError',
        }
    ]


def test_sanitized_success_report_retains_performance_fields() -> None:
    report = sanitize_metrics_report(
        {
            'run': {'status': 'completed', 'force': True},
            'summary': {
                'files_total': 1,
                'processing_seconds': 10.0,
                'transcript_segments': 5,
            },
            'files': [
                {
                    'file': 'private-speaker.flac',
                    'status': 'processed',
                    'total_seconds': 9.5,
                    'vad': {'chunk_count': 2, 'speech_seconds': 30.0},
                    'transcription': {
                        'written_vtt_cue_count': 5,
                        'chunks': [],
                    },
                }
            ],
            'combine': {
                'status': 'completed',
                'seconds': 0.1,
                'output_files': ['/private/transcript.txt'],
            },
        }
    )

    assert report['run'] == {'status': 'completed', 'force': True}
    assert report['summary']['processing_seconds'] == 10.0
    assert report['files'][0] == {
        'file_id': 'file-001',
        'status': 'processed',
        'total_seconds': 9.5,
        'vad': {'chunk_count': 2, 'speech_seconds': 30.0},
        'transcription': {
            'written_vtt_cue_count': 5,
            'chunks': [],
        },
    }
    assert report['combine']['output_file_count'] == 1


def test_sanitized_combine_error_retains_type_without_message() -> None:
    report = sanitize_metrics_report(
        {
            'combine': {
                'status': 'error',
                'seconds': 0.5,
                'error_type': 'CombineError',
                'error': 'private combine failure',
                'output_files': [],
            }
        }
    )

    assert report['combine'] == {
        'status': 'error',
        'seconds': 0.5,
        'error_type': 'CombineError',
        'output_file_count': 0,
    }
