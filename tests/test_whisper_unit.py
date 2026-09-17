"""Fast unit tests for the Whisper wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import get_whisper_options
from src.whisper import (
    WhisperError,
    _load_whisper_model,
    dedupe_segments,
    regenerate_vtt_with_confidence,
    load_whisper_model,
    transcribe_audio_segments,
    transcribe_segment,
)


def test_whisper_options_carry_configured_prompt_across_windows() -> None:
    with patch('src.config.WHISPER_CARRY_INITIAL_PROMPT', True):
        assert get_whisper_options()['carry_initial_prompt'] is True


def test_load_whisper_model_is_cached_per_process(tmp_path: Path) -> None:
    model_name = 'test-model'
    (tmp_path / f'{model_name}.pt').touch()
    expected_model = object()
    _load_whisper_model.cache_clear()

    try:
        with patch('src.whisper.WHISPER_MODELS_DIR', tmp_path), patch(
            'src.whisper.get_whisper_device', return_value='cpu'
        ), patch('src.whisper.whisper.load_model', return_value=expected_model) as load:
            first = load_whisper_model(model_name)
            second = load_whisper_model(model_name)

        assert first is expected_model
        assert second is expected_model
        load.assert_called_once()
    finally:
        _load_whisper_model.cache_clear()


def test_transcribe_segment_passes_wav_array_to_model(tmp_path: Path) -> None:
    audio_path = tmp_path / 'segment.wav'
    audio_path.touch()
    waveform = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    model = MagicMock()
    timing = {}
    model.transcribe.return_value = {
        'segments': [{'start': 0.0, 'end': 1.0, 'text': ' hello'}]
    }

    with patch('src.whisper.sf.read', return_value=(waveform, 16000)) as read:
        segments = transcribe_segment(
            audio_path, offset=5.0, model=model, timing=timing
        )

    read.assert_called_once_with(str(audio_path), dtype='float32', always_2d=False)
    transcribe_input = model.transcribe.call_args.args[0]
    assert isinstance(transcribe_input, np.ndarray)
    np.testing.assert_array_equal(transcribe_input, waveform)
    assert segments[0]['start'] == 5.0
    assert segments[0]['end'] == 6.0
    assert timing['audio_seconds'] == round(3 / 16000, 6)
    assert timing['status'] == 'processed'
    assert timing['inference_seconds'] >= 0
    assert timing['total_seconds'] >= timing['inference_seconds']


def test_transcribe_segment_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    audio_path = tmp_path / 'segment.wav'
    audio_path.touch()
    model = MagicMock()

    with patch(
        'src.whisper.sf.read',
        return_value=(np.zeros(10, dtype=np.float32), 44100),
    ), pytest.raises(WhisperError, match='Expected 16000Hz'):
        transcribe_segment(audio_path, model=model)

    model.transcribe.assert_not_called()


def test_empty_transcription_replaces_stale_vtt(tmp_path: Path) -> None:
    segment_path = tmp_path / 'segment.wav'
    segment_path.touch()
    output_path = tmp_path / 'speaker.vtt'
    output_path.write_text('WEBVTT\n\nOLD TEXT\n', encoding='utf-8')

    metrics = {}
    with patch('src.whisper.load_whisper_model', return_value=object()), patch(
        'src.whisper.transcribe_segment', return_value=[]
    ):
        segments = transcribe_audio_segments(
            [(segment_path, 0.0)], output_path, metrics=metrics
        )

    assert segments == []
    assert output_path.read_text(encoding='utf-8') == 'WEBVTT\n\n'
    assert metrics['chunk_count'] == 1
    assert metrics['chunks'][0]['status'] == 'empty'
    assert metrics['raw_result_segment_count'] == 0
    assert metrics['written_vtt_cue_count'] == 0
    assert metrics['total_seconds'] >= 0


def test_metrics_distinguish_raw_results_from_written_vtt_cues(
    tmp_path: Path,
) -> None:
    first = tmp_path / 'first.wav'
    second = tmp_path / 'second.wav'
    third = tmp_path / 'third.wav'
    output_path = tmp_path / 'speaker.vtt'
    for path in (first, second, third):
        path.touch()

    metrics = {}
    with patch('src.whisper.load_whisper_model', return_value=object()), patch(
        'src.whisper.transcribe_segment',
        side_effect=[
            [{'start': 0.0, 'end': 1.0, 'text': 'Repeated line'}],
            [{'start': 1.0, 'end': 2.0, 'text': ' Repeated line '}],
            [{'start': 2.0, 'end': 3.0, 'text': '   '}],
        ],
    ):
        segments = transcribe_audio_segments(
            [(first, 0.0), (second, 1.0), (third, 2.0)],
            output_path,
            metrics=metrics,
        )

    assert len(segments) == 3
    assert metrics['raw_result_segment_count'] == 3
    assert metrics['written_vtt_cue_count'] == 1
    assert output_path.read_text(encoding='utf-8').count('-->') == 1


def test_all_segment_failures_fail_the_transcription(tmp_path: Path) -> None:
    first = tmp_path / 'first.wav'
    second = tmp_path / 'second.wav'
    first.touch()
    second.touch()

    metrics = {}
    with patch('src.whisper.load_whisper_model', return_value=object()), patch(
        'src.whisper.transcribe_segment', side_effect=WhisperError('decode failed')
    ), pytest.raises(WhisperError, match='All 2 audio segments failed'):
        transcribe_audio_segments(
            [(first, 0.0), (second, 1.0)],
            tmp_path / 'out.vtt',
            metrics=metrics,
        )

    assert metrics['failed_chunk_count'] == 2
    assert [chunk['status'] for chunk in metrics['chunks']] == ['error', 'error']
    assert metrics['total_seconds'] >= 0



def _seg(start: float, end: float, text: str) -> dict:
    return {'start': start, 'end': end, 'text': text}


def test_dedupe_consecutive_drops_adjacent_repeat_only() -> None:
    segments = [
        _seg(0.0, 1.0, 'Yeah.'),
        _seg(1.2, 2.0, 'Yeah.'),      # hallucinated repeat, 0.2 s later
        _seg(600.0, 601.0, 'Yeah.'),  # genuine, ten minutes later
        _seg(601.5, 602.0, '   '),    # blank
    ]
    kept = dedupe_segments(segments, strategy='consecutive', window_seconds=2.0)
    assert [s['start'] for s in kept] == [0.0, 600.0]


def test_dedupe_global_matches_legacy_behaviour() -> None:
    segments = [_seg(0.0, 1.0, 'Yeah.'), _seg(600.0, 601.0, 'Yeah.')]
    assert len(dedupe_segments(segments, strategy='global')) == 1
    assert len(dedupe_segments(segments, strategy='none')) == 2


def test_dedupe_sorts_by_start_before_comparing() -> None:
    segments = [_seg(5.0, 6.0, 'B'), _seg(0.0, 1.0, 'A'), _seg(1.1, 2.0, 'A')]
    kept = dedupe_segments(segments, strategy='consecutive', window_seconds=2.0)
    assert [s['text'] for s in kept] == ['A', 'B']


def test_regenerate_vtt_reads_pipeline_json_and_filters(tmp_path: Path) -> None:
    import json

    json_path = tmp_path / "3-nilbits_transcription.json"
    json_path.write_text(json.dumps({
        "audio_path": "x.wav",
        "mapping_file": "m.json",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Keep me", "confidence": 90.0},
            {"start": 2.0, "end": 3.0, "text": "Drop me", "confidence": 20.0},
            {"start": 3.1, "end": 4.0, "text": "Keep me", "confidence": 95.0},
            {"start": 60.0, "end": 61.0, "text": "Keep me", "confidence": 95.0},
        ],
    }), encoding="utf-8")
    vtt = tmp_path / "3-nilbits.vtt"

    kept = regenerate_vtt_with_confidence(json_path, vtt, 50.0)

    text = vtt.read_text(encoding="utf-8")
    assert "Drop me" not in text
    # 0.0 and 3.1 "Keep me" are not consecutive within the 2 s window once
    # "Drop me" is filtered? They are: 3.1 - 1.0 = 2.1 > 2.0, so both stay.
    assert text.count("Keep me") == 3
    assert [s["start"] for s in kept] == [0.0, 3.1, 60.0]



def test_interrupted_transcription_resumes_from_checkpoint(tmp_path: Path) -> None:
    from src.pipeline_cache import load_chunk_checkpoint

    clips = [tmp_path / f'c{i}.wav' for i in range(3)]
    for c in clips:
        c.touch()
    jobs = [(clips[0], 0.0), (clips[1], 10.0), (clips[2], 20.0)]
    progress = tmp_path / 'progress.jsonl'
    vtt = tmp_path / 'out.vtt'
    seg = lambda t, text: [{'start': t, 'end': t + 1.0, 'text': text, 'confidence': 90.0}]

    # First run: chunk 2 fails; chunks 1 and 3 are checkpointed.
    metrics = {}
    with patch('src.whisper.load_whisper_model', return_value=object()), patch(
        'src.whisper.transcribe_segment',
        side_effect=[seg(0.0, 'one'), WhisperError('boom'), seg(20.0, 'three')],
    ):
        transcribe_audio_segments(jobs, vtt, metrics=metrics, checkpoint_path=progress, checkpoint_key='fp')
    assert metrics['failed_chunk_count'] == 1
    assert progress.exists()
    assert set(load_chunk_checkpoint(progress, 'fp')) == {1, 3}

    # Second run: only chunk 2 is transcribed; output equals a clean run.
    metrics = {}
    with patch('src.whisper.load_whisper_model', return_value=object()), patch(
        'src.whisper.transcribe_segment', side_effect=[seg(10.0, 'two')]
    ) as ts:
        segments = transcribe_audio_segments(jobs, vtt, metrics=metrics, checkpoint_path=progress, checkpoint_key='fp')
    assert ts.call_count == 1
    assert [s['text'] for s in segments] == ['one', 'two', 'three']
    assert metrics['resumed_chunk_count'] == 2
    assert metrics['failed_chunk_count'] == 0
    assert [c['status'] for c in metrics['chunks']] == ['resumed', 'processed', 'resumed']
    assert not progress.exists()
    assert vtt.read_text(encoding='utf-8').count('-->') == 3


def test_checkpoint_with_changed_settings_is_discarded(tmp_path: Path) -> None:
    clip = tmp_path / 'c.wav'
    clip.touch()
    progress = tmp_path / 'progress.jsonl'
    progress.write_text('{"key": "old"}\n{"index": 1, "segments": [{"start": 0, "end": 1, "text": "stale"}], "timings": {}}\n')
    with patch('src.whisper.load_whisper_model', return_value=object()), patch(
        'src.whisper.transcribe_segment', return_value=[{'start': 0.0, 'end': 1.0, 'text': 'fresh', 'confidence': 90.0}]
    ) as ts:
        segments = transcribe_audio_segments([(clip, 0.0)], tmp_path / 'o.vtt', checkpoint_path=progress, checkpoint_key='new')
    assert ts.call_count == 1
    assert segments[0]['text'] == 'fresh'
