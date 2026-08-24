"""Fast unit tests for the Whisper wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.whisper import WhisperError, load_whisper_model, transcribe_segment


def test_load_whisper_model_is_cached_per_process(tmp_path: Path) -> None:
    model_name = 'test-model'
    (tmp_path / f'{model_name}.pt').touch()
    expected_model = object()
    load_whisper_model.cache_clear()

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
        load_whisper_model.cache_clear()


def test_transcribe_segment_passes_wav_array_to_model(tmp_path: Path) -> None:
    audio_path = tmp_path / 'segment.wav'
    audio_path.touch()
    waveform = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    model = MagicMock()
    model.transcribe.return_value = {
        'segments': [{'start': 0.0, 'end': 1.0, 'text': ' hello'}]
    }

    with patch('src.whisper.sf.read', return_value=(waveform, 16000)) as read:
        segments = transcribe_segment(audio_path, offset=5.0, model=model)

    read.assert_called_once_with(str(audio_path), dtype='float32', always_2d=False)
    transcribe_input = model.transcribe.call_args.args[0]
    assert isinstance(transcribe_input, np.ndarray)
    np.testing.assert_array_equal(transcribe_input, waveform)
    assert segments[0]['start'] == 5.0
    assert segments[0]['end'] == 6.0


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
