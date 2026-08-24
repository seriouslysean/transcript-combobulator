"""Fast unit tests for the Whisper wrapper."""

from pathlib import Path
from unittest.mock import patch

from src.whisper import load_whisper_model


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
