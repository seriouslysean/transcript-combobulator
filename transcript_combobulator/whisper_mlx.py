"""mlx-whisper runtime: Whisper on the Apple Silicon GPU.

Selected with ``WHISPER_RUNTIME=mlx``. Gives ``whisper.py`` the one method it
uses from stock openai-whisper, ``transcribe(audio, **get_whisper_options())``,
so VAD, checkpointing, dedup, the repetition filter, and the failure policy are
the same code for both runtimes. Nothing here is imported unless the mlx
runtime is selected.
"""

from pathlib import Path
from typing import Any

import numpy.typing as npt

from transcript_combobulator.config import MLX_WEIGHTS_FILENAME
from transcript_combobulator.logging_config import get_logger
from transcript_combobulator.whisper import WhisperError

logger = get_logger(__name__)


class MlxWhisperModel:
    """A loaded mlx-whisper checkpoint with openai-whisper's ``transcribe``."""

    def __init__(self, model_dir: Path) -> None:
        self._path = str(model_dir)
        try:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder
        except ImportError as e:
            raise WhisperError(
                "WHISPER_RUNTIME=mlx needs mlx-whisper: `make install EXTRAS=dev,mac` "
                "(Apple Silicon only)"
            ) from e
        # mlx-whisper loads lazily on the first transcribe() and keeps one model
        # per process in ModelHolder. Load now so the cost lands in the model
        # load metric, and a bad checkpoint fails before any audio is touched.
        try:
            ModelHolder.get_model(self._path, mx.float16)
        except Exception as e:
            raise WhisperError(
                f"Failed to load mlx-whisper model {self._path}: {e}"
            ) from e

    def transcribe(self, audio: npt.NDArray[Any], **options: Any) -> dict[str, Any]:
        import mlx_whisper

        result: dict[str, Any] = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self._path, verbose=None, **options
        )
        return result


def load_mlx_whisper_model(model_dir: Path) -> MlxWhisperModel:
    """Load the mlx checkpoint in ``model_dir`` (see ``make setup-whisper``)."""
    weights = model_dir / MLX_WEIGHTS_FILENAME
    if not weights.is_file() or not (model_dir / "config.json").is_file():
        raise WhisperError(
            f"mlx model not found in {model_dir}. Run `make setup-whisper` with "
            "WHISPER_RUNTIME=mlx first."
        )
    return MlxWhisperModel(model_dir)
