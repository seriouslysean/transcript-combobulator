"""Conversion to 16 kHz mono WAV."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from src.audio_utils import (
    AudioValidationError,
    convert_to_wav,
    needs_conversion,
    validate_audio_file,
)


def _stereo_44k(path: Path, seconds: float = 2.0, peak: float = 0.25) -> None:
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    left = peak * np.sin(2 * np.pi * 440 * t)
    right = (peak / 2) * np.sin(2 * np.pi * 660 * t)
    sf.write(str(path), np.stack([left, right], axis=1).astype(np.float32), sr)


def test_ffmpeg_conversion_yields_16k_mono_peak_normalized(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _stereo_44k(src)
    out = tmp_path / "out.wav"
    assert needs_conversion(src)

    with patch("src.audio_utils.AUDIO_CONVERTER", "ffmpeg"):
        convert_to_wav(src, out)

    info = validate_audio_file(out)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert abs(info["duration"] - 2.0) < 0.002
    data, _ = sf.read(str(out), dtype="float32")
    assert 0.98 <= float(np.abs(data).max()) <= 1.0
    assert not needs_conversion(out)
    assert not list(tmp_path.glob("*.unnormalized.wav"))


def test_torchaudio_path_matches_ffmpeg_within_tolerance(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _stereo_44k(src)
    via_ffmpeg = tmp_path / "ffmpeg.wav"
    via_torch = tmp_path / "torch.wav"
    with patch("src.audio_utils.AUDIO_CONVERTER", "ffmpeg"):
        convert_to_wav(src, via_ffmpeg)
    with patch("src.audio_utils.AUDIO_CONVERTER", "torchaudio"):
        convert_to_wav(src, via_torch)
    a, _ = sf.read(str(via_ffmpeg), dtype="float32")
    b, _ = sf.read(str(via_torch), dtype="float32")
    n = min(len(a), len(b))
    assert abs(len(a) - len(b)) <= 32
    # Different resamplers; the waveforms should agree closely after the ramp-in.
    assert np.abs(a[1000:n - 1000] - b[1000:n - 1000]).mean() < 0.02


def test_missing_ffmpeg_is_a_clear_error(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _stereo_44k(src)
    with patch("src.audio_utils.AUDIO_CONVERTER", "ffmpeg"), \
         patch("src.audio_utils.shutil.which", return_value=None), \
         pytest.raises(AudioValidationError, match="ffmpeg not found"):
        convert_to_wav(src, tmp_path / "out.wav")


def test_existing_correct_output_is_reused(tmp_path: Path) -> None:
    out = tmp_path / "already.wav"
    sf.write(str(out), np.zeros(16000, dtype=np.float32), 16000)
    with patch("src.audio_utils.subprocess.run") as run:
        convert_to_wav(tmp_path / "missing-input.wav", out)
    run.assert_not_called()
