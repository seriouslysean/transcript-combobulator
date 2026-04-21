"""Audio format validation and conversion to 16kHz mono WAV."""

from pathlib import Path
from typing import Any

import soundfile as sf
import torchaudio

from src.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac', '.opus'}


class AudioValidationError(Exception):
    """Exception raised for audio validation errors."""


def validate_audio_file(file_path: Path) -> dict[str, Any]:
    """Return {sample_rate, channels, duration, format, frames} for an audio file."""
    if not file_path.exists():
        raise AudioValidationError(f"Audio file not found: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported audio format: {file_path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    try:
        with sf.SoundFile(str(file_path)) as f:
            return {
                'sample_rate': f.samplerate,
                'channels': f.channels,
                'duration': f.frames / f.samplerate,
                'format': file_path.suffix.lower(),
                'frames': f.frames,
            }
    except Exception as e:
        raise AudioValidationError(f"Failed to read audio file {file_path}: {e}") from e


def needs_conversion(file_path: Path, target_sample_rate: int = 16000) -> bool:
    """True if the file isn't already 16kHz mono WAV."""
    try:
        info = validate_audio_file(file_path)
    except AudioValidationError:
        return True
    return bool(
        info['format'] != '.wav'
        or info['sample_rate'] != target_sample_rate
        or info['channels'] != 1
    )


def convert_to_wav(
    input_path: Path, output_path: Path, target_sample_rate: int = 16000
) -> None:
    """Convert any supported format to 16kHz mono WAV, normalized to [-1, 1].

    Skips the work if output already exists with the right format.
    """
    if output_path.exists():
        try:
            info = validate_audio_file(output_path)
            if (
                info['sample_rate'] == target_sample_rate
                and info['format'] == '.wav'
                and info['channels'] == 1
            ):
                logger.info(f"Using existing converted file: {output_path}")
                return
        except AudioValidationError:
            logger.info(f"Existing file {output_path} is invalid, reconverting...")

    try:
        input_info = validate_audio_file(input_path)
        logger.info(
            f"Converting {input_path} (SR: {input_info['sample_rate']}, "
            f"Channels: {input_info['channels']}) to WAV"
        )

        wav, sr = torchaudio.load(str(input_path))

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            logger.info("Converted to mono")

        if sr != target_sample_rate:
            wav = torchaudio.transforms.Resample(sr, target_sample_rate)(wav)
            logger.info(f"Resampled from {sr}Hz to {target_sample_rate}Hz")

        wav = wav / (wav.abs().max() + 1e-8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav, target_sample_rate)

        info = validate_audio_file(output_path)
        logger.info(
            f"Conversion successful: {output_path} "
            f"(SR: {info['sample_rate']}, Duration: {info['duration']:.2f}s)"
        )
    except Exception as e:
        raise AudioValidationError(f"Failed to convert {input_path} to WAV: {e}") from e


def get_audio_info_summary(file_path: Path) -> str:
    """Human-readable one-liner with format, sample rate, channels, duration."""
    try:
        info = validate_audio_file(file_path)
        return (
            f"{file_path.name}: {info['format'].upper()}, "
            f"{info['sample_rate']}Hz, {info['channels']} channel(s), "
            f"{info['duration']:.2f}s"
        )
    except AudioValidationError as e:
        return f"{file_path.name}: ERROR - {e}"
