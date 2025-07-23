"""Audio utility functions for format validation and conversion."""

import soundfile as sf
import torchaudio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.logging_config import get_logger
logger = get_logger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac', '.opus'}

class AudioValidationError(Exception):
    """Exception raised for audio validation errors."""
    pass

def validate_audio_file(file_path: Path) -> Dict[str, Any]:
    """Validate audio file format and properties.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dict containing audio properties: sample_rate, channels, duration, format
        
    Raises:
        AudioValidationError: If file is invalid or unsupported
    """
    if not file_path.exists():
        raise AudioValidationError(f"Audio file not found: {file_path}")
    
    # Check file extension
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported audio format: {file_path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        # Use soundfile to get metadata (supports more formats)
        with sf.SoundFile(str(file_path)) as f:
            sample_rate = f.samplerate
            channels = f.channels
            frames = f.frames
            duration = frames / sample_rate
            
        return {
            'sample_rate': sample_rate,
            'channels': channels,
            'duration': duration,
            'format': file_path.suffix.lower(),
            'frames': frames
        }
        
    except Exception as e:
        raise AudioValidationError(f"Failed to read audio file {file_path}: {e}")

def needs_conversion(file_path: Path, target_sample_rate: int = 16000) -> bool:
    """Check if audio file needs conversion to target format.
    
    Args:
        file_path: Path to the audio file
        target_sample_rate: Target sample rate (default: 16000)
        
    Returns:
        bool: True if conversion is needed
    """
    try:
        info = validate_audio_file(file_path)
        
        # Check if conversion is needed
        needs_format_conversion = info['format'] != '.wav'
        needs_resampling = info['sample_rate'] != target_sample_rate
        needs_mono_conversion = info['channels'] != 1
        
        return needs_format_conversion or needs_resampling or needs_mono_conversion
        
    except AudioValidationError:
        return True  # If we can't validate, assume conversion is needed

def convert_to_wav(input_path: Path, output_path: Path, target_sample_rate: int = 16000) -> None:
    """Convert audio file to WAV format with specified sample rate.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        target_sample_rate: Target sample rate (default: 16000)
        
    Raises:
        AudioValidationError: If conversion fails
    """
    # Check if output already exists and is valid
    if output_path.exists():
        try:
            output_info = validate_audio_file(output_path)
            if (output_info['sample_rate'] == target_sample_rate and 
                output_info['format'] == '.wav' and 
                output_info['channels'] == 1):
                logger.info(f"Using existing converted file: {output_path}")
                return
        except AudioValidationError:
            logger.info(f"Existing file {output_path} is invalid, reconverting...")
    
    try:
        # Validate input file
        input_info = validate_audio_file(input_path)
        logger.info(f"Converting {input_path} (SR: {input_info['sample_rate']}, "
                   f"Channels: {input_info['channels']}) to WAV")
        
        # Load audio using torchaudio for better format support
        wav, sr = torchaudio.load(str(input_path))
        
        # Convert to mono if needed
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            logger.info("Converted to mono")
        
        # Resample if needed
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            wav = resampler(wav)
            logger.info(f"Resampled from {sr}Hz to {target_sample_rate}Hz")
        
        # Normalize audio to [-1, 1] range
        wav = wav / (wav.abs().max() + 1e-8)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV
        torchaudio.save(str(output_path), wav, target_sample_rate)
        
        # Validate output
        output_info = validate_audio_file(output_path)
        logger.info(f"Conversion successful: {output_path} "
                   f"(SR: {output_info['sample_rate']}, Duration: {output_info['duration']:.2f}s)")
        
    except Exception as e:
        raise AudioValidationError(f"Failed to convert {input_path} to WAV: {e}")

def get_audio_info_summary(file_path: Path) -> str:
    """Get human-readable summary of audio file properties.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        str: Summary string
    """
    try:
        info = validate_audio_file(file_path)
        return (f"{file_path.name}: {info['format'].upper()}, "
                f"{info['sample_rate']}Hz, {info['channels']} channel(s), "
                f"{info['duration']:.2f}s")
    except AudioValidationError as e:
        return f"{file_path.name}: ERROR - {e}"