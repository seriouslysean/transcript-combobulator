"""Configuration settings for the project.

This module loads and validates environment variables for audio processing
and Whisper transcription settings. It provides type-safe access to these
settings with sensible defaults.
"""

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def validate_audio_settings() -> None:
    """Validate audio processing settings are within acceptable ranges."""
    if SAMPLE_RATE <= 0:
        raise ValueError("SAMPLE_RATE must be positive")
    if not 0 <= VAD_THRESHOLD <= 1:
        raise ValueError("VAD_THRESHOLD must be between 0 and 1")
    if VAD_MIN_SPEECH_DURATION <= 0:
        raise ValueError("VAD_MIN_SPEECH_DURATION must be positive")
    if VAD_MIN_SILENCE_DURATION <= 0:
        raise ValueError("VAD_MIN_SILENCE_DURATION must be positive")
    if PADDING_SECONDS < 0:
        raise ValueError("PADDING_SECONDS must be non-negative")

def validate_whisper_settings() -> None:
    """Validate Whisper transcription settings are within acceptable ranges."""
    if WHISPER_BEAM_SIZE <= 0:
        raise ValueError("WHISPER_BEAM_SIZE must be positive")
    if WHISPER_ENTROPY_THOLD <= 0:
        raise ValueError("WHISPER_ENTROPY_THOLD must be positive")
    if WHISPER_MAX_CONTEXT <= 0:
        raise ValueError("WHISPER_MAX_CONTEXT must be positive")
    if not 0 <= WHISPER_TEMPERATURE <= 1:
        raise ValueError("WHISPER_TEMPERATURE must be between 0 and 1")

# Audio processing settings
SAMPLE_RATE: Final[int] = int(os.getenv('SAMPLE_RATE', '16000'))
VAD_THRESHOLD: Final[float] = float(os.getenv('VAD_THRESHOLD', '0.2'))
VAD_MIN_SPEECH_DURATION: Final[float] = float(os.getenv('VAD_MIN_SPEECH_DURATION', '0.3')) * 1000
VAD_MIN_SILENCE_DURATION: Final[float] = float(os.getenv('VAD_MIN_SILENCE_DURATION', '0.5')) * 1000
PADDING_SECONDS: Final[float] = float(os.getenv('PADDING_SECONDS', '0.3'))

# Directory paths
ROOT_DIR: Final[Path] = Path(__file__).parent.parent
INPUT_DIR: Final[Path] = ROOT_DIR / 'tmp' / 'input'
OUTPUT_DIR: Final[Path] = ROOT_DIR / 'tmp' / 'output'
TRANSCRIPTIONS_DIR: Final[Path] = ROOT_DIR / 'tmp' / 'transcriptions'

# Whisper.cpp paths
WHISPER_CPP_PATH: Final[Path] = ROOT_DIR / os.getenv('WHISPER_CPP_PATH', 'deps/whisper.cpp/build/bin')
WHISPER_MODEL_PATH: Final[Path] = ROOT_DIR / os.getenv('WHISPER_MODEL_PATH', 'deps/whisper.cpp/models/ggml-large-v3.bin')

# Whisper.cpp settings
WHISPER_PROMPT: Final[str] = os.getenv('WHISPER_PROMPT', '')
WHISPER_BEAM_SIZE: Final[int] = int(os.getenv('WHISPER_BEAM_SIZE', '8'))
WHISPER_ENTROPY_THOLD: Final[float] = float(os.getenv('WHISPER_ENTROPY_THOLD', '2.4'))
WHISPER_MAX_CONTEXT: Final[int] = int(os.getenv('WHISPER_MAX_CONTEXT', '128'))
WHISPER_TEMPERATURE: Final[float] = float(os.getenv('WHISPER_TEMPERATURE', '0.0'))
WHISPER_WORD_THOLD: Final[float] = float(os.getenv('WHISPER_WORD_THOLD', '0.6'))
WHISPER_LANGUAGE: Final[str] = os.getenv('WHISPER_LANGUAGE', 'en')

# Validate settings
validate_audio_settings()
validate_whisper_settings()

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, TRANSCRIPTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
