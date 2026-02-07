"""Configuration module for the application."""

import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
# If ENV_FILE is set in the environment, use it as the env file, otherwise default to .env
_env_file = os.environ.get('ENV_FILE') or '.env'
load_dotenv(dotenv_path=_env_file, override=True)

def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_float_env(key: str, default: float) -> float:
    """Get float environment variable."""
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

def get_int_env(key: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

def require_env(key: str) -> str:
    """Get required environment variable."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} must be set in .env file")
    return value

# Base Directories
# Paths
ROOT_DIR = Path(os.getcwd()).resolve()
TMP_DIR = ROOT_DIR / 'tmp'
INPUT_DIR = TMP_DIR / 'input'
OUTPUT_DIR = TMP_DIR / 'output'
TRANSCRIPTIONS_DIR = TMP_DIR / 'transcriptions'
WHISPER_MODELS_DIR = ROOT_DIR / 'models'

def get_output_path_for_input(input_path: Path) -> Path:
    """Generate output path that preserves input directory structure.

    Args:
        input_path: Path to input audio file

    Returns:
        Output directory path that mirrors input structure
    """
    try:
        rel_path = input_path.relative_to(INPUT_DIR)
    # rel_path: 2025-07-27/5-username.flac -> parent: 2025-07-27, stem: 5-username
        output_dir = OUTPUT_DIR / rel_path.parent / input_path.stem
    except ValueError:
        # If input is not under INPUT_DIR, use just the stem
        output_dir = OUTPUT_DIR / input_path.stem
    return output_dir

# ── Parallel Processing Settings ──
PARALLEL_JOBS = get_int_env('PARALLEL_JOBS', 2)
TORCH_THREADS = get_int_env('TORCH_THREADS', 0)  # 0 = auto-detect

# ── Audio Processing Settings ──
SAMPLE_RATE = get_int_env('SAMPLE_RATE', 16000)
TRANSCRIPTION_MODE = os.getenv('TRANSCRIPTION_MODE', 'vad')

# ── VAD defaults ──
VAD_THRESHOLD = get_float_env('VAD_THRESHOLD', 0.5)
VAD_MIN_SPEECH_DURATION = get_float_env('VAD_MIN_SPEECH_DURATION', 0.5)
VAD_MIN_SILENCE_DURATION = get_float_env('VAD_MIN_SILENCE_DURATION', 1.0)
PADDING_SECONDS = get_float_env('PADDING_SECONDS', 0.3)

# ── Whisper defaults (optimized for single-speaker channels) ──
# These can be overridden via .env but have sensible project-wide defaults
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3-turbo')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
WHISPER_FP16 = get_bool_env('WHISPER_FP16', False)

# Whisper Transcription Settings
WHISPER_LANGUAGE = os.getenv('WHISPER_LANGUAGE', 'en')
WHISPER_TEMPERATURE = get_float_env('WHISPER_TEMPERATURE', 0.0)
WHISPER_BEAM_SIZE = get_int_env('WHISPER_BEAM_SIZE', 1)
WHISPER_WORD_TIMESTAMPS = get_bool_env('WHISPER_WORD_TIMESTAMPS', False)
WHISPER_CONDITION_ON_PREVIOUS = get_bool_env('WHISPER_CONDITION_ON_PREVIOUS', False)
WHISPER_NO_SPEECH_THRESHOLD = get_float_env('WHISPER_NO_SPEECH_THRESHOLD', 0.6)
WHISPER_LOGPROB_THRESHOLD = get_float_env('WHISPER_LOGPROB_THRESHOLD', -1.0)
WHISPER_COMPRESSION_RATIO_THRESHOLD = get_float_env('WHISPER_COMPRESSION_RATIO_THRESHOLD', 2.4)

# Context Settings
WHISPER_PROMPT = os.getenv('WHISPER_PROMPT', '')

# Confidence Settings
WHISPER_CONFIDENCE_THRESHOLD = get_float_env('WHISPER_CONFIDENCE_THRESHOLD', 50.0)

# Output Settings
SAVE_JSON = get_bool_env('SAVE_JSON', True)

def get_whisper_options() -> Dict[str, Any]:
    """Get all whisper options as a dictionary for model.transcribe()."""
    return {
        'language': WHISPER_LANGUAGE,
        'temperature': WHISPER_TEMPERATURE,
        'beam_size': WHISPER_BEAM_SIZE if WHISPER_BEAM_SIZE > 1 else None,
        'condition_on_previous_text': WHISPER_CONDITION_ON_PREVIOUS,
        'no_speech_threshold': WHISPER_NO_SPEECH_THRESHOLD,
        'logprob_threshold': WHISPER_LOGPROB_THRESHOLD,
        'compression_ratio_threshold': WHISPER_COMPRESSION_RATIO_THRESHOLD,
        'word_timestamps': WHISPER_WORD_TIMESTAMPS,
        'initial_prompt': WHISPER_PROMPT if WHISPER_PROMPT else None,
        'fp16': WHISPER_FP16,
    }
