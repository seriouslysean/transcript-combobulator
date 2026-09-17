"""Central configuration. Loads a single .env file and exposes typed settings.

Respects ENV_FILE to override the default .env. This module is imported early
by every other module — do not add imports from src.* here.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Anchor every path to the repository, not the caller's cwd, so the tools behave
# the same from cron, systemd, or another directory. PROJECT_ROOT (shell env,
# not .env) overrides for unusual layouts.
ROOT_DIR = Path(
    os.environ.get('PROJECT_ROOT') or Path(__file__).resolve().parents[1]
).resolve()


def _resolve_env_file(value: str) -> Path:
    """Relative ENV_FILE resolves against cwd if present there, else ROOT_DIR."""
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return ROOT_DIR / candidate


_env_file = _resolve_env_file(os.environ.get('ENV_FILE') or '.env')
load_dotenv(dotenv_path=_env_file, override=True)


def get_bool_env(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')


def get_float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def get_int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} must be set in .env file")
    return value


# ── Paths ──
TMP_DIR = ROOT_DIR / 'tmp'
INPUT_DIR = TMP_DIR / 'input'
OUTPUT_DIR = TMP_DIR / 'output'
TRANSCRIPTIONS_DIR = TMP_DIR / 'transcriptions'
WHISPER_MODELS_DIR = ROOT_DIR / 'models'


def get_output_path_for_input(input_path: Path) -> Path:
    """Mirror input dir structure under OUTPUT_DIR, scoped to a per-file subdir."""
    try:
        rel_path = input_path.relative_to(INPUT_DIR)
        return OUTPUT_DIR / rel_path.parent / input_path.stem
    except ValueError:
        return OUTPUT_DIR / input_path.stem


# ── Parallel Processing ──
PARALLEL_JOBS = get_int_env('PARALLEL_JOBS', 2)
TORCH_THREADS = get_int_env('TORCH_THREADS', 0)  # 0 = auto-detect per worker
WORKER_NICE = get_int_env('WORKER_NICE', 10)  # niceness increment; 0 = unchanged

# ── Batch Run Guards ──
# MEMORY_GUARD caps active workers so the estimated per-worker footprint
# (roughly 3x the whisper checkpoint size plus runtime) fits within
# MEMORY_GUARD_FRACTION of physical RAM. It only ever lowers PARALLEL_JOBS.
MEMORY_GUARD = get_bool_env('MEMORY_GUARD', True)
MEMORY_GUARD_FRACTION = get_float_env('MEMORY_GUARD_FRACTION', 0.85)
# MAPPING_PRECHECK validates TRANSCRIPT_N_* against the input files before any
# transcription starts, instead of failing at the combine step hours later.
MAPPING_PRECHECK = get_bool_env('MAPPING_PRECHECK', True)
# LOG_FILE: where batch runs persist worker logs. Empty = <output>/<session>/
# <session>.log; 'none' disables file logging (the pre-guard behaviour).
LOG_FILE = os.getenv('LOG_FILE', '').strip().strip('"')

# ── Audio Processing ──
WHISPER_SAMPLE_RATE = 16000


def _validate_sample_rate(sample_rate: int) -> int:
    """Whisper interprets ndarray audio at a fixed 16 kHz sample rate."""
    if sample_rate != WHISPER_SAMPLE_RATE:
        raise ValueError(
            f"SAMPLE_RATE must be {WHISPER_SAMPLE_RATE} for Whisper; got {sample_rate}"
        )
    return sample_rate


SAMPLE_RATE = _validate_sample_rate(get_int_env('SAMPLE_RATE', WHISPER_SAMPLE_RATE))
TRANSCRIPTION_MODE = os.getenv('TRANSCRIPTION_MODE', 'vad')

# ── VAD ──
VAD_THRESHOLD = get_float_env('VAD_THRESHOLD', 0.5)
# Preserve isolated short replies such as "yes", "no", and spoken numbers.
VAD_MIN_SPEECH_DURATION = get_float_env('VAD_MIN_SPEECH_DURATION', 0.25)
# Whisper encodes a full 30-second window for every VAD segment. A longer
# silence threshold avoids turning short pauses into separate encoder passes.
VAD_MIN_SILENCE_DURATION = get_float_env('VAD_MIN_SILENCE_DURATION', 3.0)
PADDING_SECONDS = get_float_env('PADDING_SECONDS', 0.3)

# ── Whisper (optimized for single-speaker channels) ──
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3-turbo')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
WHISPER_FP16 = get_bool_env('WHISPER_FP16', False)
WHISPER_LANGUAGE = os.getenv('WHISPER_LANGUAGE', 'en')
WHISPER_TEMPERATURE = get_float_env('WHISPER_TEMPERATURE', 0.0)
WHISPER_BEAM_SIZE = get_int_env('WHISPER_BEAM_SIZE', 1)
WHISPER_WORD_TIMESTAMPS = get_bool_env('WHISPER_WORD_TIMESTAMPS', False)
WHISPER_CONDITION_ON_PREVIOUS = get_bool_env('WHISPER_CONDITION_ON_PREVIOUS', False)
WHISPER_CARRY_INITIAL_PROMPT = get_bool_env('WHISPER_CARRY_INITIAL_PROMPT', True)
WHISPER_NO_SPEECH_THRESHOLD = get_float_env('WHISPER_NO_SPEECH_THRESHOLD', 0.6)
WHISPER_LOGPROB_THRESHOLD = get_float_env('WHISPER_LOGPROB_THRESHOLD', -1.0)
WHISPER_COMPRESSION_RATIO_THRESHOLD = get_float_env(
    'WHISPER_COMPRESSION_RATIO_THRESHOLD', 2.4
)
WHISPER_PROMPT = os.getenv('WHISPER_PROMPT', '')
WHISPER_CONFIDENCE_THRESHOLD = get_float_env('WHISPER_CONFIDENCE_THRESHOLD', 50.0)

# ── Output ──
SAVE_JSON = get_bool_env('SAVE_JSON', True)


def get_whisper_options() -> dict[str, Any]:
    """Kwargs for whisper.Whisper.transcribe()."""
    return {
        'language': WHISPER_LANGUAGE,
        'temperature': WHISPER_TEMPERATURE,
        'beam_size': WHISPER_BEAM_SIZE if WHISPER_BEAM_SIZE > 1 else None,
        'condition_on_previous_text': WHISPER_CONDITION_ON_PREVIOUS,
        'carry_initial_prompt': WHISPER_CARRY_INITIAL_PROMPT,
        'no_speech_threshold': WHISPER_NO_SPEECH_THRESHOLD,
        'logprob_threshold': WHISPER_LOGPROB_THRESHOLD,
        'compression_ratio_threshold': WHISPER_COMPRESSION_RATIO_THRESHOLD,
        'word_timestamps': WHISPER_WORD_TIMESTAMPS,
        'initial_prompt': WHISPER_PROMPT or None,
        'fp16': WHISPER_FP16,
    }


def get_pipeline_fingerprint_settings() -> dict[str, Any]:
    """Return output-affecting settings used by the per-file resume cache."""
    return {
        'sample_rate': SAMPLE_RATE,
        'transcription_mode': TRANSCRIPTION_MODE,
        'vad': {
            'threshold': VAD_THRESHOLD,
            'min_speech_duration': VAD_MIN_SPEECH_DURATION,
            'min_silence_duration': VAD_MIN_SILENCE_DURATION,
            'padding_seconds': PADDING_SECONDS,
        },
        'whisper_model': WHISPER_MODEL,
        'whisper_device': WHISPER_DEVICE,
        'whisper_options': get_whisper_options(),
    }
