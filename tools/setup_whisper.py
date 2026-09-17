#!/usr/bin/env python3
"""Download the configured whisper model into models/ without loading it.

whisper.load_model would also instantiate the fp32 model (about 3x the file
size in RAM); on a Pi that is most of the memory for a step that only needs
the bytes on disk. This uses whisper's own downloader, which also verifies
the checksum and skips the download when the file already matches.
"""

import sys
from pathlib import Path

import whisper

from src.config import WHISPER_MODEL, WHISPER_MODELS_DIR


def setup_whisper(model_name: str, models_dir: Path) -> bool:
    """Ensure models_dir/<model_name>.pt exists and matches whisper's checksum."""
    url = whisper._MODELS.get(model_name)
    if url is None:
        print(f"Unknown whisper model {model_name!r}. Available: {', '.join(whisper.available_models())}")
        return False
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}.pt"
        if model_path.exists():
            print(f"Verifying {model_path}...")
        else:
            print(f"Downloading {model_name} to {models_dir}...")
        whisper._download(url, str(models_dir), in_memory=False)
        print(f"Model ready: {model_path} ({model_path.stat().st_size / 2**20:.0f} MiB)")
        return True
    except Exception as e:
        print(f"Error setting up whisper: {e}")
        return False


def main() -> None:
    if not WHISPER_MODEL:
        print("Error: WHISPER_MODEL is not set")
        sys.exit(1)
    if not setup_whisper(WHISPER_MODEL, WHISPER_MODELS_DIR):
        sys.exit(1)


if __name__ == '__main__':
    main()
