#!/usr/bin/env python3
"""Download a whisper model into the local models/ directory."""

import os
import sys
from pathlib import Path

import whisper

from src.config import WHISPER_DEVICE, WHISPER_FP16, WHISPER_MODEL


def setup_whisper(model_name: str, models_dir: Path) -> bool:
    """Download model_name to models_dir if not already present."""
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ['WHISPER_MODELS_DIR'] = str(models_dir)

        device = WHISPER_DEVICE
        if device == 'cpu' and WHISPER_FP16:
            print("Warning: FP16 not supported on CPU, forcing FP32")

        model_path = models_dir / f"{model_name}.pt"
        if model_path.exists():
            print(f"Model {model_name} already exists at {model_path}")
            return True

        print(f"Downloading {model_name} model to {models_dir}...")
        whisper.load_model(model_name, device=device, download_root=str(models_dir))
        print(f"Successfully downloaded {model_name} model")
        return True
    except Exception as e:
        print(f"Error setting up whisper: {e}")
        return False


def main() -> None:
    model = WHISPER_MODEL
    if not model:
        print("Error: WHISPER_MODEL is not set")
        sys.exit(1)
    models_dir = Path('models')
    print(f"Checking for {model} model in {models_dir}...")
    if setup_whisper(model, models_dir):
        print("Setup completed successfully!")
    else:
        print("Setup failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
