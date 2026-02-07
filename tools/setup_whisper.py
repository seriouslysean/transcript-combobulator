#!/usr/bin/env python3
"""Setup whisper and download model."""

import os
from pathlib import Path
import torch
import whisper
from dotenv import load_dotenv

from src.config import TMP_DIR, WHISPER_MODEL

def get_whisper_config() -> dict:
    """Get whisper configuration from environment variables.

    Returns:
        dict: Configuration for whisper with device and compute type
    """
    load_dotenv()

    device = os.getenv('WHISPER_DEVICE', 'cpu')
    fp16 = os.getenv('WHISPER_FP16', 'false').lower() in ('true', '1', 'yes', 'on')

    # Validate FP16 based on device
    if device == 'cpu' and fp16:
        print("Warning: FP16 not supported on CPU, forcing FP32")
        fp16 = False

    return {
        'device': device,
        'fp16': fp16
    }

def setup_whisper(model_name: str, models_dir: Path) -> bool:
    """Set up whisper and download model.

    Args:
        model_name: Name of the whisper model to download
        models_dir: Directory to store models

    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Create models directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)

        # Set WHISPER_MODELS_DIR environment variable
        os.environ['WHISPER_MODELS_DIR'] = str(models_dir)

        # Get whisper configuration
        config = get_whisper_config()

        # Check if model already exists
        model_path = models_dir / f"{model_name}.pt"
        if model_path.exists():
            print(f"Model {model_name} already exists at {model_path}")
            return True

        print(f"Downloading {model_name} model to {models_dir}...")

        # This will download the model to the specified cache location
        model = whisper.load_model(
            model_name,
            device=config['device'],
            download_root=str(models_dir)
        )

        print(f"Successfully downloaded {model_name} model")
        return True

    except Exception as e:
        print(f"Error setting up whisper: {e}")
        return False

def main():
    """Download and setup whisper model."""
    # Load environment variables
    load_dotenv()

    # Get model from environment
    model = os.getenv('WHISPER_MODEL')
    if not model:
        print("Error: WHISPER_MODEL environment variable is required")
        exit(1)

    # Setup whisper with model in models directory
    models_dir = Path('models')
    print(f"Checking for {model} model in {models_dir}...")
    if setup_whisper(model, models_dir):
        print("Setup completed successfully!")
    else:
        print("Setup failed!")
        exit(1)

if __name__ == '__main__':
    main()
