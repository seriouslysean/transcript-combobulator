#!/usr/bin/env python3
"""Test whisper transcription."""

import os
from pathlib import Path
import whisper
import torch
from dotenv import load_dotenv

from src.whisper import load_whisper_model

def test_whisper(audio_file: str | Path) -> None:
    """Test whisper transcription on a single file.

    Args:
        audio_file: Path to audio file to test
    """
    load_dotenv()

    print(f"\nTesting whisper transcription on {audio_file}...")
    print(f"Device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    try:
        # Load model
        model = load_whisper_model()

        # Transcribe
        result = model.transcribe(str(audio_file))
        print("\nTranscription result:")
        print("-" * 80)
        print(result["text"])
        print("-" * 80)

    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test whisper transcription")
    parser.add_argument("audio_file", help="Path to audio file to test")
    args = parser.parse_args()

    test_whisper(args.audio_file)
