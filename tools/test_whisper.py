#!/usr/bin/env python3
"""Quick smoke test: run whisper on a single file and print the transcript."""

import argparse
from pathlib import Path

import torch

from src.whisper import load_whisper_model


def test_whisper(audio_file: str | Path) -> None:
    print(f"\nTesting whisper transcription on {audio_file}...")
    print(f"Device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    try:
        model = load_whisper_model()
        result = model.transcribe(str(audio_file))
        print("\nTranscription result:")
        print("-" * 80)
        print(result["text"])
        print("-" * 80)
    except Exception as e:
        print(f"Error during transcription: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test whisper transcription")
    parser.add_argument("audio_file", help="Path to audio file to test")
    args = parser.parse_args()
    test_whisper(args.audio_file)
