#!/usr/bin/env python3
"""Create test files for the test suite."""

import torch
import torchaudio
import soundfile as sf
from pathlib import Path

def create_padded_audio(
    input_path: Path,
    output_path: Path,
    num_copies: int = 3,
    silence_duration: float = 5.0
) -> None:
    """Create a test audio file with multiple copies and silence padding.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the padded audio
        num_copies: Number of copies to include
        silence_duration: Duration of silence between copies in seconds
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    # Load audio
    wav, sr = torchaudio.load(input_path)

    # Create silence
    silence_samples = int(silence_duration * sr)
    silence = torch.zeros((wav.shape[0], silence_samples))

    # Create padded audio
    padded_wav = wav
    for _ in range(num_copies - 1):
        padded_wav = torch.cat([padded_wav, silence, wav], dim=1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, padded_wav.T.numpy(), sr)
    print(f"Created padded audio: {output_path}")

def main():
    """Create test files from the JFK audio."""
    root_dir = Path(__file__).parent.parent
    source_file = root_dir / 'samples' / 'jfk.wav'
    input_dir = root_dir / 'tmp' / 'input'

    if not source_file.exists():
        print(f"Error: Source file {source_file} not found")
        return

    # Create simple test file
    test_file = input_dir / 'test_jfk.wav'
    import shutil
    shutil.copy2(source_file, test_file)
    print(f"Created {test_file}")

    # Create padded test file
    create_padded_audio(
        input_path=source_file,
        output_path=input_dir / 'test_jfk_padded.wav',
        num_copies=3,
        silence_duration=5.0
    )

if __name__ == '__main__':
    main()
