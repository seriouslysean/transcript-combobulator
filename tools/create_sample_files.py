#!/usr/bin/env python3
"""Create sample files for development and testing."""

import shutil
import soundfile as sf
import numpy as np
import torch
import torchaudio
from pathlib import Path
import argparse

def create_copies(input_path: Path, output_dir: Path, num_copies: int = 5, prefix: str = "") -> None:
    """Create multiple copies of an input file with numbered names.

    Args:
        input_path: Path to the input audio file
        output_dir: Directory to save the copies
        num_copies: Number of copies to create
        prefix: Optional prefix for the output files (e.g. 'test_' for test files)
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if num_copies == 1:
        # For single copy, don't add a number
        dest_file = output_dir / f"{prefix}{input_path.stem}{input_path.suffix}"
        shutil.copy2(input_path, dest_file)
        print(f"Created {dest_file}")
    else:
        # For multiple copies, add numbers
        for i in range(num_copies):
            dest_file = output_dir / f"{prefix}{input_path.stem}_{i+1}{input_path.suffix}"
            shutil.copy2(input_path, dest_file)
            print(f"Created {dest_file}")

def create_padded_version(
    input_path: Path,
    output_dir: Path,
    prefix: str = "",
    silence_duration: float = 4.0,  # 4 seconds of silence between segments
    num_copies: int = 3  # Always create 3 copies
) -> None:
    """Create a padded version of the audio file with multiple copies and silence between.

    Args:
        input_path: Path to the input audio file
        output_dir: Directory to save the padded version
        prefix: Optional prefix for the output file (e.g. 'test_' for test files)
        silence_duration: Duration of silence in seconds between segments
        num_copies: Number of copies of the speech to include (default 3)
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}{input_path.stem}_padded{input_path.suffix}"

    # Load audio using torchaudio for consistent handling
    wav, sr = torchaudio.load(input_path)
    silence_samples = int(silence_duration * sr)
    silence = torch.zeros((wav.shape[0], silence_samples))

    # Create pattern: speech -> silence -> speech -> silence -> speech
    padded_wav = wav
    for _ in range(num_copies - 1):
        padded_wav = torch.cat([padded_wav, silence, wav], dim=1)

    sf.write(output_path, padded_wav.T.numpy(), sr)
    print(f"Created padded version: {output_path}")

def main():
    """Create sample files from the JFK audio."""
    parser = argparse.ArgumentParser(description="Create sample files")
    parser.add_argument('--prefix', default="", help='Prefix for output files (e.g. "test_")')
    parser.add_argument('--copies', type=int, default=5, help='Number of individual files to create')
    parser.add_argument('--padded-copies', type=int, default=3, help='Number of speech segments to include in the padded version (will be appended with silence between)')
    parser.add_argument('--session', default="jfk-sample", help='Session directory name (default: jfk-sample)')
    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent
    samples_dir = root_dir / 'samples'
    
    # Check if samples directory exists
    if not samples_dir.exists():
        print(f"Error: Samples directory not found at {samples_dir}")
        print("Please create the samples/ directory and add your audio files.")
        print("Example: samples/jfk.wav")
        return
    
    # Check for WAV files
    wav_files = list(samples_dir.glob('*.wav'))
    if not wav_files:
        print(f"Error: No WAV files found in {samples_dir}")
        print("Please add sample audio files (*.wav) to the samples/ directory.")
        return
    
    # Create files in session directory structure
    if args.prefix.startswith("test_"):
        input_dir = root_dir / 'tmp' / 'input'  # Test files go in root input
    else:
        input_dir = root_dir / 'tmp' / 'input' / args.session  # Sample files go in session folder
    
    print(f"Creating sample files in {input_dir}")
    
    # Process each WAV file in samples directory
    for sample_file in wav_files:
        # Create individual numbered files
        create_copies(
            input_path=sample_file,
            output_dir=input_dir,
            num_copies=args.copies,
            prefix=args.prefix
        )
        # Create one file with appended segments and 4s silence between them
        create_padded_version(
            input_path=sample_file,
            output_dir=input_dir,
            prefix=args.prefix,
            num_copies=args.padded_copies
        )

if __name__ == '__main__':
    main()
