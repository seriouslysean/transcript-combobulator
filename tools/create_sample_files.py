#!/usr/bin/env python3
"""Create sample files for transcription testing."""

import shutil
from pathlib import Path

def create_sample_files(input_path: Path, output_dir: Path, num_copies: int = 5) -> None:
    """Create multiple copies of an input file with numbered names.

    Args:
        input_path: Path to the input audio file
        output_dir: Directory to save the copies
        num_copies: Number of copies to create
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_copies):
        dest_file = output_dir / f"{input_path.stem}_{i+1}{input_path.suffix}"
        shutil.copy2(input_path, dest_file)
        print(f"Created {dest_file}")

def main():
    """Create sample files from the JFK audio."""
    root_dir = Path(__file__).parent.parent
    source_file = root_dir / 'deps' / 'whisper.cpp' / 'samples' / 'jfk.wav'
    input_dir = root_dir / 'tmp' / 'input'

    if not source_file.exists():
        print(f"Error: Source file {source_file} not found")
        return

    create_sample_files(source_file, input_dir)

if __name__ == '__main__':
    main()
