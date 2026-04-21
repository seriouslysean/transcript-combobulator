#!/usr/bin/env python3
"""Create sample/test audio files from inputs in samples/ for development and testing."""

import argparse
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def create_copies(
    input_path: Path,
    output_dir: Path,
    num_copies: int = 5,
    prefix: str = "",
) -> None:
    """Duplicate input_path into output_dir, numbered ``{prefix}{stem}_N{suffix}``.

    ``num_copies=1`` produces a single unnumbered copy (``{prefix}{stem}{suffix}``).
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = input_path.stem, input_path.suffix

    if num_copies == 1:
        dest = output_dir / f"{prefix}{stem}{suffix}"
        shutil.copy2(input_path, dest)
        print(f"Created {dest}")
        return

    for i in range(num_copies):
        dest = output_dir / f"{prefix}{stem}_{i + 1}{suffix}"
        shutil.copy2(input_path, dest)
        print(f"Created {dest}")


def create_padded_audio(
    input_path: Path,
    output_path: Path,
    num_copies: int = 3,
    silence_duration: float = 4.0,
) -> None:
    """Write a file that repeats input_path ``num_copies`` times with silence between."""
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    wav, sr = torchaudio.load(str(input_path))
    silence = torch.zeros((wav.shape[0], int(silence_duration * sr)))

    padded_wav = wav
    for _ in range(num_copies - 1):
        padded_wav = torch.cat([padded_wav, silence, wav], dim=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), padded_wav.T.numpy(), sr)
    print(f"Created padded version: {output_path}")


def create_sample_files(
    prefix: str = "",
    copies: int = 5,
    padded_copies: int = 3,
    session: str = "jfk-sample",
) -> None:
    """Create numbered copies + a padded file for every WAV in samples/.

    Test files (prefix starting with ``test_``) go to ``tmp/input/``.
    Sample files go to ``tmp/input/{session}/``.
    """
    root_dir = Path(__file__).parent.parent
    samples_dir = root_dir / 'samples'

    if not samples_dir.exists():
        print(f"Error: Samples directory not found at {samples_dir}")
        print("Please create the samples/ directory and add your audio files.")
        return

    wav_files = list(samples_dir.glob('*.wav'))
    if not wav_files:
        print(f"Error: No WAV files found in {samples_dir}")
        return

    input_dir = (
        root_dir / 'tmp' / 'input'
        if prefix.startswith("test_")
        else root_dir / 'tmp' / 'input' / session
    )
    print(f"Creating sample files in {input_dir}")

    for sample_file in wav_files:
        create_copies(sample_file, input_dir, num_copies=copies, prefix=prefix)
        create_padded_audio(
            input_path=sample_file,
            output_path=input_dir / f"{prefix}{sample_file.stem}_padded{sample_file.suffix}",
            num_copies=padded_copies,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create sample files")
    parser.add_argument('--prefix', default="", help='Prefix for output files (e.g. "test_")')
    parser.add_argument('--copies', type=int, default=5)
    parser.add_argument('--padded-copies', type=int, default=3)
    parser.add_argument('--session', default="jfk-sample")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    create_sample_files(
        prefix=args.prefix,
        copies=args.copies,
        padded_copies=args.padded_copies,
        session=args.session,
    )


if __name__ == '__main__':
    main()
