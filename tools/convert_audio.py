#!/usr/bin/env python3
"""Convert audio files to 16kHz mono WAV.

Without --input: walks tmp/input/ and converts everything under it, mirroring
the directory structure into tmp/output/.

With --input <path>: converts just that file. The output path mirrors its
position relative to tmp/input/; if outside tmp/input/, the output filename
is derived from the input name.
"""

import argparse
import sys
from pathlib import Path

from src.audio_utils import (
    SUPPORTED_FORMATS,
    convert_to_wav,
    get_audio_info_summary,
    needs_conversion,
)
from src.config import INPUT_DIR, OUTPUT_DIR


def _output_path_for(input_path: Path) -> Path:
    """Mirror input_path under OUTPUT_DIR, appending _16khz before .wav."""
    input_path = input_path.resolve()
    try:
        rel_parent = input_path.relative_to(INPUT_DIR).parent
    except ValueError:
        rel_parent = Path()
    return OUTPUT_DIR / rel_parent / f"{input_path.stem}_16khz.wav"


def convert_one(input_file: Path) -> None:
    print(f"Input: {get_audio_info_summary(input_file)}")
    output_file = _output_path_for(input_file)
    if needs_conversion(input_file):
        convert_to_wav(input_file, output_file)
    else:
        print("No conversion needed")
    print(f"Output: {output_file}")


def convert_all() -> None:
    audio_files = [
        f for f in INPUT_DIR.rglob('*')
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]
    print(f"Found {len(audio_files)} audio files")
    for f in audio_files:
        print(f"Processing: {get_audio_info_summary(f)}")
        if needs_conversion(f):
            convert_to_wav(f, _output_path_for(f))
        else:
            print(f"Skipping {f.name} (already 16kHz WAV)")
    print("Conversion complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert audio to 16kHz mono WAV")
    parser.add_argument('--input', type=str, help='Specific file to convert')
    args = parser.parse_args()

    if args.input:
        convert_one(Path(args.input))
    else:
        if not INPUT_DIR.exists():
            print(f"Input directory not found: {INPUT_DIR}")
            sys.exit(1)
        convert_all()


if __name__ == '__main__':
    main()
