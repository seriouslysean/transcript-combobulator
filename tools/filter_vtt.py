"""Rewrite a speaker's VTT from the pipeline's saved JSON, filtered by confidence.

No whisper inference: reads <stem>_transcription.json written by the pipeline
and re-emits the VTT with the same dedup rules, keeping only segments at or
above the confidence threshold.
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    WHISPER_CONFIDENCE_THRESHOLD,
    get_output_path_for_input,
    vtt_path_for_input,
)
from src.whisper import WhisperError, regenerate_vtt_with_confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Original input audio (locates the JSON and VTT)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=WHISPER_CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence 0-100 (default {WHISPER_CONFIDENCE_THRESHOLD})",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file).resolve()
    json_path = get_output_path_for_input(input_file) / f"{input_file.stem}_transcription.json"
    vtt_path = vtt_path_for_input(input_file)
    try:
        kept = regenerate_vtt_with_confidence(json_path, vtt_path, args.threshold)
    except WhisperError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Wrote {len(kept)} cues at confidence >= {args.threshold} to {vtt_path}")


if __name__ == "__main__":
    main()
