"""Process a single audio file: convert -> VAD -> transcribe."""

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, MutableMapping, Optional

from src.audio_utils import convert_to_wav, needs_conversion
from src.config import get_output_path_for_input
from src.logging_config import setup_logging
from src.transcribe import transcribe_segments
from src.vad import process_audio

logger = logging.getLogger(__name__)


def main(
    input_path: str,
    status_dict: Optional[MutableMapping[str, Any]] = None,
    status_key: Optional[str] = None,
) -> None:
    input_file = Path(input_path).resolve()
    setup_logging()

    def _update_status(status: str) -> None:
        if status_dict is not None and status_key is not None:
            status_dict[status_key] = status

    def _progress_callback(phase: str, current: int, total: int) -> None:
        if phase == "loading":
            _update_status("loading model")
        else:
            _update_status(f"transcribing {current}/{total}")

    output_dir = get_output_path_for_input(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}.wav"

    _update_status("converting")
    logger.info(f"Step 1: Converting {input_file.name} if needed...")
    if needs_conversion(input_file):
        convert_to_wav(input_file, output_file)
    else:
        if not output_file.exists():
            shutil.copy(input_file, output_file)
        logger.info("No conversion needed, copied to output directory")

    _update_status("splitting")
    logger.info(f"Step 2: Processing VAD on {output_file.name}...")
    process_audio(output_file)

    _update_status("loading model")
    logger.info("Step 3: Transcribing segments...")
    transcribe_segments(output_file, input_file, progress_callback=_progress_callback)

    _update_status("done")
    logger.info("Step 4: All processing complete for this file")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: process_single_file.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
