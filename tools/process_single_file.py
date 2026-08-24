"""Process a single audio file: convert -> VAD -> transcribe."""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any, MutableMapping, Optional

from src.audio_utils import convert_to_wav, needs_conversion
from src.config import get_output_path_for_input
from src.logging_config import setup_logging
from src.pipeline_cache import (
    get_manifest_path,
    is_pipeline_complete,
    write_pipeline_manifest,
)
from src.transcribe import transcribe_segments
from src.vad import process_audio

logger = logging.getLogger(__name__)


def main(
    input_path: str,
    status_dict: Optional[MutableMapping[str, Any]] = None,
    status_key: Optional[str] = None,
    force: bool = False,
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
    manifest_path = get_manifest_path(output_dir, input_file.stem)

    if not force and is_pipeline_complete(input_file, manifest_path):
        _update_status("cached")
        logger.info(f"Using completed pipeline output for {input_file.name}")
        return

    _update_status("converting")
    logger.info(f"Step 1: Converting {input_file.name} if needed...")
    # Reaching this point means the completion cache missed or was bypassed.
    # Do not let convert_to_wav's derived-file reuse hide a changed source.
    if output_file.resolve() != input_file:
        output_file.unlink(missing_ok=True)
    if needs_conversion(input_file):
        convert_to_wav(input_file, output_file)
    else:
        if not output_file.exists():
            shutil.copy(input_file, output_file)
        logger.info("No conversion needed, copied to output directory")

    _update_status("splitting")
    logger.info(f"Step 2: Processing VAD on {output_file.name}...")
    _, vad_segments = process_audio(output_file)

    _update_status("loading model")
    logger.info("Step 3: Transcribing segments...")
    transcription = transcribe_segments(
        output_file,
        input_file,
        progress_callback=_progress_callback,
    )

    artifacts = [
        output_file,
        Path(transcription['vtt_file']),
        Path(transcription['json_file']),
        Path(transcription['mapping_file']),
        *(Path(segment['segment_file']) for segment in vad_segments),
    ]
    write_pipeline_manifest(input_file, manifest_path, artifacts)

    _update_status("done")
    logger.info("Step 4: All processing complete for this file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument("--force", action="store_true", help="Ignore completed output")
    args = parser.parse_args()
    main(args.input_file, force=args.force)
