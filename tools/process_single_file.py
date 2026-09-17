"""Process a single audio file: convert -> VAD -> transcribe."""

import argparse
import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, MutableMapping, Optional

from src.audio_utils import (
    AudioValidationError,
    convert_to_wav,
    needs_conversion,
    validate_audio_file,
)
from src.config import FAIL_ON_PARTIAL_TRANSCRIPTION, get_output_path_for_input
from src.logging_config import setup_logging
from src.pipeline_cache import (
    get_manifest_path,
    is_pipeline_complete,
    write_pipeline_manifest,
)
from src.transcribe import TranscriptionError, transcribe_segments
from src.telemetry import elapsed_seconds, utc_now_iso
from src.vad import process_audio

logger = logging.getLogger(__name__)


@contextmanager
def _timed_stage(metrics: dict[str, Any], stage: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        metrics.setdefault('stages', {})[f'{stage}_seconds'] = elapsed_seconds(
            started, time.perf_counter()
        )


def main(
    input_path: str,
    status_dict: Optional[MutableMapping[str, Any]] = None,
    status_key: Optional[str] = None,
    force: bool = False,
    metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    input_file = Path(input_path).resolve()
    setup_logging()
    file_metrics = metrics if metrics is not None else {}
    file_metrics.clear()
    started = time.perf_counter()
    file_metrics.update(
        {
            'file': input_file.name,
            'input_path': str(input_file),
            'input_bytes': input_file.stat().st_size,
            'forced': force,
            'cache_hit': False,
            'status': 'running',
            'started_at': utc_now_iso(),
            'stages': {},
        }
    )

    def _update_status(status: str) -> None:
        if status_dict is not None and status_key is not None:
            status_dict[status_key] = status

    def _progress_callback(phase: str, current: int, total: int) -> None:
        if phase == "loading":
            _update_status("loading model")
        else:
            _update_status(f"transcribing {current}/{total}")

    try:
        with _timed_stage(file_metrics, 'input_probe'):
            try:
                audio_info = validate_audio_file(input_file)
            except AudioValidationError:
                file_metrics['input_audio_seconds'] = None
            else:
                file_metrics['input_audio_seconds'] = round(
                    float(audio_info['duration']), 6
                )

        with _timed_stage(file_metrics, 'setup'):
            output_dir = get_output_path_for_input(input_file)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{input_file.stem}.wav"
            manifest_path = get_manifest_path(output_dir, input_file.stem)
            file_metrics['output_dir'] = str(output_dir)

        with _timed_stage(file_metrics, 'cache_check'):
            cache_hit = not force and is_pipeline_complete(input_file, manifest_path)
        if cache_hit:
            file_metrics['cache_hit'] = True
            file_metrics['status'] = 'cached'
            _update_status("cached")
            logger.info(f"Using completed pipeline output for {input_file.name}")
            return file_metrics

        with _timed_stage(file_metrics, 'cache_invalidation'):
            # A completion record must describe only the run that produced the
            # current artifacts. Leave the pipeline uncached if interrupted.
            manifest_path.unlink(missing_ok=True)

        _update_status("converting")
        logger.info(f"Step 1: Converting {input_file.name} if needed...")
        with _timed_stage(file_metrics, 'conversion'):
            # A cache miss must not reuse audio derived from an older source.
            if output_file.resolve() != input_file:
                output_file.unlink(missing_ok=True)
            conversion_required = needs_conversion(input_file)
            file_metrics['conversion'] = {
                'required': conversion_required,
                'action': 'converted' if conversion_required else 'copied',
            }
            if conversion_required:
                convert_to_wav(input_file, output_file)
            else:
                if not output_file.exists():
                    shutil.copy(input_file, output_file)
                logger.info("No conversion needed, copied to output directory")

        _update_status("splitting")
        logger.info(f"Step 2: Processing VAD on {output_file.name}...")
        with _timed_stage(file_metrics, 'vad'):
            _, vad_segments = process_audio(output_file)
        file_metrics['vad'] = {
            'chunk_count': len(vad_segments),
            'speech_seconds': round(
                sum(
                    float(segment['end_seconds'])
                    - float(segment['start_seconds'])
                    for segment in vad_segments
                ),
                6,
            ),
        }

        _update_status("loading model")
        logger.info("Step 3: Transcribing segments...")
        transcription_metrics: dict[str, Any] = {}
        file_metrics['transcription'] = transcription_metrics
        with _timed_stage(file_metrics, 'transcription'):
            transcription = transcribe_segments(
                output_file,
                input_file,
                progress_callback=_progress_callback,
                metrics=transcription_metrics,
            )
        file_metrics['transcription'] = transcription.get(
            'metrics', transcription_metrics
        )

        # Per-chunk failures are logged and skipped inside whisper so one bad
        # segment does not lose the file, but a file with gaps must not be
        # recorded as complete: the cache would then block the retry.
        failed_chunks = int(file_metrics['transcription'].get('failed_chunk_count', 0) or 0)
        if failed_chunks and FAIL_ON_PARTIAL_TRANSCRIPTION:
            total_chunks = file_metrics['transcription'].get('chunk_count', '?')
            raise TranscriptionError(
                f"{failed_chunks} of {total_chunks} segments failed to transcribe for "
                f"{input_file.name}; rerun to retry "
                "(FAIL_ON_PARTIAL_TRANSCRIPTION=false accepts partial output)"
            )

        artifacts = [
            output_file,
            Path(transcription['vtt_file']),
            Path(transcription['json_file']),
            Path(transcription['mapping_file']),
            *(Path(segment['segment_file']) for segment in vad_segments),
        ]
        with _timed_stage(file_metrics, 'manifest_write'):
            write_pipeline_manifest(input_file, manifest_path, artifacts)
        file_metrics['artifact_count'] = len(artifacts)
        file_metrics['status'] = 'processed'

        _update_status("done")
        logger.info("Step 4: All processing complete for this file")
        return file_metrics
    except Exception as e:
        file_metrics['status'] = 'error'
        file_metrics['error_type'] = type(e).__name__
        file_metrics['error'] = str(e)
        raise
    finally:
        file_metrics['finished_at'] = utc_now_iso()
        file_metrics['total_seconds'] = elapsed_seconds(
            started, time.perf_counter()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument("--force", action="store_true", help="Ignore completed output")
    args = parser.parse_args()
    main(args.input_file, force=args.force)
