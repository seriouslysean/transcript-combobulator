"""Per-file pipeline: convert -> VAD -> transcribe, with stage-level resume."""

import json
import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, MutableMapping, Optional

from transcript_combobulator.audio_utils import (
    AudioValidationError,
    convert_to_wav,
    needs_conversion,
    validate_audio_file,
)
from transcript_combobulator.config import (
    FAIL_ON_PARTIAL_TRANSCRIPTION,
    get_output_path_for_input,
    vtt_name_for_stem,
)
from transcript_combobulator.logging_config import setup_logging
from transcript_combobulator.pipeline_cache import (
    build_stage_fingerprints,
    get_manifest_path,
    get_progress_path,
    invalidate_stage,
    is_pipeline_complete,
    load_manifest,
    record_stage,
    stage_is_complete,
    write_pipeline_manifest,
)
from transcript_combobulator.transcribe import TranscriptionError, transcribe_segments
from transcript_combobulator.telemetry import elapsed_seconds, utc_now_iso
from transcript_combobulator.vad import process_audio
from transcript_combobulator.whisper import regenerate_vtt_with_confidence

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


def process_file(
    input_path: str | Path,
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

        fingerprints = build_stage_fingerprints(input_file)
        progress_path = get_progress_path(output_dir, input_file.stem)
        mapping_path = output_dir / f"{input_file.stem}_mapping.json"
        json_path = output_dir / f"{input_file.stem}_transcription.json"
        vtt_path = output_dir / vtt_name_for_stem(input_file.stem)
        with _timed_stage(file_metrics, 'cache_invalidation'):
            if force:
                # A forced run trusts nothing it finds.
                manifest_path.unlink(missing_ok=True)
                progress_path.unlink(missing_ok=True)
            else:
                # The completion record must describe only the run that
                # produced the current artifacts. Earlier stage records stay
                # and are validated one by one before anything is reused.
                invalidate_stage(manifest_path, 'vtt')
        stages = load_manifest(manifest_path)

        _update_status("converting")
        logger.info(f"Step 1: Converting {input_file.name} if needed...")
        with _timed_stage(file_metrics, 'conversion'):
            if stage_is_complete(stages, 'conversion', fingerprints['conversion']):
                file_metrics['conversion'] = {'required': False, 'action': 'cached'}
                logger.info("Conversion already complete for this source; reusing")
            else:
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
                record_stage(manifest_path, 'conversion', fingerprints['conversion'], [output_file])
                stages = load_manifest(manifest_path)

        _update_status("splitting")
        logger.info(f"Step 2: Processing VAD on {output_file.name}...")
        with _timed_stage(file_metrics, 'vad'):
            if stage_is_complete(stages, 'vad', fingerprints['vad']):
                with open(mapping_path, encoding='utf-8') as f:
                    vad_segments = json.load(f)['segments']
                file_metrics['vad_cached'] = True
                logger.info(f"VAD already complete; reusing {len(vad_segments)} segments")
            else:
                _, vad_segments = process_audio(output_file)
                record_stage(
                    manifest_path,
                    'vad',
                    fingerprints['vad'],
                    [mapping_path, *(Path(s['segment_file']) for s in vad_segments)],
                )
                stages = load_manifest(manifest_path)
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
        transcription: dict[str, Any]
        with _timed_stage(file_metrics, 'transcription'):
            if stage_is_complete(stages, 'inference', fingerprints['inference']):
                # Only presentation settings changed (or the VTT went missing):
                # rewrite the VTT from the saved JSON without any inference.
                logger.info("Transcription already complete; rewriting VTT from saved JSON")
                with _timed_stage(file_metrics, 'vtt_rewrite'):
                    regenerate_vtt_with_confidence(json_path, vtt_path, None)
                transcription = {
                    'vtt_file': str(vtt_path),
                    'json_file': str(json_path),
                    'mapping_file': str(mapping_path),
                    'metrics': {
                        'chunk_count': len(vad_segments),
                        'failed_chunk_count': 0,
                        'resumed_chunk_count': len(vad_segments),
                        'chunks': [],
                        'inference_cached': True,
                    },
                }
            else:
                transcription = transcribe_segments(
                    output_file,
                    input_file,
                    progress_callback=_progress_callback,
                    metrics=transcription_metrics,
                    checkpoint_path=progress_path,
                    checkpoint_key=fingerprints['inference'],
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

        if not stage_is_complete(stages, 'inference', fingerprints['inference']):
            record_stage(
                manifest_path,
                'inference',
                fingerprints['inference'],
                [Path(transcription['json_file']), Path(transcription['vtt_file'])],
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
