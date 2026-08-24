"""Parallel batch processor with live progress display."""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, MutableMapping, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    build_run_summary,
    elapsed_seconds,
    utc_now_iso,
    write_metrics_report,
)

# Ensure spawn method for macOS torch compatibility
multiprocessing.set_start_method("spawn", force=True)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


def _calculate_torch_threads(
    max_workers: int,
    configured_threads: int,
    cpu_count: int | None = None,
) -> int:
    """Resolve threads per worker, splitting detected CPUs across active workers."""
    if configured_threads > 0:
        return configured_threads
    detected_cpus = cpu_count if cpu_count is not None else os.cpu_count()
    return max(1, (detected_cpus or 4) // max(1, max_workers))


def find_audio_files(target_dir: Path) -> list[Path]:
    """Find audio files in target directory, excluding converted files."""
    files = []
    for f in sorted(target_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS and "_converted" not in f.stem:
            files.append(f)
    return files


def _worker(
    file_path: str,
    status_dict: "MutableMapping[str, str]",
    status_key: str,
    force: bool,
) -> tuple[str, str, str, dict[str, Any]]:
    """Worker function that runs in a subprocess.

    Returns:
        (filename, status, error_message_or_empty, per_file_metrics)
    """
    # Suppress all logging output — the rich table is the UI
    logging.disable(logging.CRITICAL)

    # Also suppress stdout/stderr from submodules
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    file_metrics: dict[str, Any] = {}
    try:
        from tools.process_single_file import main as process_main
        process_main(
            file_path,
            status_dict=status_dict,
            status_key=status_key,
            force=force,
            metrics=file_metrics,
        )
        result_status = 'cached' if file_metrics.get('cache_hit') else 'done'
        return (Path(file_path).name, result_status, "", file_metrics)
    except Exception as e:
        file_metrics.setdefault('file', Path(file_path).name)
        file_metrics.setdefault('status', 'error')
        file_metrics.setdefault('error_type', type(e).__name__)
        file_metrics.setdefault('error', str(e))
        return (Path(file_path).name, "error", str(e), file_metrics)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        logging.disable(logging.NOTSET)


def _initialize_worker(torch_threads: int, worker_nice: int) -> None:
    """Apply process-wide scheduling and Torch settings once per worker."""
    if worker_nice:
        try:
            os.nice(worker_nice)
        except OSError:
            pass

    if torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(1)
        except (RuntimeError, ValueError):
            pass


def _build_table(
    file_names: list[str],
    status_dict: "MutableMapping[str, str]",
    max_workers: int,
) -> Table:
    """Build a rich Table showing current progress."""
    table = Table(title=f"Transcription Progress ({max_workers} workers)")
    table.add_column("Speaker", style="cyan", min_width=24)
    table.add_column("Status", min_width=20)

    for name in file_names:
        raw_status = status_dict.get(name, "waiting")
        label, style = _status_display(raw_status)
        table.add_row(name, Text(label, style=style))

    return table


def _status_display(raw_status: str) -> tuple[str, str]:
    """Map a raw status string to a display label and style."""
    static = {
        "waiting": ("waiting", "dim"),
        "converting": ("converting", "yellow"),
        "splitting": ("splitting", "yellow"),
        "loading model": ("loading model", "blue"),
        "cached": ("\u2713 cached", "green"),
        "done": ("\u2713 done", "green"),
    }
    if raw_status in static:
        return static[raw_status]
    if raw_status.startswith("transcribing"):
        return (raw_status, "magenta")
    if raw_status.startswith("error"):
        return ("\u2717 error", "red bold")
    return (raw_status, "")


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _format_metric_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    return f"{value:.2f}s"


def _build_file_metrics_table(file_metrics: list[dict[str, Any]]) -> Table:
    """Build the post-run per-file timing summary."""
    table = Table(title="Per-file Pipeline Metrics")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Audio", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Convert", justify="right")
    table.add_column("VAD", justify="right")
    table.add_column("Model", justify="right")
    table.add_column("Inference", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("xRT", justify="right")

    for metrics in sorted(file_metrics, key=lambda item: str(item.get('file', ''))):
        stages = metrics.get('stages', {})
        transcription = metrics.get('transcription', {})
        inference_seconds = sum(
            float(chunk.get('timings', {}).get('inference_seconds', 0.0))
            for chunk in transcription.get('chunks', [])
        )
        audio_seconds = metrics.get('input_audio_seconds')
        total_seconds = metrics.get('total_seconds')
        x_realtime = (
            audio_seconds / total_seconds
            if isinstance(audio_seconds, (int, float))
            and isinstance(total_seconds, (int, float))
            and total_seconds
            else None
        )
        status = str(metrics.get('status', 'unknown'))
        status_style = {
            'processed': 'green',
            'cached': 'green',
            'error': 'red bold',
        }.get(status, '')
        table.add_row(
            str(metrics.get('file', '-')),
            Text(status, style=status_style),
            _format_duration(float(audio_seconds))
            if isinstance(audio_seconds, (int, float))
            else '-',
            _format_metric_seconds(total_seconds),
            _format_metric_seconds(stages.get('conversion_seconds')),
            _format_metric_seconds(stages.get('vad_seconds')),
            _format_metric_seconds(transcription.get('model_load_seconds')),
            _format_metric_seconds(inference_seconds),
            str(metrics.get('vad', {}).get('chunk_count', '-')),
            f"{x_realtime:.2f}x" if x_realtime is not None else '-',
        )
    return table


def _build_run_metrics_table(
    summary: dict[str, Any], metrics_path: Optional[Path]
) -> Table:
    """Build a compact aggregate summary for the terminal."""
    table = Table(title="Session Telemetry")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    file_status = (
        f"{summary['files_processed']} processed, "
        f"{summary['files_cached']} cached, {summary['files_failed']} failed"
    )
    table.add_row("Files", file_status)
    table.add_row("Input audio", _format_duration(summary['input_audio_seconds']))
    table.add_row("Processing wall time", _format_duration(summary['processing_seconds']))
    table.add_row(
        "Audio throughput",
        f"{summary['audio_x_realtime']:.2f}x realtime"
        if summary['audio_x_realtime'] is not None
        else '-',
    )
    table.add_row(
        "Real-time factor",
        f"{summary['real_time_factor']:.4f}"
        if summary['real_time_factor'] is not None
        else '-',
    )
    table.add_row("VAD chunks", str(summary['vad_chunks']))
    table.add_row("Transcript segments", str(summary['transcript_segments']))
    table.add_row(
        "Inference throughput",
        f"{summary['inference_x_realtime']:.2f}x realtime"
        if summary['inference_x_realtime'] is not None
        else '-',
    )
    table.add_row(
        "Metrics report",
        str(metrics_path) if metrics_path is not None else "not written",
    )
    return table


def _publish_metrics_report(
    metrics_path: Path, report: dict[str, Any]
) -> tuple[Optional[Path], Optional[str]]:
    """Write telemetry and return the path only when publication succeeds."""
    try:
        write_metrics_report(metrics_path, report)
    except OSError as e:
        return None, str(e)
    return metrics_path, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process audio files with live progress")
    parser.add_argument("target_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--session", type=str, default=None, help="Session name for combine step")
    parser.add_argument("--force", action="store_true", help="Ignore completed output")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.is_dir():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    files = find_audio_files(target_dir)
    if not files:
        print(f"No audio files found in {target_dir}")
        sys.exit(1)

    # Load config (imports dotenv and captures the run profile once).
    from src.config import (
        OUTPUT_DIR,
        PADDING_SECONDS,
        PARALLEL_JOBS,
        SAMPLE_RATE,
        TORCH_THREADS,
        VAD_MIN_SILENCE_DURATION,
        VAD_MIN_SPEECH_DURATION,
        VAD_THRESHOLD,
        WHISPER_BEAM_SIZE,
        WHISPER_CARRY_INITIAL_PROMPT,
        WHISPER_COMPRESSION_RATIO_THRESHOLD,
        WHISPER_CONDITION_ON_PREVIOUS,
        WHISPER_DEVICE,
        WHISPER_FP16,
        WHISPER_LANGUAGE,
        WHISPER_LOGPROB_THRESHOLD,
        WHISPER_MODEL,
        WHISPER_NO_SPEECH_THRESHOLD,
        WHISPER_PROMPT,
        WHISPER_TEMPERATURE,
        WHISPER_WORD_TIMESTAMPS,
        WORKER_NICE,
    )

    max_workers = min(max(1, PARALLEL_JOBS), len(files))
    torch_threads = _calculate_torch_threads(max_workers, TORCH_THREADS)

    file_names = [f.name for f in files]

    # Shared dict for cross-process status updates
    manager = multiprocessing.Manager()
    status_dict = manager.dict({name: "waiting" for name in file_names})

    print(f"Processing {len(files)} audio files in {target_dir}")
    started_at = utc_now_iso()
    run_started = time.perf_counter()
    processing_started = time.perf_counter()

    errors: list[tuple[str, str]] = []
    completed_file_metrics: list[dict[str, Any]] = []

    # Install a SIGINT handler that forcefully kills child processes.
    # Without this, ProcessPoolExecutor and Manager ignore the first Ctrl+C.
    executor_ref: ProcessPoolExecutor | None = None

    def _sigint_handler(signum, frame):
        print("\nCancelled.")
        if executor_ref is not None:
            for pid in executor_ref._processes:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        manager.shutdown()
        sys.exit(130)

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)

    try:
        with Live(_build_table(file_names, status_dict, max_workers), refresh_per_second=4) as live:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_initialize_worker,
                initargs=(torch_threads, WORKER_NICE),
            ) as executor:
                executor_ref = executor
                futures = {}
                for f in files:
                    fut = executor.submit(
                        _worker,
                        str(f),
                        status_dict,
                        f.name,
                        args.force,
                    )
                    futures[fut] = f.name

                # Poll for status updates instead of blocking on as_completed
                while True:
                    live.update(_build_table(file_names, status_dict, max_workers))

                    done_futures = [f for f in futures if f.done()]
                    for fut in done_futures:
                        if fut in futures:
                            name, status, err_msg, file_metrics = fut.result()
                            completed_file_metrics.append(file_metrics)
                            if status == "error":
                                status_dict[name] = "error"
                                errors.append((name, err_msg))
                            del futures[fut]

                    if not futures:
                        break

                    time.sleep(0.25)

                # Final refresh
                live.update(_build_table(file_names, status_dict, max_workers))
    finally:
        signal.signal(signal.SIGINT, old_handler)

    processing_seconds = elapsed_seconds(processing_started, time.perf_counter())
    print(f"Processed {len(files)} files in {_format_duration(processing_seconds)}")

    session_name = args.session or target_dir.name
    metrics_path = OUTPUT_DIR / session_name / f"{session_name}-metrics.json"
    combine_metrics: dict[str, Any] = {
        'status': 'skipped' if errors else 'running',
        'seconds': 0.0,
        'output_files': [],
    }
    combine_error: Optional[str] = None

    if not errors:
        print("Combining transcripts...")
        from src.combine import combine_transcripts_from_env

        combine_started = time.perf_counter()
        try:
            output_files = combine_transcripts_from_env(OUTPUT_DIR, session_name)
            combine_metrics['status'] = 'completed'
            combine_metrics['output_files'] = [str(path) for path in output_files]
            for path in output_files:
                print(f"Combined transcript: {path}")
        except Exception as e:
            combine_error = str(e)
            combine_metrics['status'] = 'error'
            combine_metrics['error_type'] = type(e).__name__
            combine_metrics['error'] = combine_error
        finally:
            combine_metrics['seconds'] = elapsed_seconds(
                combine_started, time.perf_counter()
            )

    wall_seconds = elapsed_seconds(run_started, time.perf_counter())
    completed_file_metrics.sort(key=lambda item: str(item.get('file', '')))
    summary = build_run_summary(
        completed_file_metrics,
        processing_seconds=processing_seconds,
        wall_seconds=wall_seconds,
        max_workers=max_workers,
    )
    report = {
        'schema_version': TELEMETRY_SCHEMA_VERSION,
        'run': {
            'session': session_name,
            'status': 'failed' if errors or combine_error else 'completed',
            'started_at': started_at,
            'finished_at': utc_now_iso(),
            'target_dir': str(target_dir),
            'environment_file': os.environ.get('ENV_FILE') or '.env',
            'force': args.force,
        },
        'runtime': {
            'parallel_jobs_configured': PARALLEL_JOBS,
            'active_workers': max_workers,
            'torch_threads_per_worker': torch_threads,
            'worker_nice': WORKER_NICE,
        },
        'configuration': {
            'sample_rate': SAMPLE_RATE,
            'whisper': {
                'model': WHISPER_MODEL,
                'device': WHISPER_DEVICE,
                'fp16': WHISPER_FP16,
                'language': WHISPER_LANGUAGE,
                'temperature': WHISPER_TEMPERATURE,
                'beam_size': WHISPER_BEAM_SIZE,
                'word_timestamps': WHISPER_WORD_TIMESTAMPS,
                'condition_on_previous_text': WHISPER_CONDITION_ON_PREVIOUS,
                'carry_initial_prompt': WHISPER_CARRY_INITIAL_PROMPT,
                'prompt_configured': bool(WHISPER_PROMPT),
                'no_speech_threshold': WHISPER_NO_SPEECH_THRESHOLD,
                'logprob_threshold': WHISPER_LOGPROB_THRESHOLD,
                'compression_ratio_threshold': (
                    WHISPER_COMPRESSION_RATIO_THRESHOLD
                ),
            },
            'vad': {
                'threshold': VAD_THRESHOLD,
                'min_speech_duration': VAD_MIN_SPEECH_DURATION,
                'min_silence_duration': VAD_MIN_SILENCE_DURATION,
                'padding_seconds': PADDING_SECONDS,
            },
        },
        'summary': summary,
        'files': completed_file_metrics,
        'combine': combine_metrics,
    }
    published_metrics_path, metrics_error = _publish_metrics_report(
        metrics_path, report
    )
    if metrics_error:
        print(f"Metrics report error: {metrics_error}")

    console = Console()
    console.print(_build_file_metrics_table(completed_file_metrics))
    console.print(_build_run_metrics_table(summary, published_metrics_path))

    if errors:
        print("\nErrors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        sys.exit(1)
    if combine_error:
        print(f"Combine error: {combine_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
