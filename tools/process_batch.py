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

from rich.live import Live
from rich.table import Table
from rich.text import Text

# Ensure spawn method for macOS torch compatibility
multiprocessing.set_start_method("spawn", force=True)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


def find_audio_files(target_dir: Path) -> list[Path]:
    """Find audio files in target directory, excluding converted files."""
    files = []
    for f in sorted(target_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS and "_converted" not in f.stem:
            files.append(f)
    return files


def _worker(file_path: str, status_dict: dict, status_key: str, torch_threads: int) -> tuple[str, str, str]:
    """Worker function that runs in a subprocess.

    Returns:
        (filename, "done" or "error", error_message_or_empty)
    """
    # Lower scheduling priority so foreground apps stay responsive
    try:
        os.nice(10)
    except OSError:
        pass

    # Limit torch threads per worker
    if torch_threads > 0:
        try:
            import torch
            torch.set_num_threads(torch_threads)
        except Exception:
            pass

    # Suppress all logging output — the rich table is the UI
    logging.disable(logging.CRITICAL)

    # Also suppress stdout/stderr from submodules
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        from tools.process_single_file import main as process_main
        process_main(file_path, status_dict=status_dict, status_key=status_key)
        return (Path(file_path).name, "done", "")
    except Exception as e:
        return (Path(file_path).name, "error", str(e))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        logging.disable(logging.NOTSET)


def _build_table(file_names: list[str], status_dict: dict, max_workers: int) -> Table:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process audio files with live progress")
    parser.add_argument("target_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--session", type=str, default=None, help="Session name for combine step")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.is_dir():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    files = find_audio_files(target_dir)
    if not files:
        print(f"No audio files found in {target_dir}")
        sys.exit(1)

    # Load config (imports dotenv, reads PARALLEL_JOBS etc.)
    from src.config import PARALLEL_JOBS, TORCH_THREADS

    max_workers = PARALLEL_JOBS
    torch_threads = TORCH_THREADS
    if torch_threads == 0:
        torch_threads = max(1, (os.cpu_count() or 4) // 4)

    file_names = [f.name for f in files]

    # Shared dict for cross-process status updates
    manager = multiprocessing.Manager()
    status_dict = manager.dict({name: "waiting" for name in file_names})

    print(f"Processing {len(files)} audio files in {target_dir}")
    start_time = time.monotonic()

    errors: list[tuple[str, str]] = []

    # Install a SIGINT handler that forcefully kills child processes.
    # Without this, ProcessPoolExecutor and Manager ignore the first Ctrl+C.
    executor_ref = None

    def _sigint_handler(signum, frame):
        print("\nCancelled.")
        if executor_ref is not None:
            for pid in executor_ref._processes:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        sys.exit(130)

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)

    try:
        with Live(_build_table(file_names, status_dict, max_workers), refresh_per_second=4) as live:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                executor_ref = executor
                futures = {}
                for f in files:
                    fut = executor.submit(
                        _worker,
                        str(f),
                        status_dict,
                        f.name,
                        torch_threads,
                    )
                    futures[fut] = f.name

                # Poll for status updates instead of blocking on as_completed
                while True:
                    live.update(_build_table(file_names, status_dict, max_workers))

                    done_futures = [f for f in futures if f.done()]
                    for fut in done_futures:
                        if fut in futures:
                            name, status, err_msg = fut.result()
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

    elapsed = time.monotonic() - start_time
    print(f"Processed {len(files)} files in {_format_duration(elapsed)}")

    if errors:
        print("\nErrors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        sys.exit(1)

    # Combine step
    session_name = args.session or target_dir.name
    print("Combining transcripts...")
    from src.combine import combine_transcripts_from_env
    from src.config import OUTPUT_DIR
    try:
        output_files = combine_transcripts_from_env(OUTPUT_DIR, session_name)
        for p in output_files:
            print(f"Combined transcript: {p}")
    except Exception as e:
        print(f"Combine error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
