"""File-handler helpers used by batch workers."""

import logging
from pathlib import Path

from transcript_combobulator.logging_config import add_file_handler, remove_file_handler


def test_file_handler_persists_records_with_context(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "session.log"
    root = logging.getLogger()
    before = list(root.handlers)

    handler = add_file_handler(log_file, context={"audio_file": "3-nilbits.flac"})
    try:
        logging.getLogger("transcript_combobulator.vad").info("Found %d segments", 12)
    finally:
        remove_file_handler(handler)

    text = log_file.read_text(encoding="utf-8")
    assert "[3-nilbits.flac]" in text
    assert "transcript_combobulator.vad - INFO - Found 12 segments" in text
    assert root.handlers == before


def test_file_handler_without_context_uses_plain_format(tmp_path: Path) -> None:
    log_file = tmp_path / "plain.log"
    handler = add_file_handler(log_file)
    try:
        logging.getLogger("process_batch").warning("Memory guard: running 1")
    finally:
        remove_file_handler(handler)

    line = log_file.read_text(encoding="utf-8").strip()
    assert line.endswith("process_batch - WARNING - Memory guard: running 1")
    assert "[" not in line.split(" - ")[0].split(" ", 2)[-1]
