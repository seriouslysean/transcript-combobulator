"""Logging setup shared across modules."""

import logging
import os
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure the root logger. Level is read from LOG_LEVEL env if set."""
    fmt = format_string or _DEFAULT_FORMAT
    log_level = os.getenv('LOG_LEVEL', level).upper()

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(file_handler)

    return logging.getLogger(__name__)


class _ContextFilter(logging.Filter):
    """Inject fixed fields (e.g. the audio file a worker owns) into records."""

    def __init__(self, context: dict[str, str]) -> None:
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._context.items():
            setattr(record, key, value)
        return True


def add_file_handler(
    log_file: Path,
    level: str = "INFO",
    context: Optional[dict[str, str]] = None,
) -> logging.Handler:
    """Attach a file handler to the root logger without adding a stream handler.

    Batch workers and the batch parent use this so logs persist to disk while
    the rich progress UI keeps the terminal. Remove the returned handler with
    ``remove_file_handler`` when the unit of work ends; pooled workers are
    reused across files.
    """
    log_level = getattr(logging, os.getenv('LOG_LEVEL', level).upper())
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(log_level)
    if context:
        fields = ' '.join(f'%({key})s' for key in context)
        fmt = f"%(asctime)s [{fields}] %(name)s - %(levelname)s - %(message)s"
        handler.addFilter(_ContextFilter(context))
    else:
        fmt = _DEFAULT_FORMAT
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > log_level:
        root.setLevel(log_level)
    return handler


def remove_file_handler(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger."""
    return logging.getLogger(name)
