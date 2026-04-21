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


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger."""
    return logging.getLogger(name)
