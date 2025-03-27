"""Simple print utilities."""

def info(message: str) -> None:
    """Print an info message."""
    print(f"INFO: {message}")

def error(message: str) -> None:
    """Print an error message."""
    print(f"ERROR: {message}")

def warning(message: str) -> None:
    """Print a warning message."""
    print(f"WARNING: {message}")

def debug(message: str) -> None:
    """Print a debug message."""
    print(f"DEBUG: {message}")
