"""Config-aware completion manifests for resumable per-file processing."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.config import get_pipeline_fingerprint_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

PIPELINE_CACHE_VERSION = 1


def get_manifest_path(output_dir: Path, audio_stem: str) -> Path:
    """Return the completion-manifest path for an input audio file."""
    return output_dir / f"{audio_stem}_pipeline_manifest.json"


def build_pipeline_fingerprint(input_path: Path) -> str:
    """Hash source metadata and output-affecting configuration."""
    stat = input_path.stat()
    payload = {
        'cache_version': PIPELINE_CACHE_VERSION,
        'source': {
            'path': str(input_path.resolve()),
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        },
        'settings': get_pipeline_fingerprint_settings(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def is_pipeline_complete(input_path: Path, manifest_path: Path) -> bool:
    """Return whether a manifest matches the source/config and all artifacts exist."""
    if not manifest_path.is_file():
        return False

    try:
        with open(manifest_path, encoding='utf-8') as f:
            manifest: dict[str, Any] = json.load(f)
        if manifest.get('fingerprint') != build_pipeline_fingerprint(input_path):
            return False
        artifacts = manifest.get('artifacts')
        if not isinstance(artifacts, list) or not artifacts:
            return False
        return all(isinstance(path, str) and Path(path).is_file() for path in artifacts)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning(f"Ignoring invalid pipeline manifest: {manifest_path}")
        return False


def write_pipeline_manifest(
    input_path: Path,
    manifest_path: Path,
    artifacts: Iterable[Path],
) -> None:
    """Atomically record a completed pipeline run and its required artifacts."""
    artifact_paths = [str(path.resolve()) for path in artifacts]
    payload = {
        'fingerprint': build_pipeline_fingerprint(input_path),
        'artifacts': artifact_paths,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    try:
        with open(temporary_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        temporary_path.replace(manifest_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        logger.warning(f"Could not write pipeline manifest: {manifest_path}")
