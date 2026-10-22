"""Read and verify a recording bundle: a folder of audio plus ``manifest.json``.

The format is ``docs/recording-bundle.md``. Every refusal raises ``BundleError`` with a
message naming the file or track at fault. Nothing here reads audio beyond hashing it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from transcript_combobulator.batch import AUDIO_EXTENSIONS, find_audio_files

BUNDLE_SCHEMA_VERSION = 1
MANIFEST_NAME = 'manifest.json'

_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


class BundleError(Exception):
    """The folder is not a bundle combobulator accepts."""


@dataclass(frozen=True)
class Track:
    track_id: str
    file: str
    path: Path
    stem: str
    sha256: str
    speaker_key: str | None
    username: str | None
    size: int


@dataclass(frozen=True)
class Bundle:
    root: Path
    recording_id: str
    source: str | None
    started_at: str | None
    tracks: list[Track]


def _optional_str(entry: dict[str, Any], key: str, where: str) -> str | None:
    if key not in entry:
        raise BundleError(f'{where}: {key} is required (null when unknown)')
    value = entry[key]
    if value is not None and (not isinstance(value, str) or not value):
        raise BundleError(f'{where}: {key} must be a non-empty string or null')
    return value


def _track(root: Path, index: int, entry: Any) -> Track:
    where = f'track #{index}'
    if not isinstance(entry, dict):
        raise BundleError(f'{where}: must be an object')
    track_id = entry.get('track_id')
    if not isinstance(track_id, str) or not track_id:
        raise BundleError(f'{where}: track_id must be a non-empty string')
    where = f'track {track_id}'
    file = entry.get('file')
    if not isinstance(file, str) or not file:
        raise BundleError(f'{where}: file must be a non-empty string')
    where = f'track {track_id} ({file})'
    sha256 = entry.get('sha256')
    if not isinstance(sha256, str) or not _SHA256_RE.match(sha256):
        raise BundleError(f'{where}: sha256 must be 64 lowercase hex digits')
    speaker_key = _optional_str(entry, 'speaker_key', where)
    username = _optional_str(entry, 'username', where)

    rel = PurePosixPath(file)
    if rel.is_absolute() or file.startswith('/'):
        raise BundleError(f'{where}: file must be a path relative to the bundle')
    if '..' in rel.parts:
        raise BundleError(f'{where}: file must not contain ".."')
    if len(rel.parts) != 1:
        raise BundleError(f'{where}: files in subfolders are not supported yet; run reads only the top of a folder')
    path = (root / file).resolve()
    if not path.is_relative_to(root):
        raise BundleError(f'{where}: file resolves outside the bundle')
    if rel.suffix.lower() not in AUDIO_EXTENSIONS:
        raise BundleError(
            f'{where}: run cannot read {rel.suffix or "a file with no extension"}; '
            f'formats: {", ".join(sorted(AUDIO_EXTENSIONS))}'
        )
    if not path.is_file():
        raise BundleError(f'{where}: file is missing')
    return Track(track_id, file, path, rel.stem, sha256, speaker_key, username, path.stat().st_size)


def _first_duplicate(tracks: list[Track], key: Callable[[Track], str | None]) -> tuple[Track, Track] | None:
    seen: dict[str, Track] = {}
    for t in tracks:
        k = key(t)
        if k is None:
            continue
        if k in seen:
            return seen[k], t
        seen[k] = t
    return None


def _sha256(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_bundle(root: Path) -> Bundle:
    """Read ``root/manifest.json`` and check the bundle against version 1, hashes last."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise BundleError(f'{root}: not a folder')
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        raise BundleError(f'{root}: not a bundle (incomplete: no {MANIFEST_NAME})')
    try:
        body = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BundleError(f'{manifest}: cannot read {MANIFEST_NAME}: {e}') from e
    if not isinstance(body, dict):
        raise BundleError(f'{manifest}: {MANIFEST_NAME} must hold an object')

    version = body.get('schema_version')
    if type(version) is not int or version != BUNDLE_SCHEMA_VERSION:
        raise BundleError(
            f'{manifest}: schema_version {version!r} is not supported; this combobulator reads '
            f'{BUNDLE_SCHEMA_VERSION}'
        )
    recording_id = body.get('recording_id')
    if not isinstance(recording_id, str) or not recording_id:
        raise BundleError(f'{manifest}: recording_id must be a non-empty string')
    source = body.get('source')
    if source is not None and not isinstance(source, str):
        raise BundleError(f'{manifest}: source must be a string')
    started_at = body.get('started_at')
    if started_at is not None:
        try:
            parsed = datetime.fromisoformat(started_at) if isinstance(started_at, str) else None
        except ValueError:
            parsed = None
        if parsed is None or parsed.tzinfo is None:
            raise BundleError(f'{manifest}: started_at {started_at!r} must be ISO 8601 with an offset, or null')
    entries = body.get('tracks')
    if not isinstance(entries, list) or not entries:
        raise BundleError(f'{manifest}: tracks must list at least one track')

    tracks = [_track(root, i, e) for i, e in enumerate(entries, 1)]
    checks: list[tuple[Callable[[Track], str | None], str]] = [
        (lambda t: t.track_id, 'track_id'),
        (lambda t: t.stem.casefold(), 'file stem (ignoring case)'),
        (lambda t: t.speaker_key, 'speaker_key'),
    ]
    for key, what in checks:
        if dup := _first_duplicate(tracks, key):
            a, b = dup
            raise BundleError(f'tracks {a.track_id} ({a.file}) and {b.track_id} ({b.file}) share a {what}: {key(a)}')

    listed = {t.file.casefold() for t in tracks}
    unlisted = [f.name for f in find_audio_files(root) if f.name.casefold() not in listed]
    if unlisted:
        raise BundleError(
            f'{root}: audio not in the manifest, which run would transcribe as extra speakers: {", ".join(unlisted)}'
        )

    for t in tracks:
        if _sha256(t.path) != t.sha256:
            raise BundleError(f'track {t.track_id} ({t.file}): sha256 does not match the file')
    return Bundle(root, recording_id, source, started_at, tracks)
