"""Tests for config-aware per-file pipeline completion manifests."""

import json
from pathlib import Path

from src.pipeline_cache import (
    build_pipeline_fingerprint,
    is_pipeline_complete,
    write_pipeline_manifest,
)


def test_fingerprint_changes_when_source_changes(tmp_path: Path) -> None:
    source = tmp_path / 'speaker.flac'
    source.write_bytes(b'first')
    original = build_pipeline_fingerprint(source)

    source.write_bytes(b'changed contents')

    assert build_pipeline_fingerprint(source) != original


def test_completed_manifest_requires_every_artifact(tmp_path: Path) -> None:
    source = tmp_path / 'speaker.flac'
    source.write_bytes(b'audio')
    first_artifact = tmp_path / 'speaker.vtt'
    second_artifact = tmp_path / 'speaker.json'
    first_artifact.touch()
    second_artifact.touch()
    manifest = tmp_path / 'manifest.json'

    write_pipeline_manifest(source, manifest, [first_artifact, second_artifact])
    assert is_pipeline_complete(source, manifest)

    second_artifact.unlink()
    assert not is_pipeline_complete(source, manifest)


def test_invalid_manifest_is_not_complete(tmp_path: Path) -> None:
    source = tmp_path / 'speaker.flac'
    source.touch()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{invalid', encoding='utf-8')

    assert not is_pipeline_complete(source, manifest)


def test_manifest_does_not_match_changed_source(tmp_path: Path) -> None:
    source = tmp_path / 'speaker.flac'
    source.write_bytes(b'original')
    artifact = tmp_path / 'speaker.vtt'
    artifact.touch()
    manifest = tmp_path / 'manifest.json'
    write_pipeline_manifest(source, manifest, [artifact])

    source.write_bytes(b'updated source')

    assert not is_pipeline_complete(source, manifest)


def test_manifest_contains_resolved_artifacts(tmp_path: Path) -> None:
    source = tmp_path / 'speaker.flac'
    source.touch()
    artifact = tmp_path / 'speaker.vtt'
    artifact.touch()
    manifest = tmp_path / 'manifest.json'

    write_pipeline_manifest(source, manifest, [artifact])

    data = json.loads(manifest.read_text(encoding='utf-8'))
    assert data['artifacts'] == [str(artifact.resolve())]
