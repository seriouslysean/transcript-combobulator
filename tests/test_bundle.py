"""Reading a recording bundle (docs/recording-bundle.md): pure Python over synthetic folders."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from transcript_combobulator.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    load_bundle,
)


def make_bundle(root: Path, tracks: list[dict[str, Any]], **manifest: Any) -> Path:
    """Write audio files and a manifest; each track dict may override any manifest field."""
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, t in enumerate(tracks, 1):
        file = t.get("file", f"{i}-speaker{i}.flac")
        data = t.pop("_bytes", b"fLaC" + bytes(100 * i))
        if t.pop("_write", True) and file:
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            (root / file).write_bytes(data)
        entry = {"track_id": str(i), "file": file, "sha256": hashlib.sha256(data).hexdigest(),
                 "speaker_key": f"discord:{i}", "username": f"speaker{i}"}
        entry.update(t)
        entries.append(entry)
    body = {"schema_version": 1, "recording_id": "rec-1", "source": "test",
            "started_at": "2026-10-18T23:04:11.000Z", "tracks": entries}
    body.update(manifest)
    (root / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
    return root


def test_schema_version_is_one() -> None:
    assert BUNDLE_SCHEMA_VERSION == 1


def test_a_valid_bundle_loads_in_manifest_order(tmp_path: Path) -> None:
    b = load_bundle(make_bundle(tmp_path / "b", [{}, {"username": None}, {}]))
    assert b.recording_id == "rec-1"
    assert b.started_at == "2026-10-18T23:04:11.000Z"
    assert [t.track_id for t in b.tracks] == ["1", "2", "3"]
    assert [t.stem for t in b.tracks] == ["1-speaker1", "2-speaker2", "3-speaker3"]
    assert b.tracks[1].username is None
    assert b.tracks[2].size > b.tracks[0].size
    assert b.tracks[0].path == (tmp_path / "b" / "1-speaker1.flac").resolve()


def test_unknown_fields_are_ignored(tmp_path: Path) -> None:
    b = load_bundle(make_bundle(tmp_path / "b", [{"extra": 1}], future_field="x"))
    assert len(b.tracks) == 1


def test_missing_manifest_is_not_a_bundle(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    with pytest.raises(BundleError, match="incomplete: no manifest.json"):
        load_bundle(tmp_path / "b")


@pytest.mark.parametrize("version", [0, 2, "1", None])
def test_other_schema_versions_are_refused(tmp_path: Path, version: Any) -> None:
    with pytest.raises(BundleError, match="schema_version"):
        load_bundle(make_bundle(tmp_path / "b", [{}], schema_version=version))


def test_empty_tracks_are_refused(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="tracks"):
        load_bundle(make_bundle(tmp_path / "b", []))


def test_bad_json_is_refused(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "manifest.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(BundleError, match="manifest.json"):
        load_bundle(tmp_path / "b")


@pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.flac", "a/../../x.flac"])
def test_paths_must_stay_inside(tmp_path: Path, bad: str) -> None:
    with pytest.raises(BundleError, match="track 1"):
        load_bundle(make_bundle(tmp_path / "b", [{"file": bad, "_write": False}]))


def test_a_symlink_out_of_the_bundle_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.flac"
    outside.write_bytes(b"fLaC")
    root = make_bundle(tmp_path / "b", [{"file": "1-x.flac", "_write": False,
                                         "sha256": hashlib.sha256(b"fLaC").hexdigest()}])
    (root / "1-x.flac").symlink_to(outside)
    with pytest.raises(BundleError, match="outside"):
        load_bundle(root)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="2-speaker2.flac"):
        load_bundle(make_bundle(tmp_path / "b", [{}, {"_write": False}]))


def test_a_wrong_checksum_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="sha256.*2-speaker2.flac|2-speaker2.flac.*sha256"):
        load_bundle(make_bundle(tmp_path / "b", [{}, {"sha256": "0" * 64}]))


def test_stems_must_be_unique_ignoring_case(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="Pikel.flac.*pikel.wav|pikel.wav.*Pikel.flac"):
        load_bundle(make_bundle(tmp_path / "b", [{"file": "Pikel.flac"}, {"file": "pikel.wav"}]))


def test_two_tracks_with_one_speaker_key_are_refused(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="discord:7"):
        load_bundle(make_bundle(tmp_path / "b", [{"speaker_key": "discord:7"}, {"speaker_key": "discord:7"}]))


def test_null_speakers_are_allowed(tmp_path: Path) -> None:
    b = load_bundle(make_bundle(tmp_path / "b", [{"speaker_key": None, "username": None}, {"speaker_key": None}]))
    assert [t.speaker_key for t in b.tracks] == [None, None]


def test_files_in_subfolders_are_refused_for_now(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="a/1.flac"):
        load_bundle(make_bundle(tmp_path / "b", [{"file": "a/1.flac"}]))


def test_audio_the_manifest_does_not_list_is_refused(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "b", [{}])
    (root / "stray.flac").write_bytes(b"fLaC")
    (root / "notes.txt").write_text("not audio")
    with pytest.raises(BundleError, match="stray.flac"):
        load_bundle(root)


def test_a_format_run_cannot_read_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="1-x.mid"):
        load_bundle(make_bundle(tmp_path / "b", [{"file": "1-x.mid"}]))


@pytest.mark.parametrize("field", ["track_id", "file", "sha256"])
def test_required_track_fields(tmp_path: Path, field: str) -> None:
    with pytest.raises(BundleError, match=field):
        load_bundle(make_bundle(tmp_path / "b", [{field: None}]))


def test_track_ids_must_be_unique(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="track_id"):
        load_bundle(make_bundle(tmp_path / "b", [{"track_id": "1"}, {"track_id": "1"}]))


@pytest.mark.parametrize("started_at", ["2026-10-18", "2026-10-18T23:04:11", "yesterday", 5])
def test_started_at_needs_an_offset(tmp_path: Path, started_at: Any) -> None:
    with pytest.raises(BundleError, match="started_at"):
        load_bundle(make_bundle(tmp_path / "b", [{}], started_at=started_at))


def test_started_at_may_be_null(tmp_path: Path) -> None:
    assert load_bundle(make_bundle(tmp_path / "b", [{}], started_at=None)).started_at is None


def test_recording_id_is_required(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="recording_id"):
        load_bundle(make_bundle(tmp_path / "b", [{}], recording_id=""))


def test_speaker_fields_must_be_present_even_when_null(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "b", [{}])
    body = json.loads((root / "manifest.json").read_text())
    del body["tracks"][0]["speaker_key"]
    (root / "manifest.json").write_text(json.dumps(body))
    with pytest.raises(BundleError, match="speaker_key"):
        load_bundle(root)
