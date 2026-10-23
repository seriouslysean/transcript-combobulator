"""Pinned sample downloads: synthetic bytes, no network."""

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

from transcript_combobulator import samples
from transcript_combobulator.samples import SampleError, download_verified

DATA = b"RIFF" + bytes(3000)
SHA = hashlib.sha256(DATA).hexdigest()


def serve(monkeypatch: pytest.MonkeyPatch, data: bytes) -> list[str]:
    calls: list[str] = []

    def urlopen(url: str, timeout: float = 0) -> Any:
        calls.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr(samples.urllib.request, "urlopen", urlopen)
    return calls


def test_a_matching_download_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, DATA)
    dest = download_verified(
        "https://example.invalid/a.wav", tmp_path / "c" / "a.wav", SHA
    )
    assert dest.read_bytes() == DATA
    assert not (tmp_path / "c" / "a.wav.part").exists()


def test_a_cached_file_with_the_right_checksum_is_not_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.wav").write_bytes(DATA)
    calls = serve(monkeypatch, b"never read")
    download_verified("https://example.invalid/a.wav", tmp_path / "a.wav", SHA)
    assert calls == []


def test_a_changed_source_is_refused_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, DATA + b"x")
    with pytest.raises(SampleError, match="sha256"):
        download_verified("https://example.invalid/a.wav", tmp_path / "a.wav", SHA)
    assert list(tmp_path.iterdir()) == []


def test_a_stale_cached_file_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.wav").write_bytes(b"old")
    serve(monkeypatch, DATA)
    assert (
        download_verified(
            "https://example.invalid/a.wav", tmp_path / "a.wav", SHA
        ).read_bytes()
        == DATA
    )
