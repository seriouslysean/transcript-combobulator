"""Pinned sample downloads: synthetic bytes, no network."""

import hashlib
import io
import zipfile
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


def _ami_zip(path: Path) -> Path:
    """A two-speaker stand-in for AMI's annotation zip: one meeting, its words and segments."""
    nite = 'xmlns:nite="http://nite.sourceforge.net/"'
    m = samples.AMI_MEETING
    meetings = (
        f'<corpus {nite}><meeting observation="{m}">'
        '<speaker nxt_agent="A" channel="1" global_name="MEE001"/>'
        '<speaker nxt_agent="B" channel="0" global_name="FEE002"/>'
        "</meeting></corpus>"
    )
    t = samples.AMI_START_SECONDS

    def w(i: str, s: float, text: str, **attrs: str) -> str:
        extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
        return f'<w nite:id="{i}" starttime="{t + s}" endtime="{t + s + 0.4}"{extra}>{text}</w>'

    words_a = (
        f"<nite:root {nite}>"
        + w("a1", -1.0, "before")  # starts before the window
        + w("a0", -0.2, "edge")  # straddles the window's start
        + w("a2", 1.0, "hello")
        + w("a3", 1.4, ",", punc="true")
        + w("a4", 1.5, "wor", trunc="true")
        + '<vocalsound nite:id="a5" starttime="242" endtime="243" type="laugh"/>'
        + w("a6", 2.0, "world")
        + w("a7", 2.4, ".", punc="true")
        + w("a8", samples.AMI_SECONDS - 0.2, "late")  # straddles the window's end
        + w("a9", samples.AMI_SECONDS + 1.0, "after")
        + "</nite:root>"
    )
    words_b = (
        f"<nite:root {nite}>"
        + w("b1", 0.5, "hi")
        + w("b2", 3.0, "bye")
        + "</nite:root>"
    )

    def segs(agent: str, *spans: tuple[str, str]) -> str:
        hrefs = [
            f"{m}.{agent}.words.xml#id({first})..id({last})" for first, last in spans
        ]
        return (
            f"<nite:root {nite}>"
            + "".join(
                f'<segment nite:id="s{agent}{n}"><nite:child href="{h}"/></segment>'
                for n, h in enumerate(hrefs)
            )
            + "</nite:root>"
        )

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("corpusResources/meetings.xml", meetings)
        z.writestr(f"words/{m}.A.words.xml", words_a)
        z.writestr(f"words/{m}.B.words.xml", words_b)
        z.writestr(f"segments/{m}.A.segments.xml", segs("A", ("a1", "a9")))
        z.writestr(
            f"segments/{m}.B.segments.xml", segs("B", ("b1", "b1"), ("b2", "b2"))
        )
    return path


def test_the_ami_reference_maps_channels_and_filters_words(tmp_path: Path) -> None:
    text, sidecar = samples.ami_human_reference(_ami_zip(tmp_path / "ami.zip"))
    body = text.split("\n\n", 1)[1].splitlines()
    # Channel picks the label, not the agent letter; lines sort by start time; words that
    # cross either edge of the window are dropped.
    assert body == [
        "Speaker A: hi [00:00:00.500 --> 00:00:00.900]",
        "Speaker B: hello, world. [00:00:01.000 --> 00:00:02.400]",
        "Speaker A: bye [00:00:03.000 --> 00:00:03.400]",
    ]
    assert sidecar["speakers"]["Speaker B"]["nxt_agent"] == "A"
    assert sidecar["speakers"]["Speaker B"]["headset"].endswith("Headset-1.wav")
    start = samples.AMI_START_SECONDS
    assert sidecar["lines"][1]["words"] == [
        ["a2", start + 1.0, start + 1.4],
        ["a6", start + 2.0, start + 2.4],
    ]
    assert sidecar["sha256"] == samples.AMI_ANNOTATIONS_SHA256
