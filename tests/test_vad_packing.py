"""Island packing in VAD (silero mocked; fast)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.vad import pack_islands, process_audio

SR = 16000


def test_pack_islands_respects_budget_and_gap() -> None:
    # lengths in samples: 5 s, 10 s, 20 s, 3 s, 40 s (oversize), 2 s
    islands = [(0, 5 * SR), (100 * SR, 110 * SR), (200 * SR, 220 * SR), (300 * SR, 303 * SR),
               (400 * SR, 440 * SR), (500 * SR, 502 * SR)]
    groups = pack_islands(islands, max_samples=28 * SR, gap_samples=SR // 2)
    # 5 + 0.5 + 10 = 15.5 fits; +0.5+20 would be 36 > 28 -> new group
    # 20 + 0.5 + 3 = 23.5 fits; 40 alone (oversize); 2 alone after it
    assert groups == [[0, 1], [2, 3], [4], [5]]


def test_pack_islands_off_is_identity_shape() -> None:
    assert pack_islands([], 28 * SR, SR // 2) == []
    assert pack_islands([(0, SR)], 28 * SR, SR // 2) == [[0]]


def _write_noise(path: Path, seconds: float) -> None:
    rng = np.random.default_rng(0)
    sf.write(str(path), (0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32), SR)


def test_process_audio_packs_and_records_pieces(tmp_path: Path) -> None:
    wav = tmp_path / "3-nilbits.wav"
    _write_noise(wav, 120.0)
    # three islands: 10-12, 30-33, 90-95 (seconds)
    stamps = [{"start": 10.0, "end": 12.0}, {"start": 30.0, "end": 33.0}, {"start": 90.0, "end": 95.0}]

    with patch("src.vad.get_speech_timestamps", return_value=stamps), \
         patch("src.vad.load_vad_model", return_value=object()), \
         patch("src.vad.VAD_PACK_ISLANDS", True), \
         patch("src.vad.VAD_PACK_MAX_SECONDS", 28.0), \
         patch("src.vad.VAD_PACK_GAP_SECONDS", 0.5), \
         patch("src.vad.PADDING_SECONDS", 0.3):
        _, segments = process_audio(wav)

    # 2.6 + 0.5 + 3.6 + 0.5 + 5.6 = 12.8 s -> one clip
    assert len(segments) == 1
    seg = segments[0]
    assert seg["start_seconds"] == 10.0 and seg["end_seconds"] == 95.0
    pieces = seg["pieces"]
    assert [round(p["source_start_seconds"], 3) for p in pieces] == [9.7, 29.7, 89.7]
    assert [round(p["clip_offset_seconds"], 3) for p in pieces] == [0.0, 3.1, 7.2]
    info = sf.info(seg["segment_file"])
    assert abs(info.duration - 12.8) < 0.01
    mapping = json.loads((tmp_path / "3-nilbits_mapping.json").read_text())
    assert mapping["segments"][0]["pieces"] == pieces


def test_process_audio_unpacked_writes_one_clip_per_island(tmp_path: Path) -> None:
    wav = tmp_path / "3-nilbits.wav"
    _write_noise(wav, 60.0)
    stamps = [{"start": 10.0, "end": 12.0}, {"start": 30.0, "end": 33.0}]
    with patch("src.vad.get_speech_timestamps", return_value=stamps), \
         patch("src.vad.load_vad_model", return_value=object()), \
         patch("src.vad.VAD_PACK_ISLANDS", False):
        _, segments = process_audio(wav)
    assert len(segments) == 2
    assert all("pieces" not in s for s in segments)
    assert abs(sf.info(segments[0]["segment_file"]).duration - 2.6) < 0.01
