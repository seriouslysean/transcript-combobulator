"""Packed-clip timestamp mapping."""

import pytest

from src.timemap import ClipTimeMap, Piece


def _two_islands() -> ClipTimeMap:
    # island A: source 10.0-14.0 at clip 0.0-4.0; 0.5 s gap; island B: source 60.0-63.0 at clip 4.5-7.5
    return ClipTimeMap((
        Piece(clip_offset=0.0, source_start=10.0, duration=4.0),
        Piece(clip_offset=4.5, source_start=60.0, duration=3.0),
    ))


def test_single_island_entry_is_a_plain_offset() -> None:
    assert ClipTimeMap.from_mapping_entry({"start_seconds": 5.3, "clip_start_seconds": 5.0}) == 5.0
    assert ClipTimeMap.from_mapping_entry({"start_seconds": 5.3}) == 5.3


def test_packed_entry_builds_map() -> None:
    entry = {
        "start_seconds": 10.3,
        "pieces": [
            {"clip_offset_seconds": 0.0, "source_start_seconds": 10.0, "duration_seconds": 4.0},
            {"clip_offset_seconds": 4.5, "source_start_seconds": 60.0, "duration_seconds": 3.0},
        ],
    }
    tm = ClipTimeMap.from_mapping_entry(entry)
    assert isinstance(tm, ClipTimeMap)
    assert tm.map_span(1.0, 2.0) == (11.0, 12.0)
    assert tm.map_span(5.0, 6.0) == (60.5, 61.5)


def test_cue_crossing_splice_is_clamped_to_its_island() -> None:
    tm = _two_islands()
    start, end = tm.map_span(3.0, 5.5)
    assert start == 13.0
    assert end == 14.0  # never stretched across the removed 46 s of silence


def test_cue_starting_in_gap_moves_to_next_island() -> None:
    tm = _two_islands()
    start, end = tm.map_span(4.2, 5.0)
    assert start == 60.0
    assert end == 60.5


def test_cue_past_last_island_clamps() -> None:
    tm = _two_islands()
    start, end = tm.map_span(9.0, 9.5)
    assert start == 63.0
    assert end == pytest.approx(63.01)


def test_pieces_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        ClipTimeMap((Piece(4.5, 60.0, 3.0), Piece(0.0, 10.0, 4.0)))


def test_crosses_splice_only_when_reaching_next_island() -> None:
    tm = _two_islands()
    assert not tm.crosses_splice(0.0, 3.9)
    assert not tm.crosses_splice(0.0, 4.4)   # bleeds into the gap, not the island
    assert tm.crosses_splice(0.0, 5.0)
    assert not tm.crosses_splice(5.0, 7.5)   # last island, nothing later
