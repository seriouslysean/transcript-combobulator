"""Map timestamps inside a packed clip back to the source recording.

When VAD islands are packed into one clip, the silence between them is not
carried into the clip. Each island becomes a piece with its offset inside the
clip and its start in the source, so whisper's clip-relative timestamps can
be mapped back without ever attributing speech to removed silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True, slots=True)
class Piece:
    clip_offset: float      # seconds from clip start where this island begins
    source_start: float     # seconds in the source recording (padded clip start)
    duration: float         # seconds of this island inside the clip

    @property
    def clip_end(self) -> float:
        return self.clip_offset + self.duration


@dataclass(frozen=True, slots=True)
class ClipTimeMap:
    pieces: tuple[Piece, ...]

    def __post_init__(self) -> None:
        if not self.pieces:
            raise ValueError("ClipTimeMap needs at least one piece")
        offsets = [p.clip_offset for p in self.pieces]
        if offsets != sorted(offsets):
            raise ValueError("pieces must be in clip order")

    @classmethod
    def from_mapping_entry(cls, entry: dict[str, Any]) -> Union[float, ClipTimeMap]:
        """A plain offset for single-island entries, a map for packed ones.

        Entries written before ``clip_start_seconds`` existed fall back to the
        (late) speech start, matching the previous behaviour for old caches.
        """
        pieces = entry.get('pieces')
        if not pieces:
            return float(entry.get('clip_start_seconds', entry['start_seconds']))
        return cls(tuple(
            Piece(
                clip_offset=float(p['clip_offset_seconds']),
                source_start=float(p['source_start_seconds']),
                duration=float(p['duration_seconds']),
            )
            for p in pieces
        ))

    def _piece_for(self, clip_time: float) -> Piece:
        """The island a clip time belongs to.

        A time that falls in the inserted gap after an island is assigned to
        the next island, so a cue that starts a hair early in the gap lands
        on the speech it precedes rather than in silence that does not exist
        in the source.
        """
        current = self.pieces[0]
        for piece in self.pieces:
            if clip_time >= piece.clip_end and piece is not self.pieces[-1]:
                continue
            if clip_time < piece.clip_offset:
                return piece
            current = piece
            if clip_time < piece.clip_end:
                return piece
        return current

    def crosses_splice(self, start: float, end: float, tolerance: float = 0.1) -> bool:
        """True if a clip-relative span reaches into a later island.

        Whisper sometimes emits one segment over two packed islands; its text
        would then be attributed to the first island's time. The caller
        re-decodes the islands individually when this happens.
        """
        piece = self._piece_for(start)
        later = [p for p in self.pieces if p.clip_offset > piece.clip_offset]
        if not later:
            return False
        return end > later[0].clip_offset + tolerance

    def map_span(self, start: float, end: float) -> tuple[float, float]:
        """Clip-relative (start, end) -> source seconds.

        Both ends are mapped through the piece that contains ``start``; an
        end that runs past that island's splice is clamped to the island's
        end, so no cue is ever stretched across removed silence.
        """
        piece = self._piece_for(start)
        local_start = min(max(start, piece.clip_offset), piece.clip_end)
        local_end = min(max(end, local_start), piece.clip_end)
        source_start = piece.source_start + (local_start - piece.clip_offset)
        source_end = piece.source_start + (local_end - piece.clip_offset)
        if source_end <= source_start:
            source_end = source_start + 0.01
        return source_start, source_end


Offset = Union[float, ClipTimeMap]
