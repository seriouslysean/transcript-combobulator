"""Write one session's own env file from a recording bundle: settings from a base, speakers from the tracks.

``combobulator run`` loads exactly one ``ENV_FILE``, and a session's speakers change from
game to game (who attended, who is the DM, a new player). So each session gets a full env
of its own, built from a recording bundle (``docs/recording-bundle.md``):

- **Settings** (everything that is not a ``TRANSCRIPT_N_*`` row) are copied from the base
  unchanged, line for line. The base is a campaign's env, or the repo's committed
  ``.env.example``. Speaker rows in the base are ignored.
- **Speaker rows** are rebuilt from the bundle's tracks, one per track, DM first. A
  speakers file says who is on which track (matched on ``speaker_key``, then ``username``).
  A track with no match gets a *provisional* row, marked ``PROVISIONAL`` in its
  description so it shows in the transcript's Summary header.
- ``TRANSCRIPT_N_USERNAME`` is the track's file stem, because ``run`` names each speaker's
  output folder after the stem and ``combine`` matches on it.

What a later edit costs (checked against ``config.get_stage_fingerprint_settings``; see
``tests/test_session_env.py``): ``NAME``, ``LABEL`` and ``DESCRIPTION`` are read only when
transcripts are combined, so fixing them needs ``combobulator combine <session>`` and no
inference. ``WHISPER_PROMPT`` is part of the inference fingerprint, so changing it in the
base re-runs transcription.

The env is written outside the bundle (``--output`` is required): a published bundle never
changes. An existing output file is never overwritten without ``--force``: it may hold edits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from transcript_combobulator.bundle import Bundle, BundleError, Track, load_bundle
from transcript_combobulator.combine import (
    CombineError,
    load_speaker_mapping,
    match_usernames,
    validate_speaker_mapping,
)
from transcript_combobulator.config import ROOT_DIR, parse_env_text, read_env_file

PROVISIONAL = "PROVISIONAL"
CHARACTER_UNSET = f"{PROVISIONAL}: character not set"
SEVERAL_PEOPLE = "may be several people"
DM_GUESSED = f"{PROVISIONAL}: DM guessed from the largest track"

_ROWS_HEADER = "# Speakers for this session: USERNAME is the track's file stem, DM first."
_TEXT_FIELDS = ("name", "label", "description")
_MAX_FIELD = 200
_ROW_RE = re.compile(r"\s*TRANSCRIPT_\d+_[A-Z]+\s*=")


class SessionEnvError(Exception):
    """The bundle, speakers file, base env, or resulting speaker map is unusable."""


@dataclass(frozen=True)
class Speaker:
    """One entry of the speakers file."""

    speaker_key: str | None
    username: str | None
    name: str
    label: str
    description: str
    dm: bool

    def describe(self) -> str:
        return f"speakers entry {self.speaker_key or self.username}"


@dataclass
class Row:
    track: Track
    name: str
    label: str
    description: str
    dm: bool = False

    @property
    def provisional(self) -> list[str]:
        """Which fields are guesses, read from the row itself so a kept file reports the same."""
        fallback = _fallback(self.track)
        unset = CHARACTER_UNSET in self.description
        guessed = DM_GUESSED in self.description
        flags = []
        if (unset or guessed) and self.name == fallback:
            flags.append("name")
        if unset and self.label == fallback:
            flags.append("label")
        if unset:
            flags.append("description")
        if guessed:
            flags.append("dm")
        return flags


@dataclass
class SessionEnvResult:
    path: Path
    status: str  # "written", or "kept" (an existing file was validated and left alone)
    bundle: Bundle
    rows: list[Row]
    session_date: str | None
    absent: list[dict[str, str | None]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def started_at(self) -> str | None:
        return self.bundle.started_at

    @property
    def provisional(self) -> list[str]:
        return [r.track.track_id for r in self.rows if r.provisional]

    def as_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "status": self.status,
            "recording_id": self.bundle.recording_id,
            "started_at": self.started_at,
            "session_date": self.session_date,
            "speakers": [
                {"track_id": r.track.track_id, "file": r.track.file, "speaker_key": r.track.speaker_key,
                 "username": r.track.username, "name": r.name, "label": r.label, "dm": r.dm,
                 "provisional": r.provisional}
                for r in self.rows
            ],
            "provisional": self.provisional,
            "absent": self.absent,
            "warnings": self.warnings,
        }


def _fallback(track: Track) -> str:
    return track.username or track.track_id


def _where(track: Track) -> str:
    return f"track {track.track_id} ({track.file})"


def _quote(value: str) -> str:
    """A value python-dotenv reads back unchanged (checked by the roundtrip test)."""
    if any(ord(c) < 32 for c in value):
        raise SessionEnvError("speaker fields cannot contain control characters or newlines")
    if "${" in value:
        # config.load_dotenv expands ${VAR} in any quoting, and no escape stops it
        raise SessionEnvError("speaker fields cannot contain ${ (it would be read as a variable)")
    if not any(c in value for c in '"\\$'):
        return f'"{value}"'
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def load_speakers(path: Path) -> list[Speaker]:
    """Read and check a speakers file: {"speakers": [{speaker_key?, username?, name, label, description, dm?}]}."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SessionEnvError(f"cannot read speakers file {path}: {e}") from None
    entries = data.get("speakers") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise SessionEnvError(f'{path}: a speakers file is {{"speakers": [...]}}')
    speakers = []
    for i, entry in enumerate(entries, 1):
        where = f"speakers entry #{i}"
        if not isinstance(entry, dict):
            raise SessionEnvError(f"{where}: must be an object")
        ids = {}
        for key in ("speaker_key", "username"):
            value = entry.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SessionEnvError(f"{where}: {key} must be a non-empty string or absent")
            ids[key] = value
        if not any(ids.values()):
            raise SessionEnvError(f"{where}: needs a speaker_key or username")
        where = f"speakers entry {ids['speaker_key'] or ids['username']}"
        text = {}
        for key in _TEXT_FIELDS:
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > _MAX_FIELD:
                raise SessionEnvError(f"{where}: {key} must be 1 to {_MAX_FIELD} characters of text")
            try:
                _quote(value)
            except SessionEnvError as e:
                raise SessionEnvError(f"{where}: {key}: {e}") from None
            text[key] = value.strip()
        dm = entry.get("dm", False)
        if not isinstance(dm, bool):
            raise SessionEnvError(f"{where}: dm must be true or false")
        speakers.append(Speaker(ids["speaker_key"], ids["username"], text["name"], text["label"],
                                text["description"], dm))
    return speakers


def _matches(track: Track, speaker: Speaker) -> bool:
    if track.speaker_key and speaker.speaker_key:
        return track.speaker_key == speaker.speaker_key
    return track.username is not None and track.username == speaker.username


def match_speakers(tracks: list[Track], speakers: list[Speaker]) -> dict[str, Speaker]:
    """track_id -> its speakers entry. Refuses a track or an entry that matches twice."""
    found: dict[str, Speaker] = {}
    for speaker in speakers:
        hits = [t for t in tracks if _matches(t, speaker)]
        if len(hits) > 1:
            raise SessionEnvError(f"{speaker.describe()} matches {_where(hits[0])} and {_where(hits[1])}")
    for t in tracks:
        hits_s = [s for s in speakers if _matches(t, s)]
        if len(hits_s) > 1:
            raise SessionEnvError(f"{_where(t)} matches {hits_s[0].describe()} and {hits_s[1].describe()}")
        if hits_s:
            found[t.track_id] = hits_s[0]
    return found


def _check_stems(tracks: list[Track]) -> None:
    """combine matches USERNAME as a whole token, so 'bob' would also claim 'bob-2'."""
    for a in tracks:
        for b in tracks:
            if a is not b and match_usernames(b.stem, {a.stem: {}}):
                raise SessionEnvError(
                    f"{_where(a)} and {_where(b)}: the stem {a.stem!r} is a whole word of {b.stem!r}, "
                    "so combine could not tell them apart; rename one"
                )


def _session_date(started_at: str | None, tz: str | None) -> str | None:
    try:
        zone = ZoneInfo(tz) if tz else None
    except (ZoneInfoNotFoundError, ValueError):
        raise SessionEnvError(f"unknown time zone: {tz}") from None
    if started_at is None:
        return None
    return datetime.fromisoformat(started_at).astimezone(zone).date().isoformat()


def _build_rows(tracks: list[Track], found: dict[str, Speaker], dm: str | None) -> tuple[list[Row], list[str]]:
    """Return (rows with the DM first, warnings)."""
    by_track: dict[str, Row] = {}
    for t in tracks:
        s = found.get(t.track_id)
        if s is not None:
            by_track[t.track_id] = Row(t, s.name, s.label, s.description)
        else:
            unset = CHARACTER_UNSET + (f"; {SEVERAL_PEOPLE}" if t.speaker_key is None else "")
            by_track[t.track_id] = Row(t, _fallback(t), _fallback(t), unset)

    warnings: list[str] = []
    marked = [tid for tid, s in found.items() if s.dm]
    if len(marked) > 1:
        raise SessionEnvError(f'"dm": true on more than one present speaker: tracks {", ".join(sorted(marked))}')
    guessed = False
    if dm is not None:
        if dm not in by_track:
            raise SessionEnvError(f"--dm {dm}: no track has that track_id ({', '.join(by_track)})")
        if marked and marked[0] != dm:
            raise SessionEnvError(f"--dm {dm} conflicts with the speakers file, which marks track {marked[0]} as DM")
        dm_id = dm
    elif marked:
        dm_id = marked[0]
    else:
        dm_id = max(tracks, key=lambda t: t.size).track_id
        guessed = True
        warnings.append(f"DM guessed from the largest track: {_where(by_track[dm_id].track)}; confirm or pass --dm")

    dm_row = by_track[dm_id]
    dm_row.dm = True
    if guessed:
        dm_row.label, dm_row.description = "DM", f"Dungeon Master ({DM_GUESSED})"
    elif dm_row.label.strip().lower() != "dm":
        dm_row.label, dm_row.description = "DM", "Dungeon Master"
    ordered = [dm_row] + [by_track[t.track_id] for t in tracks if t.track_id != dm_id]
    labels = [r.label for r in ordered]
    if len(set(labels)) != len(labels):
        warnings.append("two speakers share a label; their lines will not be told apart")
    return ordered, warnings


def _split_base(text: str) -> tuple[list[str], list[str], list[str]]:
    """(lines before the speaker rows, lines after, other assignments inside the block)."""
    lines = text.splitlines()
    rows = [i for i, line in enumerate(lines) if _ROW_RE.match(line)]
    if not rows:
        return lines, [], []
    start = rows[0]
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1  # comments directly above the first row ("# DM") belong to the rows
    block = lines[start : rows[-1] + 1]
    stray = [
        line for line in block
        if line.strip() and not line.lstrip().startswith("#") and not _ROW_RE.match(line)
    ]
    return lines[:start], lines[rows[-1] + 1 :], stray


def _settings(values: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in values.items() if not _ROW_RE.match(f"{k}=")}


def _render_rows(rows: list[Row]) -> list[str]:
    out = [_ROWS_HEADER]
    for i, row in enumerate(rows, 1):
        out.append("")
        who = f"track {row.track.track_id}" + (f", {row.track.username}" if row.track.username else "")
        flags = [f.upper() for f in row.provisional if f != "dm"]
        notes = [*(["DM guessed from the largest track; confirm or pass --dm"] if "dm" in row.provisional else []),
                 *([f"fill in {', '.join(flags)}"] if flags else [])]
        out.append(f"# {PROVISIONAL} ({who}): " + "; ".join(notes) if notes else f"# {'DM, ' if row.dm else ''}{who}")
        for key, value in (
            ("USERNAME", row.track.stem), ("NAME", row.name),
            ("LABEL", row.label), ("DESCRIPTION", row.description),
        ):
            out.append(f"TRANSCRIPT_{i}_{key}={_quote(value)}")
    return out


def _rows_from_values(values: dict[str, str], tracks: list[Track]) -> list[Row]:
    """The rows of an env file as written, one per track, DM first."""
    mapping = load_speaker_mapping(values)
    rows = []
    for t in tracks:
        (username,) = match_usernames(t.stem, mapping)  # validate_speaker_mapping ran first
        m = mapping[username]
        rows.append(Row(t, m["name"], m["label"], m["description"], m["label"].strip().lower() == "dm"))
    return sorted(rows, key=lambda r: not r.dm)


def build_session_env(
    bundle_dir: Path,
    *,
    speakers: Path | None = None,
    base: Path | None = None,
    dm: str | None = None,
    tz: str | None = None,
    output: Path,
    force: bool = False,
) -> SessionEnvResult:
    """Write ``output``, which must be outside the bundle: a published bundle never changes."""
    try:
        bundle = load_bundle(bundle_dir)
    except BundleError as e:
        raise SessionEnvError(str(e)) from None
    tracks = bundle.tracks
    stems = [t.stem for t in tracks]
    _check_stems(tracks)
    when = _session_date(bundle.started_at, tz)
    entries = load_speakers(speakers) if speakers else []
    found = match_speakers(tracks, entries)
    matched = {id(s) for s in found.values()}
    absent: list[dict[str, str | None]] = [
        {"speaker_key": s.speaker_key, "username": s.username} for s in entries if id(s) not in matched
    ]
    if Path(output).resolve().is_relative_to(bundle.root):
        raise SessionEnvError(f"--output {output} is inside the bundle; a published bundle never changes")

    if output.exists() and not force:
        values = read_env_file(output)
        try:
            validate_speaker_mapping(stems, env=values)
        except CombineError as e:
            raise SessionEnvError(
                f"{output} exists but does not fit these tracks ({e}); edit it, or rerun with --force to regenerate it"
            ) from None
        return SessionEnvResult(output, "kept", bundle, _rows_from_values(values, tracks), when, absent)

    base_path = base or ROOT_DIR / ".env.example"
    if not base_path.is_file():
        raise SessionEnvError(f"base env not found: {base_path}")
    base_values = read_env_file(base_path)
    rows, warnings = _build_rows(tracks, found, dm)

    before, after, stray = _split_base(base_path.read_text(encoding="utf-8"))
    source = base_path.name if base else ".env.example (the repo's default settings)"
    text_lines = [
        f"# Session speaker map for recording {bundle.recording_id}, written by `combobulator session-env`.",
        f"# Settings are copied unchanged from {source}; only the speaker rows below differ.",
        "# NAME, LABEL and DESCRIPTION only affect the combined transcript: edit them, then run",
        "# `combobulator combine <session>` (no transcription). WHISPER_PROMPT and VAD_* are",
        "# different: changing them after transcription re-runs it.",
        "",
        *before, *stray, "", *_render_rows(rows), *after,
    ]
    text = "\n".join(text_lines).rstrip("\n") + "\n"

    # Prove the file before writing it: settings identical to the base, and the real
    # validator (the one `run` uses) accepts the speaker rows for these tracks.
    parsed = parse_env_text(text)
    changed = sorted(
        k for k in _settings(parsed).keys() | _settings(base_values).keys()
        if parsed.get(k) != base_values.get(k)
    )
    if changed:
        raise SessionEnvError(f"internal error: settings {changed} would change; nothing written")
    try:
        validate_speaker_mapping(stems, env=parsed)
    except CombineError as e:
        raise SessionEnvError(str(e)) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".part")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, output)
    return SessionEnvResult(output, "written", bundle, rows, when, absent, warnings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="combobulator session-env",
        description=(
            "Write a session's own env file from a recording bundle "
            "and a base env. Tracks with no known speaker get provisional rows."
        ),
    )
    parser.add_argument("--bundle", type=Path, required=True, help="recording bundle folder (docs/recording-bundle.md)")
    parser.add_argument("--speakers", type=Path, metavar="FILE", help="who is on which track (JSON)")
    parser.add_argument("--base", type=Path, help="env whose settings are copied (default: the repo's .env.example)")
    parser.add_argument("--dm", metavar="TRACK_ID", help="which track is the DM (default: the speakers file's, else the largest track, flagged)")
    parser.add_argument("--tz", metavar="ZONE", help="IANA zone for the session date (default: this host's)")
    parser.add_argument("--output", type=Path, required=True, help="where to write the env; outside the bundle")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    opts = parser.parse_args(argv)
    try:
        result = build_session_env(
            opts.bundle, speakers=opts.speakers, base=opts.base, dm=opts.dm, tz=opts.tz,
            output=opts.output, force=opts.force,
        )
    except SessionEnvError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if opts.json:
        print(json.dumps(result.as_json(), indent=2))
        return
    print(f"{'Wrote' if result.status == 'written' else 'Kept existing'} {result.path}")
    for row in result.rows:
        flag = f"   <- PROVISIONAL ({', '.join(row.provisional)})" if row.provisional else ""
        print(f"  {row.track.stem}: {row.name} / {row.label}{flag}")
    for entry in result.absent:
        print(f"  not in this recording: {entry['speaker_key'] or entry['username']}")
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if result.provisional:
        print(
            f"{len(result.provisional)} speaker(s) are provisional. Fix the speakers file and rerun with "
            f"--force, or edit {result.path}; then `combobulator combine <session>` re-merges without transcribing.",
            file=sys.stderr,
        )
