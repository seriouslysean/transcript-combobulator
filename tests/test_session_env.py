"""A session's own env file from a recording bundle: what is copied, what is rebuilt, what an edit costs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from transcript_combobulator.combine import CombineError, combine_transcripts_from_env, validate_speaker_mapping
from transcript_combobulator.config import ROOT_DIR, read_env_file
from transcript_combobulator.session_env import SessionEnvError, SessionEnvResult, _quote, build_session_env, main

ENV = "session.env"

BASE = """\
# ── Whisper Settings ──
WHISPER_MODEL=large-v3-turbo
WHISPER_PROMPT="A Dungeons & Dragons session."
VAD_THRESHOLD=0.5

# ── Speaker Mappings ──
# Username extraction regex (kept: not a speaker row)
USERNAME_EXTRACTION_REGEX="^\\d+-(.+)"  # trailing comment

# DM
TRANSCRIPT_1_USERNAME="someone"
TRANSCRIPT_1_NAME="Someone"
TRANSCRIPT_1_LABEL="DM"
TRANSCRIPT_1_DESCRIPTION="Dungeon Master"

# ── Combination Settings ──
INCLUDE_TIMESTAMPS=false
CHUNKS=2
"""

ALICE = {"speaker_key": "discord:100000000000000001", "username": "alice.dm",
         "name": "Pat", "label": "DM", "description": "Dungeon Master", "dm": True}
BOB = {"speaker_key": "discord:100000000000000002", "username": "bob",
       "name": "Robin", "label": "Milo \"Maestro\" Fairweather", "description": "A bard"}
CAROL = {"speaker_key": "discord:100000000000000003", "username": "carol",
         "name": "Sam", "label": "Orsk", "description": "A fighter"}


def track(n: int, username: str | None, size: int, key: str | None = "default", file: str | None = None) -> dict[str, Any]:
    stem = f"{n}-{(username or 'track').replace('.', '_')}"
    return {"track_id": str(n), "file": file or f"{stem}.flac", "_size": size,
            "speaker_key": f"discord:10000000000000000{n}" if key == "default" else key, "username": username}


ALICE_T, BOB_T, CAROL_T = track(1, "alice.dm", 90), track(2, "bob", 40), track(3, "carol", 30)


def make_bundle(root: Path, tracks: list[dict[str, Any]], started_at: str | None = "2026-10-19T02:30:00.000Z") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for t in tracks:
        t = dict(t)
        data = b"fLaC" + bytes(t.pop("_size"))
        (root / t["file"]).write_bytes(data)
        entries.append({**t, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema_version": 1, "recording_id": "AbCdEf12345", "source": "test",
                "started_at": started_at, "tracks": entries}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def build(bundle: Path, **kw: Any) -> SessionEnvResult:
    """build_session_env with the env next to the bundle, as a caller would put it."""
    kw.setdefault("output", bundle.parent / ENV)
    return build_session_env(bundle, **kw)


def speakers_file(path: Path, *entries: dict[str, Any]) -> Path:
    path.write_text(json.dumps({"speakers": list(entries)}), encoding="utf-8")
    return path


@pytest.fixture
def base(tmp_path: Path) -> Path:
    path = tmp_path / ".env.campaign"
    path.write_text(BASE, encoding="utf-8")
    return path


def rows(path: Path) -> list[tuple[str, str, str, str]]:
    values = read_env_file(path)
    out = []
    i = 1
    while f"TRANSCRIPT_{i}_USERNAME" in values:
        out.append(tuple(values[f"TRANSCRIPT_{i}_{k}"] for k in ("USERNAME", "NAME", "LABEL", "DESCRIPTION")))
        i += 1
    return out  # type: ignore[return-value]


def settings(values: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in values.items() if not k.startswith("TRANSCRIPT_")}


# ── matching tracks to speakers ──


def test_known_speakers_get_their_rows_dm_first_and_settings_copied(tmp_path: Path, base: Path) -> None:
    b = make_bundle(tmp_path / "b", [BOB_T, ALICE_T, CAROL_T])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, BOB, CAROL), base=base)
    assert result.status == "written" and result.path == b.parent / ENV
    assert rows(result.path) == [
        ("1-alice_dm", "Pat", "DM", "Dungeon Master"),
        ("2-bob", "Robin", "Milo \"Maestro\" Fairweather", "A bard"),
        ("3-carol", "Sam", "Orsk", "A fighter"),
    ]
    assert settings(read_env_file(result.path)) == settings(read_env_file(base))
    assert result.provisional == [] and result.absent == []
    assert [s["track_id"] for s in result.as_json()["speakers"]] == ["1", "2", "3"]


def test_speaker_key_wins_over_a_matching_username(tmp_path: Path) -> None:
    """Same username, different key: a different person, so the track stays provisional."""
    impostor = {**BOB, "speaker_key": "discord:999"}
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, impostor))
    assert result.provisional == ["2"]
    assert result.absent == [{"speaker_key": "discord:999", "username": "bob"}]


def test_username_matches_when_a_side_has_no_key(tmp_path: Path) -> None:
    keyless = {k: v for k, v in BOB.items() if k != "speaker_key"}
    b = make_bundle(tmp_path / "b", [ALICE_T, track(2, "bob", 40, key=None)])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, keyless))
    assert rows(result.path)[1][1:3] == ("Robin", "Milo \"Maestro\" Fairweather")


def test_an_entry_matching_two_tracks_is_refused(tmp_path: Path) -> None:
    keyless = {k: v for k, v in BOB.items() if k != "speaker_key"}
    b = make_bundle(tmp_path / "b", [ALICE_T, track(2, "bob", 40), track(3, "bob", 30, file="3-bob2.flac")])
    with pytest.raises(SessionEnvError, match="2-bob.flac.*3-bob2.flac"):
        build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, keyless))


def test_a_track_matching_two_entries_is_refused(tmp_path: Path) -> None:
    by_name = {k: v for k, v in BOB.items() if k != "speaker_key"}
    b = make_bundle(tmp_path / "b", [ALICE_T, track(2, "bob", 40, key=None)])
    with pytest.raises(SessionEnvError, match="2-bob.flac"):
        build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, by_name, {**by_name, "label": "Orsk"}))


def test_unmatched_tracks_get_provisional_rows(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T, track(3, None, 30, key=None)])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE))
    assert rows(result.path)[1:] == [
        ("2-bob", "bob", "bob", "PROVISIONAL: character not set"),
        ("3-track", "3", "3", "PROVISIONAL: character not set; may be several people"),
    ]
    assert result.provisional == ["2", "3"]
    speakers = result.as_json()["speakers"]
    assert speakers[1]["provisional"] == ["name", "label", "description"]


def test_entries_with_no_track_are_absent_not_an_error(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, BOB, CAROL))
    assert result.absent == [{"speaker_key": CAROL["speaker_key"], "username": "carol"}]


# ── who is DM ──


def test_without_a_dm_the_largest_track_is_a_provisional_guess(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [track(1, "alice.dm", 30), BOB_T, CAROL_T])
    result = build(b)
    assert rows(result.path)[0][0] == "2-bob" and rows(result.path)[0][2] == "DM"
    assert "dm" in result.as_json()["speakers"][0]["provisional"]
    assert any("largest" in w for w in result.warnings)


def test_dm_flag_picks_the_track(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T, CAROL_T])
    no_dm = {k: v for k, v in ALICE.items() if k != "dm"} | {"label": "Ranger"}
    result = build(b, speakers=speakers_file(tmp_path / "s.json", no_dm, BOB, CAROL), dm="3")
    assert rows(result.path)[0] == ("3-carol", "Sam", "DM", "Dungeon Master")
    assert result.as_json()["speakers"][0]["dm"] is True and result.provisional == []


def test_dm_flag_conflicting_with_the_speakers_file_is_refused(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    with pytest.raises(SessionEnvError, match="--dm 2"):
        build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, BOB), dm="2")


def test_dm_flag_naming_no_track_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SessionEnvError, match="--dm 9"):
        build(make_bundle(tmp_path / "b", [ALICE_T]), dm="9")


def test_two_present_dms_are_refused(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    with pytest.raises(SessionEnvError, match="dm"):
        build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, {**BOB, "dm": True}))


def test_an_absent_dm_does_not_count(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [BOB_T, CAROL_T])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, BOB, CAROL))
    assert rows(result.path)[0][0] == "2-bob"  # largest present track, guessed
    assert "dm" in result.as_json()["speakers"][0]["provisional"]


# ── the speakers file ──


@pytest.mark.parametrize("entry,match", [
    ({"name": "Pat", "label": "DM", "description": "x"}, "speaker_key or username"),
    ({**BOB, "name": ""}, "name"),
    ({**BOB, "label": "x" * 201}, "label"),
    ({**BOB, "description": "two\nlines"}, "description"),
    ({**BOB, "description": "Costs ${HOME}"}, r"\$\{"),
    ({**BOB, "dm": "yes"}, "dm"),
])
def test_bad_speaker_entries_are_refused(tmp_path: Path, entry: dict[str, Any], match: str) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    with pytest.raises(SessionEnvError, match=match):
        build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, entry))
    assert not (b.parent / ENV).exists()


def test_a_speakers_file_that_is_not_one_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text('{"people": []}')
    with pytest.raises(SessionEnvError, match="speakers"):
        build(make_bundle(tmp_path / "b", [ALICE_T]), speakers=path)


def test_unknown_speaker_fields_are_ignored(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", {**ALICE, "pronouns": "they"}))
    assert result.provisional == []


# ── bundle and stem checks ──


def test_a_bad_bundle_is_refused_with_its_reason(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    with pytest.raises(SessionEnvError, match="incomplete: no manifest.json"):
        build(tmp_path / "b")


def test_stems_that_contain_one_another_are_refused(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [track(1, "bob", 40, file="bob.flac"), track(2, "bob2", 30, file="bob-2.flac")])
    with pytest.raises(SessionEnvError, match="bob.flac.*bob-2.flac|bob-2.flac.*bob.flac"):
        build(b)
    assert not (b.parent / ENV).exists()


# ── what it writes ──


def test_session_date_is_local_to_the_zone(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T])
    assert build(b, tz="America/New_York").session_date == "2026-10-18"
    assert build(b, tz="Europe/Berlin", force=True).session_date == "2026-10-19"


def test_no_start_time_means_no_date(tmp_path: Path) -> None:
    result = build(make_bundle(tmp_path / "b", [ALICE_T], started_at=None))
    assert result.session_date is None and result.started_at is None


def test_an_unknown_zone_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SessionEnvError, match="Mars/Olympus"):
        build(make_bundle(tmp_path / "b", [ALICE_T]), tz="Mars/Olympus")


def test_default_base_is_the_repo_example_without_its_speakers(tmp_path: Path) -> None:
    result = build(make_bundle(tmp_path / "b", [ALICE_T, BOB_T]))
    example = read_env_file(ROOT_DIR / ".env.example")
    assert settings(read_env_file(result.path)) == settings(example)
    assert [r[0] for r in rows(result.path)] == ["1-alice_dm", "2-bob"]


def test_existing_output_is_kept_unless_forced(tmp_path: Path, base: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    s = speakers_file(tmp_path / "s.json", ALICE)
    first = build(b, speakers=s, base=base)
    edited = first.path.read_text().replace('TRANSCRIPT_2_LABEL="bob"', 'TRANSCRIPT_2_LABEL="Orsk"')
    first.path.write_text(edited)
    kept = build(b, speakers=s, base=base)
    assert kept.status == "kept" and first.path.read_text() == edited
    assert kept.provisional == ["2"]  # description still says PROVISIONAL
    assert kept.as_json()["speakers"][1]["label"] == "Orsk"
    forced = build(b, speakers=s, base=base, force=True)
    assert forced.status == "written" and rows(forced.path)[1][2] == "bob"


def test_existing_output_that_does_not_fit_is_refused(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    (b.parent / ENV).write_text('TRANSCRIPT_1_USERNAME="1-alice_dm"\nTRANSCRIPT_1_NAME="Pat"\n'
                                 'TRANSCRIPT_1_LABEL="DM"\nTRANSCRIPT_1_DESCRIPTION="Dungeon Master"\n')
    with pytest.raises(SessionEnvError, match="--force"):
        build(b)


def test_the_bundle_is_never_written_to(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T])
    before = sorted(p.name for p in b.iterdir())
    out = tmp_path / "envs" / "game.env"
    assert build(b, output=out).path == out and out.is_file()
    assert sorted(p.name for p in b.iterdir()) == before


@pytest.mark.parametrize("inside", ["session.env", "sub/session.env"])
def test_output_inside_the_bundle_is_refused(tmp_path: Path, inside: str) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T])
    with pytest.raises(SessionEnvError, match="inside the bundle"):
        build(b, output=b / inside)
    assert not (b / inside).exists()


def test_the_written_file_passes_runs_own_check(tmp_path: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T, track(3, None, 30, key=None)])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, BOB))
    validate_speaker_mapping(["1-alice_dm", "2-bob", "3-track"], env=read_env_file(result.path))


@pytest.mark.parametrize("value", [
    "Plain", 'Has "double" quotes', "Costs $5 and $HOME", r"back\slash", "it's", "# not a comment",
    "Ünïcode ✓", 'mix "of" $all \\ \'kinds\'', "Human Ranger, Beast Master; companion Pip (aka Abe)",
])
def test_values_roundtrip_through_the_env_parser(tmp_path: Path, value: str) -> None:
    from dotenv import dotenv_values

    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    result = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, {**BOB, "description": value}))
    assert rows(result.path)[1][3] == value
    # what `combobulator run` reads: config.load_dotenv expands ${VAR}
    assert dotenv_values(result.path)["TRANSCRIPT_2_DESCRIPTION"] == value


@pytest.mark.parametrize("value", ["${HOME}", "Costs ${X:-5}", "a${}b"])
def test_variable_reference_in_a_field_is_refused(value: str) -> None:
    """The run-time loader expands ${VAR}, even in single quotes: a name could copy a secret into the transcript."""
    with pytest.raises(SessionEnvError, match=r"\$\{"):
        _quote(value)


# ── the command ──


def test_cli_prints_the_json_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    main(["--bundle", str(b), "--output", str(tmp_path / ENV), "--speakers", str(speakers_file(tmp_path / "s.json", ALICE)), "--tz", "UTC", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"path", "status", "recording_id", "started_at", "session_date", "speakers",
                            "provisional", "absent", "warnings"}
    assert payload["recording_id"] == "AbCdEf12345" and payload["session_date"] == "2026-10-19"
    assert payload["provisional"] == ["2"]
    assert set(payload["speakers"][0]) == {"track_id", "file", "speaker_key", "username", "name", "label",
                                           "dm", "provisional"}


def test_cli_refusal_exits_1_with_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--bundle", str(tmp_path / "missing"), "--output", str(tmp_path / ENV)])
    err = capsys.readouterr().err
    assert e.value.code == 1 and "not a folder" in err and err.count("\n") == 1


def test_cli_usage_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--bundle", str(make_bundle(tmp_path / "b", [ALICE_T]))])
    assert e.value.code == 2 and "--output" in capsys.readouterr().err


# ── what a later edit costs: the resume fingerprints ──


def stage_settings(env_file: Path) -> dict[str, Any]:
    """get_stage_fingerprint_settings() as a fresh process sees it under this ENV_FILE."""
    env = {**os.environ, "ENV_FILE": str(env_file), "PYTHONPATH": str(ROOT_DIR)}
    out = subprocess.run(
        [sys.executable, "-c",
         "import json; from transcript_combobulator.config import get_stage_fingerprint_settings as g;"
         "print(json.dumps(g(), sort_keys=True, default=str))"],
        env=env, capture_output=True, text=True, check=True, cwd=ROOT_DIR,
    )
    parsed: dict[str, Any] = json.loads(out.stdout)
    return parsed


def test_generated_env_has_the_base_fingerprints(tmp_path: Path, base: Path) -> None:
    result = build(make_bundle(tmp_path / "b", [ALICE_T, BOB_T]), base=base)
    assert stage_settings(result.path) == stage_settings(base)


def test_names_labels_and_descriptions_are_in_no_fingerprint(tmp_path: Path, base: Path) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T, CAROL_T])
    out = build(b, speakers=speakers_file(tmp_path / "s.json", ALICE, BOB, CAROL), base=base).path
    before = stage_settings(out)
    out.write_text(out.read_text().replace('"Orsk"', '"Someone Else"').replace('"A bard"', '"A wizard"'))
    assert stage_settings(out) == before


def test_whisper_prompt_is_in_the_inference_fingerprint_only(tmp_path: Path, base: Path) -> None:
    out = build(make_bundle(tmp_path / "b", [ALICE_T, BOB_T]), base=base).path
    before = stage_settings(out)
    out.write_text(out.read_text().replace("A Dungeons & Dragons session.", "A session with Orsk."))
    after = stage_settings(out)
    assert [stage for stage in before if before[stage] != after[stage]] == ["inference"]


def test_usernames_are_checked_before_transcription(tmp_path: Path) -> None:
    out = build(make_bundle(tmp_path / "b", [ALICE_T, BOB_T])).path
    values = read_env_file(out)
    values["TRANSCRIPT_2_USERNAME"] = "robert"
    with pytest.raises(CombineError, match="No mapping found"):
        validate_speaker_mapping(["1-alice_dm", "2-bob"], env=values)


# ── relabel after the fact: combine only, no inference ──


def write_vtt(path: Path, lines: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "WEBVTT\n\n" + "\n".join(f"{when} --> {when[:-1]}9\n{text}\n" for when, text in lines)
    path.write_text(body)


def test_labels_can_be_fixed_after_transcription_by_recombining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = make_bundle(tmp_path / "b", [ALICE_T, BOB_T])
    s = tmp_path / "s.json"
    out = build(b, speakers=speakers_file(s, ALICE)).path
    session = tmp_path / "output" / "game"
    write_vtt(session / "1-alice_dm" / "1-alice_dm.vtt", [("00:00:01.000", "The door creaks open.")])
    write_vtt(session / "2-bob" / "2-bob.vtt", [("00:00:03.000", "I check for traps.")])

    def combine() -> str:
        for key, value in read_env_file(out).items():
            monkeypatch.setenv(key, value)
        (path,) = combine_transcripts_from_env(tmp_path / "output", "game")
        return path.read_text()

    first = combine()
    assert "bob: I check for traps." in first and "bob - bob - PROVISIONAL" in first

    build(b, speakers=speakers_file(s, ALICE, BOB), force=True)
    second = combine()
    assert "Milo \"Maestro\" Fairweather: I check for traps." in second
    assert "Robin - Milo \"Maestro\" Fairweather - A bard" in second and "PROVISIONAL" not in second
    assert "DM: The door creaks open." in second
