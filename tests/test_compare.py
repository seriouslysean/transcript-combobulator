"""Comparing two combined transcripts: pure Python over synthetic files."""

import json
from pathlib import Path

import pytest

from transcript_combobulator.compare import compare, load, main

SUMMARY = (
    "Summary:\n"
    "Pat - DM - Dungeon master\n"
    'Robin - Milo "Maestro" Fairweather - A bard - plays the lute\n'
    "Sam - Orsk - A fighter\n\n"
)
LINES = [
    "DM: You reach the gate at the end of the long road.",
    'Milo "Maestro" Fairweather: I play a quiet song for the guards.',
    "Orsk: I check the gate for traps before anyone touches it.",
    "DM: The guards wave you through and Orsk finds nothing.",
]


def _write(path: Path, lines: list[str], summary: str = SUMMARY) -> Path:
    path.write_text(summary + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_identical_runs_differ_by_nothing_and_pass_the_gate(tmp_path: Path) -> None:
    ref = load([_write(tmp_path / "a.txt", LINES)])
    report = compare(ref, load([_write(tmp_path / "b.txt", LINES)]))
    assert report["pooled_words_different_pct"] == 0.0
    assert report["totals"]["lost_spans"] == 0
    assert all(report["gate"].values())


def test_quoted_labels_and_timestamps_parse(tmp_path: Path) -> None:
    stamped = [
        f"{line} [00:0{i}:00.000 --> 00:0{i}:04.500]" for i, line in enumerate(LINES)
    ]  # combine's format
    t = load([_write(tmp_path / "a.txt", stamped)])
    assert set(t.lines) == {"DM", 'Milo "Maestro" Fairweather', "Orsk"}
    assert t.lines["Orsk"] == ["I check the gate for traps before anyone touches it."]
    assert t.cast == ['Milo "Maestro" Fairweather', "Orsk"]
    plain = load([_write(tmp_path / "b.txt", LINES)])
    assert compare(plain, t)["pooled_words_different_pct"] == 0.0


def test_chunks_are_read_in_order(tmp_path: Path) -> None:
    t = load(
        [
            _write(tmp_path / "s-1.txt", LINES[:2]),
            _write(tmp_path / "s-2.txt", LINES[2:]),
        ]
    )
    assert t.lines["DM"][-1] == "The guards wave you through and Orsk finds nothing."


def test_a_missing_line_is_a_lost_span_and_fails_the_gate(tmp_path: Path) -> None:
    ref = load([_write(tmp_path / "a.txt", LINES)])
    hyp = load([_write(tmp_path / "b.txt", [LINES[0], LINES[1], LINES[3]])])
    report = compare(ref, hyp)
    orsk = report["per_speaker"]["Orsk"]
    assert (orsk["lost_spans"], orsk["lost_words"]) == (1, 10)
    assert report["totals"]["extra_spans"] == 0
    assert not all(report["gate"].values())


def test_an_added_line_is_an_extra_span(tmp_path: Path) -> None:
    ref = load([_write(tmp_path / "a.txt", LINES)])
    hyp = load(
        [
            _write(
                tmp_path / "b.txt",
                [*LINES, "Orsk: Then I walk through the gate with my shield up."],
            )
        ]
    )
    report = compare(ref, hyp)
    assert report["per_speaker"]["Orsk"]["extra_spans"] == 1
    assert report["totals"]["lost_spans"] == 0


def test_artifacts_count_echoes_canned_lines_and_loops(tmp_path: Path) -> None:
    loop = "Orsk: " + " ".join(["we go on and on"] * 6)
    hyp = load(
        [
            _write(
                tmp_path / "b.txt",
                [*LINES, "DM: Thank you.", loop, "DM: The long road to the gate."],
            )
        ]
    )
    ref = load([_write(tmp_path / "a.txt", LINES)])
    art = compare(ref, hyp, prompt="A campaign on the long road to the gate.")[
        "hyp_artifacts"
    ]
    assert art["bare_canned_lines"] == 1
    assert art["loop_remnant_lines"] == 1
    assert (
        art["echo_lines"] == 1
    )  # only the last DM line repeats 4 prompt words in a row


def test_names_are_counted_on_each_side(tmp_path: Path) -> None:
    ref = load([_write(tmp_path / "a.txt", LINES)])
    hyp = load(
        [
            _write(
                tmp_path / "b.txt",
                [line.replace("Orsk finds", "Ork finds") for line in LINES],
            )
        ]
    )
    assert compare(ref, hyp)["names_ref_hyp"]["Orsk"] == [1, 0]


def test_without_a_summary_lines_fall_back_to_the_label_pattern(tmp_path: Path) -> None:
    t = load([_write(tmp_path / "a.txt", LINES, summary="")])
    assert set(t.lines) == {"DM", 'Milo "Maestro" Fairweather', "Orsk"}


def test_main_writes_json_and_strict_exits_on_a_failed_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _write(tmp_path / "a.txt", LINES), _write(tmp_path / "b.txt", LINES[:2])
    out = tmp_path / "r.json"
    main(["--ref", str(a), "--hyp", str(b), "--json", str(out)])
    assert json.loads(out.read_text())["totals"]["lost_spans"] == 2
    with pytest.raises(SystemExit) as e:
        main(["--ref", str(a), "--hyp", str(b), "--strict"])
    assert e.value.code == 1
    capsys.readouterr()


def test_main_refuses_a_missing_file(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.txt", LINES)
    with pytest.raises(SystemExit) as e:
        main(["--ref", str(a), "--hyp", str(tmp_path / "nope.txt")])
    assert e.value.code == 2


@pytest.mark.parametrize("empty", ["ref", "hyp"])
@pytest.mark.parametrize("text", ["not a transcript\n", "Bob: .\n"])
def test_main_refuses_a_file_with_no_spoken_words(
    empty: str, text: str, tmp_path: Path
) -> None:
    a = _write(tmp_path / "a.txt", LINES)
    blank = tmp_path / "blank.txt"
    blank.write_text(text, encoding="utf-8")
    ref, hyp = (blank, a) if empty == "ref" else (a, blank)
    with pytest.raises(SystemExit) as e:
        main(["--ref", str(ref), "--hyp", str(hyp), "--strict"])
    assert e.value.code == 2


def test_combines_own_output_reads_the_same_with_and_without_timestamps(
    tmp_path: Path,
) -> None:
    from transcript_combobulator.combine import TranscriptConfig, combine_transcripts

    configs = []
    for i, (name, label, text) in enumerate(
        [("Pat", "DM", LINES[0][4:]), ("Sam", "Orsk", LINES[2][6:])]
    ):
        vtt = tmp_path / f"{i}.vtt"
        vtt.write_text(
            f"WEBVTT\n\n00:00:0{i}.000 --> 00:00:0{i}.900\n{text}\n", encoding="utf-8"
        )
        configs.append(TranscriptConfig(name, label, "x", vtt))
    (stamped,) = combine_transcripts(
        configs, tmp_path / "t.txt", include_timestamps=True
    )
    (plain,) = combine_transcripts(
        configs, tmp_path / "p.txt", include_timestamps=False
    )
    t, p = load([stamped]), load([plain])
    assert t.lines == p.lines == {"DM": [LINES[0][4:]], "Orsk": [LINES[2][6:]]}
