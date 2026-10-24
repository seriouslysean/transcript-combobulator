"""Compare two combined transcripts of one session, per speaker: a reference run and a new run.

Reads the files ``combine`` writes (``<session>-combined.txt``, or its ``-1``/``-2``
chunks, in order) and reports numbers only, no inference:

- words different: 1 - difflib ratio over whisper-normalised words, per speaker and pooled
- lost spans: runs of 4+ reference words the new run (almost) lacks, as a share of reference
  lines; extra spans are the same the other way round. The combined files carry no
  timestamps per word, so this approximates "speech segments lost"
- artefacts per side: lines echoing 4 words of ``WHISPER_PROMPT`` (from ENV_FILE), bare canned
  lines ("Thank you."), lines of 100+ words, and lines repeating a 4-word run 5+ times (loop
  remnants: whisper's loops are collapsed before the VTT, so raw loops are not visible here)
- how often each cast name from the Summary header appears on each side

The gate is the one a runtime change had to pass on 2026-10-16: pooled words different at most
1.5%, lost spans at most 0.5% of reference lines, and no lost span over 10 words.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIN_RUN = 4
GATE_WORDS_DIFFERENT_PCT = 1.5
GATE_LOST_SPANS_PCT = 0.5
GATE_LONGEST_LOST = 10
CANNED = frozenset(
    {"thank you", "thanks for watching", "thank you for watching", "you", "bye"}
)

# A speaker line when the Summary header is missing: a label of up to 31 characters,
# quotes allowed (Milo "Maestro" Fairweather).
_SPEAKER_RE = re.compile(r'^([A-Za-z][A-Za-z .\'"“”-]{0,30}?):\s(.*)$')
# combine's INCLUDE_TIMESTAMPS suffix (`` [00:00:15.000 --> 00:00:25.380]``), so both modes
# compare the same words.
_TIMESTAMP_RE = re.compile(
    r"\s\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]$"
)


def _seconds(clock: str) -> float:
    h, m, sec = clock.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def normalize(text: str) -> str:
    """Whisper's English normaliser: the same one its own WER numbers use."""
    from whisper.normalizers import EnglishTextNormalizer

    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = EnglishTextNormalizer()
    return str(_NORMALIZER(text)).strip()


_NORMALIZER: Any = None


@dataclass
class Transcript:
    """Speaker lines in file order, and the cast's character labels (not the DM)."""

    lines: dict[str, list[str]] = field(default_factory=dict)
    cast: list[str] = field(default_factory=list)
    # Each line's (start, end) seconds when it carries combine's timestamp suffix.
    times: dict[str, list[tuple[float, float] | None]] = field(default_factory=dict)


def load(paths: Iterable[Path]) -> Transcript:
    """Read combined transcript files, in order, into one Transcript."""
    out = Transcript()
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        labels: list[str] = []
        i = 0
        if lines and lines[0].strip() == "Summary:":
            i = 1
            while i < len(lines) and lines[i].strip():
                parts = [p.strip() for p in lines[i].split(" - ", 2)]
                if len(parts) == 3:
                    labels.append(parts[1])
                    if parts[1] != "DM" and parts[1] not in out.cast:
                        out.cast.append(parts[1])
                i += 1
        # Longest label first, so "Jo Jo" is not read as "Jo" plus "Jo: ...".
        prefixes = sorted(labels, key=len, reverse=True)
        for raw in lines[i:]:
            line = raw.strip()
            speaker = next(
                (lab for lab in prefixes if line.startswith(f"{lab}: ")), None
            )
            if speaker is not None:
                text = line[len(speaker) + 2 :]
            elif not prefixes and (m := _SPEAKER_RE.match(line)):
                speaker, text = m.group(1), m.group(2)
            else:
                continue
            ts = _TIMESTAMP_RE.search(text)
            out.lines.setdefault(speaker, []).append(_TIMESTAMP_RE.sub("", text))
            out.times.setdefault(speaker, []).append(
                (_seconds(ts.group(1)), _seconds(ts.group(2))) if ts else None
            )
    return out


def words(lines: Iterable[str]) -> list[str]:
    return [w for t in lines for w in normalize(t).split()]


def spans(ref: Sequence[str], hyp: Sequence[str]) -> tuple[list[int], list[int], float]:
    """Lengths of runs in ref that hyp (almost) lacks, runs in hyp that ref lacks, and the ratio."""
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    lost: list[int] = []
    extra: list[int] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if (
            tag in ("delete", "replace")
            and i2 - i1 >= MIN_RUN
            and j2 - j1 <= 0.25 * (i2 - i1)
        ):
            lost.append(i2 - i1)
        if (
            tag in ("insert", "replace")
            and j2 - j1 >= MIN_RUN
            and i2 - i1 <= 0.25 * (j2 - j1)
        ):
            extra.append(j2 - j1)
    return lost, extra, sm.ratio()


def prompt_grams(prompt: str) -> set[tuple[str, ...]]:
    w = normalize(prompt).split() if prompt else []
    return {tuple(w[i : i + MIN_RUN]) for i in range(len(w) - MIN_RUN + 1)}


def artifacts(lines: Iterable[str], grams: set[tuple[str, ...]]) -> dict[str, int]:
    counts = dict.fromkeys(
        (
            "echo_lines",
            "bare_canned_lines",
            "lines_100_words_plus",
            "loop_remnant_lines",
        ),
        0,
    )
    for text in lines:
        norm = normalize(text)
        w = norm.split()
        runs = [tuple(w[i : i + MIN_RUN]) for i in range(len(w) - MIN_RUN + 1)]
        counts["echo_lines"] += any(r in grams for r in runs)
        counts["bare_canned_lines"] += norm in CANNED
        counts["lines_100_words_plus"] += len(w) >= 100
        counts["loop_remnant_lines"] += (
            len(w) >= 20 and max(Counter(runs).values()) >= 5
        )
    return counts


def compare(ref: Transcript, hyp: Transcript, prompt: str = "") -> dict[str, Any]:
    """The full report as a JSON-ready dict."""
    per: dict[str, dict[str, Any]] = {}
    tot: Counter[str] = Counter()
    all_ref: list[str] = []
    all_hyp: list[str] = []
    for sp in sorted(ref.lines.keys() | hyp.lines.keys()):
        rl, hl = ref.lines.get(sp, []), hyp.lines.get(sp, [])
        rw, hw = words(rl), words(hl)
        lost, extra, ratio = spans(rw, hw)
        per[sp] = {
            "ref_words": len(rw),
            "hyp_words": len(hw),
            "ref_lines": len(rl),
            "hyp_lines": len(hl),
            "words_different_pct": round(100 * (1 - ratio), 2),
            "lost_spans": len(lost),
            "lost_words": sum(lost),
            "lost_longest": max(lost, default=0),
            "extra_spans": len(extra),
            "extra_words": sum(extra),
        }
        all_ref += rw
        all_hyp += hw
        for k in (
            "ref_words",
            "hyp_words",
            "ref_lines",
            "hyp_lines",
            "lost_spans",
            "lost_words",
            "extra_spans",
            "extra_words",
        ):
            tot[k] += per[sp][k]
        tot["lost_longest"] = max(tot["lost_longest"], per[sp]["lost_longest"])
    pooled = 100 * (
        1 - difflib.SequenceMatcher(None, all_ref, all_hyp, autojunk=False).ratio()
    )
    lost_pct = 100 * tot["lost_spans"] / max(1, tot["ref_lines"])
    grams = prompt_grams(prompt)

    def name_count(t: Transcript, name: str) -> int:
        pat = re.compile(r"\b" + re.escape(name) + r"\b", re.I)
        return sum(len(pat.findall(line)) for ls in t.lines.values() for line in ls)

    return {
        "pooled_words_different_pct": round(pooled, 2),
        "lost_spans_pct_of_ref_lines": round(lost_pct, 2),
        "totals": dict(tot),
        "per_speaker": per,
        "ref_artifacts": artifacts((t for ls in ref.lines.values() for t in ls), grams),
        "hyp_artifacts": artifacts((t for ls in hyp.lines.values() for t in ls), grams),
        "names_ref_hyp": {
            n: [name_count(ref, n), name_count(hyp, n)] for n in ref.cast
        },
        "gate": {
            f"words_different_le_{GATE_WORDS_DIFFERENT_PCT}": pooled
            <= GATE_WORDS_DIFFERENT_PCT,
            f"lost_spans_le_{GATE_LOST_SPANS_PCT}pct": lost_pct <= GATE_LOST_SPANS_PCT,
            f"lost_none_over_{GATE_LONGEST_LOST}_words": tot["lost_longest"]
            <= GATE_LONGEST_LOST,
        },
    }


def accuracy(ref: Transcript, hyp: Transcript, slack: float = 1.0) -> dict[str, Any]:
    """Word error rate against a human reference, per speaker and pooled.

    Substitutions, deletions and insertions come from jiwer's alignment over whisper-normalised
    words. ``other_speaker_words`` counts insertions that another speaker's reference line says
    within ``slack`` seconds of the hypothesis line: a sign of speech picked up from someone
    else's microphone. It needs timestamps on both sides (``None`` without them), and is only
    a sign: a common word can match by chance.
    """
    import jiwer

    def heard(sp: str, t: tuple[float, float] | None) -> set[str]:
        if t is None:
            return set()
        near: set[str] = set()
        for other, lines in ref.lines.items():
            if other == sp:
                continue
            for text, rt in zip(lines, ref.times.get(other, []), strict=True):
                if rt and rt[0] - slack <= t[1] and t[0] <= rt[1] + slack:
                    near.update(normalize(text).split())
        return near

    timed = any(t for v in ref.times.values() for t in v) and any(
        t for v in hyp.times.values() for t in v
    )
    per: dict[str, dict[str, Any]] = {}
    tot: Counter[str] = Counter()
    for sp in sorted(ref.lines.keys() | hyp.lines.keys()):
        rw = words(ref.lines.get(sp, []))
        hyp_lines = hyp.lines.get(sp, [])
        hyp_times = hyp.times.get(sp, [None] * len(hyp_lines))
        hw: list[str] = []
        line_of: list[int] = []
        for n, text in enumerate(hyp_lines):
            w = normalize(text).split()
            hw += w
            line_of += [n] * len(w)
        c = Counter(hits=0, substitutions=0, deletions=0, insertions=0)
        other = 0
        if rw and hw:
            out = jiwer.process_words(" ".join(rw), " ".join(hw))
            c.update(
                hits=out.hits,
                substitutions=out.substitutions,
                deletions=out.deletions,
                insertions=out.insertions,
            )
            for chunk in out.alignments[0]:
                if chunk.type == "insert":
                    for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        other += hw[j] in heard(sp, hyp_times[line_of[j]])
        else:
            c.update(deletions=len(rw), insertions=len(hw))
        errors = c["substitutions"] + c["deletions"] + c["insertions"]
        per[sp] = {
            "ref_words": len(rw),
            "hyp_words": len(hw),
            **c,
            "other_speaker_words": other if timed else None,
            "wer_pct": round(100 * errors / max(1, len(rw)), 2),
        }
        tot.update(
            {k: v for k, v in per[sp].items() if k != "wer_pct" and v is not None}
        )
    errors = tot["substitutions"] + tot["deletions"] + tot["insertions"]
    return {
        "wer_pct": round(100 * errors / max(1, tot["ref_words"]), 2),
        "wer_pct_without_other_speaker_words": (
            round(
                100 * (errors - tot["other_speaker_words"]) / max(1, tot["ref_words"]),
                2,
            )
            if timed
            else None
        ),
        "totals": {
            **tot,
            "other_speaker_words": tot["other_speaker_words"] if timed else None,
        },
        "per_speaker": per,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="combobulator compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ref",
        nargs="+",
        type=Path,
        required=True,
        help="Reference combined files, in order",
    )
    parser.add_argument(
        "--hyp",
        nargs="+",
        type=Path,
        required=True,
        help="New combined files, in order",
    )
    parser.add_argument("--json", type=Path, help="Also write the full report here")
    parser.add_argument(
        "--strict", action="store_true", help="Exit 1 when any gate fails"
    )
    parser.add_argument(
        "--wer",
        action="store_true",
        help="Score against a human reference instead: word error rate, no gate",
    )
    args = parser.parse_args(argv)
    if args.wer and args.strict:
        parser.error("--wer reports accuracy and has no gate; drop --strict")
    for p in [*args.ref, *args.hyp]:
        if not p.is_file():
            print(f"Error: not a file: {p}", file=sys.stderr)
            sys.exit(2)

    from transcript_combobulator.config import WHISPER_PROMPT

    ref, hyp = load(args.ref), load(args.hyp)
    for name, t in (("reference", ref), ("hypothesis", hyp)):
        if not any(words(v) for v in t.lines.values()):
            print(f"Error: no spoken words in the {name}", file=sys.stderr)
            sys.exit(2)
    if args.wer:
        acc = accuracy(ref, hyp)
        if args.json:
            args.json.write_text(json.dumps(acc, indent=1), encoding="utf-8")
        without = acc["wer_pct_without_other_speaker_words"]
        print(
            f"WER against the human reference: {acc['wer_pct']}% "
            + (
                f"({without}% without words another speaker said)"
                if without is not None
                else "(words another speaker said not counted: no timestamps)"
            )
        )
        for sp, v in acc["per_speaker"].items():
            print(
                f"{sp:10} ref {v['ref_words']:6}  wer {v['wer_pct']:6}%  sub {v['substitutions']}  "
                f"del {v['deletions']}  ins {v['insertions']} (other speakers {'n/a' if v['other_speaker_words'] is None else v['other_speaker_words']})"
            )
        return
    report = compare(ref, hyp, WHISPER_PROMPT)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_speaker"}, indent=1))
    for sp, v in report["per_speaker"].items():
        print(
            f"{sp:10} words {v['ref_words']:6} vs {v['hyp_words']:6}  "
            f"different {v['words_different_pct']:5}%  lost {v['lost_spans']} "
            f"({v['lost_words']} words, longest {v['lost_longest']})  extra {v['extra_spans']}"
        )
    if args.strict and not all(report["gate"].values()):
        sys.exit(1)
