#!/usr/bin/env python3
"""Tests for the transcript combination step."""

from pathlib import Path

import pytest

from transcript_combobulator.combine import (
    CombineError,
    validate_speaker_mapping,
    TranscriptConfig,
    _normalize_for_dedup,
    combine_transcripts,
    parse_vtt_file,
    should_skip_content,
)


VTT_SAMPLE = """WEBVTT

00:00:01.000 --> 00:00:02.000
Hello, world!

00:00:02.500 --> 00:00:03.500
Hello, world!

00:00:04.000 --> 00:00:05.000
This is a test.
"""


def _write_vtt(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')


class TestNormalizeForDedup:
    def test_strips_punctuation_and_lowercases(self):
        assert _normalize_for_dedup("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_for_dedup("  a\t b\n c  ") == "a b c"

    def test_empty_returns_empty(self):
        assert _normalize_for_dedup("") == ""


class TestShouldSkipContent:
    def test_literal_match(self):
        assert should_skip_content("text with [BLANK_AUDIO] inside", ["[BLANK_AUDIO]"])

    def test_regex_match(self):
        assert should_skip_content("laughs laughs laughs", ["/(laughs\\s*){2,}/"])

    def test_no_match(self):
        assert not should_skip_content("normal speech", ["[BLANK_AUDIO]"])


class TestParseVttFile:
    def test_preserves_original_text(self, tmp_path):
        """Parsed entries keep the original capitalization and punctuation."""
        vtt = tmp_path / "sample.vtt"
        _write_vtt(vtt, VTT_SAMPLE)
        entries = parse_vtt_file(vtt, "Alice")
        assert entries[0].content == "Hello, world!"
        assert entries[2].content == "This is a test."

    def test_dedup_key_is_normalized(self, tmp_path):
        vtt = tmp_path / "sample.vtt"
        _write_vtt(vtt, VTT_SAMPLE)
        entries = parse_vtt_file(vtt, "Alice")
        assert entries[0].dedup_key == "hello world"
        assert entries[2].dedup_key == "this is a test"

    def test_skip_filter_drops_matching_entries(self, tmp_path):
        vtt = tmp_path / "sample.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n[BLANK_AUDIO]\n\n"
            "00:00:03.000 --> 00:00:04.000\nHello\n",
        )
        entries = parse_vtt_file(vtt, "Alice", skip_filters=["[BLANK_AUDIO]"])
        assert len(entries) == 1
        assert entries[0].content == "Hello"


class TestCombineTranscripts:
    def test_output_preserves_original_text(self, tmp_path):
        """Combined session transcript keeps capitalization and punctuation."""
        vtt = tmp_path / "alice.vtt"
        _write_vtt(vtt, VTT_SAMPLE)
        out = tmp_path / "combined.txt"

        combine_transcripts(
            transcript_configs=[
                TranscriptConfig(
                    name="Alice Jones",
                    label="Alice",
                    description="Party leader",
                    transcript_path=vtt,
                )
            ],
            output_path=out,
        )

        text = out.read_text()
        assert "Alice: Hello, world!" in text
        assert "Alice: This is a test." in text

    def test_consecutive_repeat_within_window_is_deduped(self, tmp_path):
        """Whisper's repeated-line hallucination: same text, back to back."""
        vtt = tmp_path / "alice.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nHello, world!\n\n"
            "00:00:02.500 --> 00:00:03.500\nhello world\n\n",
        )
        out = tmp_path / "combined.txt"
        combine_transcripts(
            transcript_configs=[
                TranscriptConfig(name="Alice", label="Alice", description="", transcript_path=vtt)
            ],
            output_path=out,
        )
        assert out.read_text().count("Alice: ") == 1

    def test_genuine_repeat_later_in_session_is_kept(self, tmp_path):
        """A speaker saying 'Yeah.' ten minutes apart is two lines, not one."""
        vtt = tmp_path / "alice.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nYeah.\n\n"
            "00:10:01.000 --> 00:10:02.000\nYeah.\n\n",
        )
        out = tmp_path / "combined.txt"
        combine_transcripts(
            transcript_configs=[
                TranscriptConfig(name="Alice", label="Alice", description="", transcript_path=vtt)
            ],
            output_path=out,
        )
        assert out.read_text().count("Alice: Yeah.") == 2

    def test_global_strategy_still_available(self, tmp_path):
        from transcript_combobulator.combine import _dedupe_entries, parse_vtt_file

        vtt = tmp_path / "alice.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nYeah.\n\n"
            "00:10:01.000 --> 00:10:02.000\nYeah.\n\n",
        )
        entries = parse_vtt_file(vtt, "Alice")
        assert len(_dedupe_entries(entries, strategy="global")) == 1
        assert len(_dedupe_entries(entries, strategy="none")) == 2
        assert len(_dedupe_entries(entries, strategy="consecutive", window_seconds=2.0)) == 2

    def test_two_speakers_same_text_both_kept(self, tmp_path):
        """Different speakers saying the same thing are not deduped."""
        a = tmp_path / "a.vtt"
        b = tmp_path / "b.vtt"
        _write_vtt(a, "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n")
        _write_vtt(b, "WEBVTT\n\n00:00:01.500 --> 00:00:02.500\nHello\n")
        out = tmp_path / "combined.txt"

        combine_transcripts(
            transcript_configs=[
                TranscriptConfig("A", "A", "", a),
                TranscriptConfig("B", "B", "", b),
            ],
            output_path=out,
        )

        text = out.read_text()
        assert "A: Hello" in text
        assert "B: Hello" in text

    def test_sorted_by_start_time(self, tmp_path):
        a = tmp_path / "a.vtt"
        b = tmp_path / "b.vtt"
        _write_vtt(a, "WEBVTT\n\n00:00:10.000 --> 00:00:11.000\nLater A\n")
        _write_vtt(b, "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nEarly B\n")
        out = tmp_path / "combined.txt"

        combine_transcripts(
            transcript_configs=[
                TranscriptConfig("A", "A", "", a),
                TranscriptConfig("B", "B", "", b),
            ],
            output_path=out,
        )

        text = out.read_text()
        assert text.index("B: Early B") < text.index("A: Later A")


class TestWhisperRepetitionCollapse:
    """Regression test for degenerate whisper output in the transcription path."""

    def test_collapse_repeated_word(self):
        from transcript_combobulator.whisper import collapse_repetition

        raw = "laughs " * 100
        assert collapse_repetition(raw.strip()) == "laughs"

    def test_leaves_normal_text_alone(self):
        from transcript_combobulator.whisper import collapse_repetition

        assert collapse_repetition("the quick brown fox") == "the quick brown fox"

    def test_short_runs_preserved(self):
        from transcript_combobulator.whisper import collapse_repetition

        # Below threshold; keep as-is.
        assert collapse_repetition("no no no") == "no no no"


def _set_mapping(monkeypatch, usernames: list[str]) -> None:
    """Point TRANSCRIPT_N_* at the given usernames and terminate the sequence."""
    for i, username in enumerate(usernames, start=1):
        monkeypatch.setenv(f"TRANSCRIPT_{i}_USERNAME", username)
        monkeypatch.setenv(f"TRANSCRIPT_{i}_NAME", f"Name {i}")
        monkeypatch.setenv(f"TRANSCRIPT_{i}_LABEL", f"Label {i}")
        monkeypatch.setenv(f"TRANSCRIPT_{i}_DESCRIPTION", f"Desc {i}")
    monkeypatch.setenv(f"TRANSCRIPT_{len(usernames) + 1}_USERNAME", "")


class TestValidateSpeakerMapping:
    """Up-front TRANSCRIPT_N_* validation shared with the combine step."""

    def test_every_dir_mapped_once_returns_mapping(self, monkeypatch):
        _set_mapping(monkeypatch, ["nilbits", "dezfrost"])
        mapping = validate_speaker_mapping(["1-nilbits", "2-dezfrost"])
        assert set(mapping) == {"nilbits", "dezfrost"}

    def test_no_mapping_configured(self, monkeypatch):
        _set_mapping(monkeypatch, [])
        with pytest.raises(CombineError, match="No transcript mappings found"):
            validate_speaker_mapping(["1-nilbits"])

    def test_unmapped_dir_names_all_reported(self, monkeypatch):
        _set_mapping(monkeypatch, ["nilbits"])
        with pytest.raises(CombineError, match="2-dezfrost.*3-hereticjd"):
            validate_speaker_mapping(["1-nilbits", "2-dezfrost", "3-hereticjd"])

    def test_ambiguous_token_match(self, monkeypatch):
        _set_mapping(monkeypatch, ["dez", "frost"])
        with pytest.raises(CombineError, match="Ambiguous username match"):
            validate_speaker_mapping(["2-dez-frost"])

    def test_username_must_match_as_whole_token(self, monkeypatch):
        """'dez' must not silently claim dezfrost's lines."""
        _set_mapping(monkeypatch, ["dez"])
        with pytest.raises(CombineError, match="No mapping found"):
            validate_speaker_mapping(["5-dezfrost"])

    def test_token_match_accepts_craig_and_converted_dir_names(self, monkeypatch):
        _set_mapping(monkeypatch, ["nilbits", "burger_bear"])
        mapping = validate_speaker_mapping(["3-nilbits", "3-nilbits_16khz", "2-burger_bear"])
        assert set(mapping) == {"nilbits", "burger_bear"}
        with pytest.raises(CombineError, match="No mapping found"):
            validate_speaker_mapping(["3-nilbits2"])

    def test_incomplete_entry_is_skipped_and_leaves_dir_unmapped(self, monkeypatch):
        _set_mapping(monkeypatch, ["nilbits"])
        monkeypatch.setenv("TRANSCRIPT_1_DESCRIPTION", "")
        with pytest.raises(CombineError, match="No transcript mappings found"):
            validate_speaker_mapping(["1-nilbits"])


class TestCombineFromEnvExplicitFiles:
    """Batch runs hand combine the exact VTTs they produced."""

    def _session(self, tmp_path, monkeypatch):
        _set_mapping(monkeypatch, ["nilbits"])
        session = tmp_path / "night"
        current = session / "3-nilbits"
        stale = session / "3-nilbits-v0"
        for d in (current, stale):
            d.mkdir(parents=True)
            _write_vtt(
                d / "3-nilbits.vtt",
                "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nRoll for initiative.\n\n",
            )
        monkeypatch.setenv("CHUNKS", "1")
        return session, current, stale

    def test_explicit_list_ignores_stale_sibling_dirs(self, tmp_path, monkeypatch):
        from transcript_combobulator.combine import combine_transcripts_from_env

        session, current, _ = self._session(tmp_path, monkeypatch)
        out = combine_transcripts_from_env(
            tmp_path, "night", vtt_files=[current / "3-nilbits.vtt"]
        )
        assert out[0].read_text().count("Roll for initiative.") == 1

    def test_glob_fallback_sees_both(self, tmp_path, monkeypatch):
        from transcript_combobulator.combine import combine_transcripts_from_env

        session, _, _ = self._session(tmp_path, monkeypatch)
        monkeypatch.setenv("DEDUPE_STRATEGY", "none")
        out = combine_transcripts_from_env(tmp_path, "night")
        assert out[0].read_text().count("Roll for initiative.") >= 1
        assert len(list(session.glob("**/*.vtt"))) == 2

    def test_missing_explicit_file_is_loud(self, tmp_path, monkeypatch):
        from transcript_combobulator.combine import combine_transcripts_from_env

        _, current, _ = self._session(tmp_path, monkeypatch)
        with pytest.raises(CombineError, match="Transcript files not found"):
            combine_transcripts_from_env(
                tmp_path, "night", vtt_files=[current / "nope.vtt"]
            )
