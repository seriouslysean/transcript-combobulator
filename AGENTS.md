# Agent Development Guidelines

Concrete rules for working on this codebase. If you're Claude Code or another
coding agent, read this first.

## Module Layout

```
src/
├── config.py           # ENV_FILE-aware settings; imported by everything
├── audio_utils.py      # Validate + convert any audio to 16kHz mono WAV
├── vad.py              # Silero VAD → segment WAVs + mapping JSON
├── whisper.py          # Whisper model wrapping; VTT writing; repetition filter
├── transcribe.py       # Pipeline entry: full-file transcribe via VAD + whisper
├── pipeline_cache.py   # Config-aware completion manifests for resumable runs
├── telemetry.py        # Per-run, per-file, and per-chunk timing reports
├── combine.py          # Merge per-user VTTs → one session transcript
└── logging_config.py   # setup_logging / get_logger

tools/
├── process_batch.py         # Parallel batch processor with rich progress UI
├── process_single_file.py   # One-file pipeline (convert → VAD → transcribe)
├── convert_audio.py         # CLI: one file or all of tmp/input/
├── create_sample_files.py   # Build sample/test audio from samples/
├── setup_whisper.py         # Download whisper model into models/
└── test_whisper.py          # Smoke-test whisper on an audio file
```

The standard pipeline: `tools/process_batch.py` →
`tools/process_single_file.py` → `src.audio_utils.convert_to_wav` →
`src.vad.process_audio` (writes segment WAVs + `<stem>_mapping.json`) →
`src.transcribe.transcribe_segments` → `src.whisper.transcribe_audio_segments`
→ `src.combine.combine_transcripts_from_env`.

## Invariants

- **One place for env loading.** `src/config.py` calls `load_dotenv` at import.
  Do NOT add `load_dotenv` anywhere else.
- **Audio is normalized once.** `src.audio_utils.convert_to_wav` normalizes to
  `[-1, 1]`. Don't re-normalize downstream.
- **Mapping JSON is written once.** `src.vad.process_audio` is the only writer.
  `src.transcribe.transcribe_audio` trusts it and reads it back.
- **Only one `transcribe_audio` public function.** It lives in
  `src.transcribe`. `src.whisper.transcribe_file_direct` is the one-shot
  (no-VAD) variant and is internal to the regenerate-VTT flow.
- **Combine preserves original text.** `_normalize_for_dedup` is used ONLY for
  dedup keys, never for the text written to the output file.
- **Dedup is consecutive, not global.** `DEDUPE_STRATEGY=consecutive` (default)
  drops a cue only when it repeats the previous kept cue for that speaker
  within `DEDUPE_WINDOW_SECONDS`. That is whisper's repeated-line
  hallucination; a genuine "Yeah." ten minutes later must survive. The same
  rule runs in `src.whisper.dedupe_segments` (per-speaker VTT) and
  `src.combine._dedupe_entries` (session). `global` is the legacy lossy mode.
- **Cue offsets use the padded clip start.** VAD writes `start_seconds` (speech)
  and `clip_start_seconds` (what the WAV actually contains); whisper's
  timestamps are relative to the clip, so `clip_start_seconds` is the offset.
- **Partial transcription is a failure.** Any failed chunk raises before the
  manifest is written (`FAIL_ON_PARTIAL_TRANSCRIPTION`), so the cache cannot
  hide a transcript with gaps. A track with no speech at all is an empty
  transcript, not an error (`ALLOW_SILENT_TRACKS`).
- **Usernames match as whole tokens** in per-speaker dir names, so `dez`
  never claims `5-dezfrost`. Batch runs hand combine the exact VTTs they
  produced; stale sibling dirs are ignored.
- **Skip filters match raw text**, not normalized text. So `[BLANK_AUDIO]` in a
  VTT line is filtered by the literal `[BLANK_AUDIO]` filter.

## Makefile Usage

Always use the Makefile:

| Command | Purpose |
|---------|---------|
| `make check-deps` | Verify Python 3.10+, `venv` module, and `ffmpeg` on the host |
| `make setup` | Check deps, create venv from `$(PYTHON)` (default `python3`), install deps, download whisper model |
| `make run folder=path/` | Run the full parallel pipeline |
| `make run-single file=path.wav` | Pipeline for one file (no combine) |
| `make process-vad file=path.wav` | VAD step only |
| `make transcribe-segments file=path.wav` | Transcription step only (needs mapping) |
| `make combine-transcripts [session=...]` | Combine step only |
| `make convert-audio [input=path]` | Convert to 16kHz mono WAV |
| `make regenerate-vtt file=path [threshold=50]` | Re-run whisper with confidence filter |
| `make create-sample-files` | Populate `tmp/input/jfk-sample/` |
| `make create-test-files` | Populate `tmp/input/test_jfk*.wav` for pytest |
| `make test` | Run pytest |
| `make lint` | Run mypy (advisory; annotation coverage not enforced) |

## Batch Run Behaviour

`tools/process_batch.py` is what automation calls, so it fails fast and leaves
a trail:

- `MAPPING_PRECHECK` (default on) runs `src.combine.validate_speaker_mapping`
  against the input stems before any worker starts. Same errors as the combine
  step, minutes earlier.
- A missing `ENV_FILE` raises at import instead of silently loading nothing.
- `LOG_FILE` (default empty) persists worker and parent logs to
  `tmp/output/<session>/<session>.log`; workers tag lines with their audio
  file. `LOG_FILE=none` restores the old drop-everything behaviour. The rich
  table still owns the terminal either way.
- SIGINT and SIGTERM both kill the workers and the Manager and exit
  `128 + signal`, so a systemd stop does not orphan spawned processes.
- All paths derive from `src.config.ROOT_DIR`, which is the repo (from
  `__file__`, or `PROJECT_ROOT` in the shell env), not the cwd. A relative
  `ENV_FILE` resolves against the cwd first, then the repo.

## Environment Files

- `.env` — production config (default).
- `.env.example` — committed template.
- `.env.jfk-sample` — committed, used by tests and `make create-sample-files`.
- `.env.<campaign>` — per-session overrides (e.g. `.env.annihilation`). Invoke
  with `ENV_FILE=.env.annihilation make run ...`.
- Never modify `.env` directly; override with `ENV_FILE=...`.

### Speaker Mapping

Per-user env vars (indexed starting at 1):

```sh
TRANSCRIPT_1_USERNAME="craig_discord_name"   # required, matched as substring in dir name
TRANSCRIPT_1_NAME="Display Name"             # required (falls back to TRANSCRIPT_1_PLAYER)
TRANSCRIPT_1_LABEL="Speaker Tag"             # required (falls back to TRANSCRIPT_1_CHARACTER)
TRANSCRIPT_1_DESCRIPTION="Short bio"         # required
```

If any of `NAME`, `LABEL`, `DESCRIPTION` is empty the mapping is skipped with a
warning. If the username is ambiguous across directories, the combine step
fails loudly.

## File Naming Conventions

- **Input audio**: `3-nilbits.flac` (number-username pattern from Discord Craig)
- **Converted**: `3-nilbits_16khz.wav`
- **Per-user output dir**: `3-nilbits_16khz/`
- **Per-user combined VTT**: `nilbits_combined.vtt` (username extracted from stem)
- **Session combined**: `<session>-combined.txt`, or chunked:
  `<session>-combined-1.txt`, `<session>-combined-2.txt`

## Performance Notes

- Whisper's `word_timestamps=True` hangs on some segments. Keep the default
  `WHISPER_WORD_TIMESTAMPS=false`.
- `beam_size=1` and `condition_on_previous_text=false` are intentional for
  VAD-segment transcription (each segment is already a speech island).
- Each parallel worker loads its own whisper model (~3x the checkpoint size at
  load: fp16 file plus fp32 params). `MEMORY_GUARD` (default on) lowers the
  active worker count so `PARALLEL_JOBS` workers fit in
  `MEMORY_GUARD_FRACTION` of physical RAM; it never raises the count. An 8 GB
  Pi with `large-v3-turbo` lands on 1 worker. `TORCH_THREADS=0` auto-splits
  threads across the active workers.
- `src.whisper.load_whisper_model` loads by file path on purpose. Loading by
  name makes whisper sha256 the whole checkpoint per worker per run.
- On macOS, `multiprocessing.set_start_method("spawn")` is mandatory for torch.
  `tools/process_batch.py` handles this at import time.

## Known Whisper Failure Modes

- **Repetition hallucination** on laughs/silence: whisper emits one word
  hundreds of times (e.g. `"laughs laughs laughs…"`). `src.whisper.collapse_repetition`
  collapses these to a single occurrence before they reach the VTT.
- **Confidence drift** on quiet or ambiguous audio: `make regenerate-vtt
  threshold=50` re-emits a filtered VTT from the saved segment JSON.

## Testing

- `tests/conftest.py` session fixture creates `tmp/input/test_jfk*.wav` from
  the committed `samples/jfk.wav`. It is autouse, so every test needs working
  audio decoding (ffmpeg) even the pure-logic ones.
- `tests/test_batch.py` is fast (mocks + dir fixtures). Safe to run on every
  change.
- `tests/test_combine.py` is fast (pure Python over synthetic VTTs).
- `tests/test_vad.py`, `tests/test_transcription.py`, `tests/test_whisper.py`
  do real whisper inference and are slow. Run only when changing the audio
  pipeline.
- Whisper segment count is nondeterministic — use range assertions, not exact
  counts.
- For similarity checks, use `difflib.SequenceMatcher`, not `set` intersection
  (repeated words break set-based similarity).

## Dependencies (gotchas)

- `torchaudio>=2.10` requires `torchcodec`, and torchcodec is ABI-locked to a
  specific torch release (0.10 ↔ 2.10, 0.11+ ↔ 2.11). Bump all three pins
  together; a loose torchcodec pin installs a mismatched build that fails at
  import. torchcodec 0.11+ is also the first line with Linux aarch64 wheels.
- `silero-vad` bundles its model inside the wheel; `load_silero_vad()` needs no
  network access.
- `ffmpeg` must be on `PATH`. Makefile recipes run under `/bin/sh` (dash on
  Debian), so keep them POSIX: `>/dev/null 2>&1`, never `&>`.

## Error Handling Style

- Custom exceptions: `AudioValidationError`, `VADError`, `WhisperError`,
  `TranscriptionError`, `CombineError`. Wrap underlying errors via `raise ... from`.
- Log warnings for skippable problems (missing segment file). Raise for
  structural problems (no speech detected, missing mapping).
- Individual segment failures do NOT fail the whole pipeline.

## Adding a New Feature

1. Check this file for existing patterns first.
2. Use a Makefile target. Add one if the operation should be reproducible.
3. Put shared logic in `src/`, glue code in `tools/`.
4. Load config only through `src.config`. Never call `load_dotenv` directly.
5. Add a test in `tests/test_<module>.py` using synthetic fixtures where
   possible. Only use the slow whisper tests when actually testing whisper
   behavior.
6. Run `make lint` and `make test` before declaring done.

## Common Pitfalls

- ❌ Editing `.env` (use `ENV_FILE=` instead)
- ❌ Calling `load_dotenv` anywhere other than `src/config.py`
- ❌ Hardcoding paths; use `src.config.OUTPUT_DIR` / `INPUT_DIR` / `get_output_path_for_input`
- ❌ Hardcoding whisper kwargs; use `src.config.get_whisper_options()`
- ❌ Normalizing audio twice (audio_utils does it)
- ❌ Normalizing transcript text for display (normalize is dedup-only)
- ❌ Re-reading a file in the same process just to verify it; trust the write
