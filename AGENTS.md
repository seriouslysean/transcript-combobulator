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
- **Skip filters match raw text**, not normalized text. So `[BLANK_AUDIO]` in a
  VTT line is filtered by the literal `[BLANK_AUDIO]` filter.

## Makefile Usage

Always use the Makefile:

| Command | Purpose |
|---------|---------|
| `make setup` | Create venv, install deps, download whisper model |
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
- Each parallel worker loads its own whisper model. Keep `PARALLEL_JOBS` small
  on low-RAM machines (default 2). `TORCH_THREADS=0` auto-splits threads
  across workers.
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
  `samples/jfk.wav`. You must have a real `samples/jfk.wav` file present.
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

- `torchaudio>=2.10` requires `torchcodec`. Both are pinned in `pyproject.toml`.
- `silero-vad` is pulled in; VAD model is downloaded on first `load_silero_vad()`
  call and cached.

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
