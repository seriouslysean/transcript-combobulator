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
├── filter_vtt.py            # Re-emit a speaker VTT from saved JSON above a confidence
├── create_sample_files.py   # Build sample/test audio from samples/
└── setup_whisper.py         # Download whisper model into models/ (no load)
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
  `src.transcribe.transcribe_audio` and the VAD-cached path in
  `tools/process_single_file.py` trust it and read it back.
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

Always use the Makefile. `make help` lists every target; this is the full set:

| Command | Purpose |
|---------|---------|
| `make check-deps` | Verify Python 3.10+, `venv` module, and `ffmpeg` on the host |
| `make setup` | Check deps, create venv from `$(PYTHON)`, `make install`, download whisper model |
| `make install [EXTRAS=dev,mac]` | `pip install -e ".[EXTRAS]"` into `.venv` |
| `make setup-whisper` | Download `WHISPER_MODEL` into `models/` (checksum-verified, no model load) |
| `make run folder=path/ [force=1]` | Full parallel pipeline for one session; the only thing automation calls |
| `make run-single file=path [force=1]` | Convert → VAD → transcribe one file, no combine |
| `make combine-transcripts session=name` | Re-merge a session's per-speaker VTTs |
| `make filter-vtt file=path [threshold=50]` | Re-emit a speaker VTT from its saved JSON above a confidence; no inference |
| `make create-sample-files` | Populate `tmp/input/jfk-sample/` |
| `make test-fast` | pytest without the `slow` (real inference) tests |
| `make test` | Whole suite; never touches `tmp/output` |
| `make lint` | `mypy --strict` |
| `make clean-output` | Delete every session's outputs under `tmp/output` |
| `make clean-tmp` | Also delete `tmp/input` and local caches |

Dropped on purpose: `convert-audio` (wrote a `_16khz` layout nothing consumed),
`regenerate-vtt` (re-transcribed the whole file without VAD), `test-segment`,
`create-test-files` (conftest does it), `process-vad`, `transcribe-segments`,
and the old `clean`, which `make test` used to call and which deleted every
session's transcripts.

## Resume Semantics

`src/pipeline_cache.py` keeps one manifest per input file with a record per
stage: `conversion`, `vad`, `inference`, `vtt`. Each stage's fingerprint is
chained from the previous one over that stage's own settings
(`src.config.get_stage_fingerprint_settings`), so a change re-runs that stage
and everything after it, nothing before it:

| Changed | Re-runs |
|---|---|
| source file, `SAMPLE_RATE` | everything |
| `VAD_*`, `PADDING_SECONDS` | VAD, inference, VTT |
| `WHISPER_*`, model file size/mtime | inference, VTT |
| `DEDUPE_*` | VTT only, rewritten from the saved JSON with no inference |

A record is trusted only if its fingerprint matches and every artifact it
names exists. Recording a stage drops every later record; a non-forced run
drops the `vtt` (completion) record before any work so an interrupted run
never looks complete. `force=1` deletes the manifest and the checkpoint.

Transcription checkpoints per chunk to `<stem>_progress.jsonl` (header line
with the inference fingerprint, then one JSON line per completed chunk). A
rerun with the same fingerprint reuses those chunks (`status: resumed`) and
transcribes only the missing ones; a truncated last line is ignored and that
chunk redone. The file is removed when every chunk succeeded and kept when
any failed, so `FAIL_ON_PARTIAL_TRANSCRIPTION` reruns retry only the failed
chunks. Verified by killing a real 2.9 h track mid-transcription and
resuming: output identical to the uninterrupted run.

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

- **Input audio**: `tmp/input/<session>/3-nilbits.flac` (number-username from Craig)
- **Per-speaker output dir**: `tmp/output/<session>/3-nilbits/` (input stem)
- **Converted**: `3-nilbits.wav` inside it (16 kHz mono PCM_16, peak-normalised)
- **Per-speaker VTT + JSON**: `3-nilbits.vtt`, `3-nilbits_transcription.json`,
  `3-nilbits_mapping.json`, `3-nilbits_segment_NNN.wav`
  (`src.config.vtt_path_for_input` is the one source of truth for the VTT path)
- **Session outputs**: `<session>-combined.txt` (or `-1.txt`, `-2.txt` when
  `CHUNKS>1`), `<session>-metrics.json`, `<session>.log`

## Performance Notes

- **Conversion streams through ffmpeg.** Measured on a 2.9 h Craig track:
  5.4 s at 41 MB peak RSS versus 13.7 s at 8.8 GB for the old in-memory
  torchaudio path (removed), with sample-identical length and a mean absolute
  difference of 2e-5. Peak normalisation is a two-pass streaming scan, so the
  "normalized once" invariant still holds. ffmpeg gets a timeout of
  max(600 s, 10x the audio duration) so a corrupt file cannot wedge a worker.
- **Silero VAD runs on one thread** (`VAD_THREADS=1`). It processes 512-sample
  frames one at a time; 1 thread measured 2.2x faster than 4 and 3.7x faster
  than 8 on Apple Silicon. The worker's whisper thread count is restored after.
- **VAD streams the WAV and uses Silero's ONNX build.** `src.vad.detect_speech`
  collects frame probabilities over 60 s blocks with the detector's state
  carried across boundaries, then applies a verbatim port of silero-vad
  6.2.1's region logic; `tests/test_vad_streaming.py` asserts identity with
  the installed `get_speech_timestamps`. Segments are written by seeking
  into the WAV. Measured on a 2.9 h track, whole `process_audio` call:
  256 MB peak RSS, regions identical to the whole-file result. The
  TorchScript build (`VAD_BACKEND=jit`) gave the same regions but its peak
  swung between 0.9 and 3.9 GB run to run, which is why onnx is the default.
- **`src/__init__.py` imports nothing.** The batch parent only needs config,
  combine, and telemetry; importing the package must not load torch.
- **One encoder pass per VAD island is the floor on stock whisper, by
  decision.** Whisper encodes a fixed 30 s window, so short islands pay full
  price (utilisation ~24% on a real session). Packing islands into one clip
  was measured at 2.45x on a 2.9 h track with identical words, but whisper
  places segment boundaries from what it hears, never from splices: 56 of
  128 islands (26 with a 2 s gap) were attributed to the previous island's
  time, up to 509 s early. Stock `word_timestamps` doubles decode cost and
  still misassigned 9 of 128. WhisperX and faster-whisper pack the same way
  and fix time with a separate wav2vec2 forced-alignment model; whisper.cpp
  shrinks the encoder per clip (`audio_ctx`, ~3x on short clips). Both are
  outside stock openai-whisper and were ruled out: this project stays on
  stock whisper, slow and correct, rather than patching the model or adding
  a second one. Do not reintroduce packing, encoder-context patches, or a
  detect-and-retry shim.
- **Temperature fallback is available.** `WHISPER_TEMPERATURE=0.0,0.2,0.4`
  enables whisper's re-decode when the compression-ratio or logprob guard
  trips; with the default scalar those guards never fire. Costs CPU only on
  segments that trip it. Not yet A/B'd on a session.

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
- **Confidence drift** on quiet or ambiguous audio: `make filter-vtt
  file=<input> threshold=50` re-emits a filtered VTT from the saved
  `_transcription.json`. No inference; seconds, not hours.

## Testing

- `tests/conftest.py` session fixture creates `tmp/input/test_jfk*.wav` from
  the committed `samples/jfk.wav`. It is autouse, so every test needs working
  audio decoding (ffmpeg) even the pure-logic ones.
- `tests/test_batch.py` is fast (mocks + dir fixtures). Safe to run on every
  change.
- `tests/test_combine.py` is fast (pure Python over synthetic VTTs).
- `tests/test_vad.py`, `tests/test_transcription.py`, `tests/test_whisper.py`
  are marked `slow` (real inference). `make test-fast` skips them; run
  `make test` when changing the audio pipeline.
- `make test` never cleans `tmp/output`. Slow tests write under
  `tmp/output/test_jfk*/`; `make clean-output` removes everything.
- Whisper segment count is nondeterministic — use range assertions, not exact
  counts.
- For similarity checks, use `difflib.SequenceMatcher`, not `set` intersection
  (repeated words break set-based similarity).

## Dependencies (gotchas)

- `torch` and `torchaudio` are an exact pair; torchaudio's wheel declares no
  dependencies so pip will not enforce it. torchaudio 2.11.0 is the final
  maintenance line and caps torch. `torchaudio` must stay importable for
  `silero-vad`, but nothing calls its I/O: audio goes through `soundfile` and
  `ffmpeg`, so `torchcodec` (ABI-locked to torch, no aarch64 wheel before
  0.11) is deliberately not a dependency. Do not reintroduce
  `torchaudio.load`/`save`.
- `silero-vad` bundles both its models inside the wheel; `load_silero_vad()`
  needs no network access. It does not declare `onnxruntime`, which the
  default `VAD_BACKEND=onnx` needs, so `pyproject.toml` pins it.
- `src.vad._speech_regions_from_probs` is a verbatim port of silero's region
  logic because silero only accepts a whole-file tensor. Bumping `silero-vad`
  must keep `tests/test_vad_streaming.py` green; if silero changes the
  algorithm, port the change, do not paper over the diff.
- `ffmpeg` must be on `PATH`. Makefile recipes run under `/bin/sh` (dash on
  Debian), so keep them POSIX: `>/dev/null 2>&1`, never `&>`.

## Error Handling Style

- Custom exceptions: `AudioValidationError`, `VADError`, `WhisperError`,
  `TranscriptionError`, `CombineError`. Wrap underlying errors via `raise ... from`.
- Log warnings for skippable problems (missing segment file). Raise for
  structural problems (missing mapping, missing model, unmapped speaker).
- A track with no detected speech is an empty transcript, not an error
  (`ALLOW_SILENT_TRACKS`).
- Individual chunk failures are logged and skipped inside whisper, but a file
  with any failed chunk raises before its manifest is written
  (`FAIL_ON_PARTIAL_TRANSCRIPTION`) so the cache cannot hide gaps.

## Adding a New Feature

1. Check this file for existing patterns first.
2. Use a Makefile target. Add one if the operation should be reproducible.
3. Put shared logic in `src/`, glue code in `tools/`.
4. Load config only through `src.config`. Never call `load_dotenv` or
   `os.getenv` for a setting anywhere else; add the constant to config.
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
