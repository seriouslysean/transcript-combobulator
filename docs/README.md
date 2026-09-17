# Transcript Combobulator

Audio transcription tool for multi-speaker recordings with separate audio files per speaker.

## What It Does

Processes separate audio files and creates organized transcripts:
- **Individual transcripts** - One VTT file per audio file/speaker
- **Combined session transcripts** - Chronological conversation flow across all speakers
- **Speaker mapping** - You configure which audio files map to which speakers

**Requires**: Separate audio files per speaker (tested with Craig Discord bot output)

## Setup

### Prerequisites

| Requirement | Debian / Raspberry Pi OS | macOS |
|---|---|---|
| Python 3.10+ with `venv` | `sudo apt install python3 python3-venv` | `brew install python` (or pyenv, python.org) |
| ffmpeg (audio decoding) | `sudo apt install ffmpeg` | `brew install ffmpeg` |

Debian bookworm ships Python 3.11 and trixie ships 3.13; both work. No
version manager is required. `make check-deps` reports what is missing.

### Install

```sh
git clone https://github.com/seriouslysean/transcript-combobulator.git
cd transcript-combobulator
make setup
```

`make setup` verifies prerequisites, creates `.venv` from `python3` on your
`PATH`, installs pinned dependencies, and downloads the Whisper model
(`WHISPER_MODEL`, ~1.6 GB for the default `large-v3-turbo`). To use a specific
interpreter:

```sh
make setup PYTHON=/usr/bin/python3.12
```

pyenv users: the committed `.python-version` still selects 3.10 through the
pyenv shim, so `make setup` behaves as before.

Low-memory hosts (for example an 8 GB Raspberry Pi) should set
`PARALLEL_JOBS=1` and a smaller `WHISPER_MODEL` in `.env`; each worker holds a
full copy of the model in RAM. `models/` can be pre-seeded by copying the `.pt`
file from another machine instead of downloading.

## Quick Start

1. **Configure speaker mappings**:
   ```sh
   cp .env.example .env
   # Edit .env to map your audio files to speakers
   ```

2. **Add audio files** (one per speaker):
   ```sh
   mkdir -p tmp/input/my-session
   cp /path/to/craig-output/*.flac tmp/input/my-session/
   ```

3. **Run transcription**:
   ```sh
   make run folder=tmp/input/my-session
   ```

4. **Find results**:
   - Individual transcripts: `tmp/output/my-session/{speaker}/{speaker}.vtt`
   - Combined transcripts: `tmp/output/my-session/my-session-combined-*.txt`

## Configuration

Speaker mapping in `.env`:
```sh
# Map audio files to speakers
TRANSCRIPT_1_USERNAME=dm              # From filename 1-dm.flac
TRANSCRIPT_1_NAME="DM"               # Display name
TRANSCRIPT_1_LABEL="DM"              # Speaker label in transcript
TRANSCRIPT_1_DESCRIPTION="Dungeon Master"

TRANSCRIPT_2_USERNAME=barbarian       # From filename 2-barbarian.flac
TRANSCRIPT_2_NAME="Player 1"
TRANSCRIPT_2_LABEL="Barbarian"
TRANSCRIPT_2_DESCRIPTION="Goliath Barbarian"
```

Put domain vocabulary and proper nouns in the prompt. The default setup
reapplies it to every internal Whisper window, including speech islands longer
than 30 seconds:

```sh
WHISPER_PROMPT="Conversation mentioning LOCATION_NAME and CHARACTER_NAME."
WHISPER_CARRY_INITIAL_PROMPT=true
VAD_MIN_SPEECH_DURATION=0.25
```

Test with examples:
```sh
ENV_FILE=.env.example make combine-transcripts session=example
```

## Supported Audio Formats

- **FLAC** (Craig Discord bot default)
- **WAV, MP3, M4A, OGG, AAC, OPUS** (auto-converted to 16kHz WAV)

## Commands

```sh
# Setup
make check-deps                         # Verify Python 3.10+, venv, and ffmpeg
make setup                              # Install dependencies and download Whisper model

# Processing
make run                                # Process all files in tmp/input/
make run folder=tmp/input/session-name  # Process specific session
make run-single file=path/to/file.flac  # Process single file
make run folder=path/to/session force=1 # Reprocess completed files

# Combination (if needed separately)
make combine-transcripts session=session-name

# Utilities
make clean                              # Clean temporary files
make test                               # Run test suite
```

### Run telemetry

Every batch run ends with per-file and session timing tables and writes a
machine-readable report beside the combined transcript:

```text
tmp/output/<session>/<session>-metrics.json
```

The report includes total audio and wall time, real-time factor, worker
utilization, cache hits, conversion/VAD/transcription stage durations, model
load reuse, and every VAD chunk's audio decode, Whisper inference, and result
processing time. Persisted files use opaque file IDs and exclude source names,
paths, environment filenames, raw error messages, transcript text, and prompt
content.

## Example Output

Combined transcript format:
```
Summary:
DM - DM - Dungeon Master
Player 1 - Barbarian - Goliath Barbarian
Player 2 - Druid - Human Druid

TRANSCRIPT:
DM: The wind howls through the ruined village.
Barbarian: That's a 16 on my save.
Druid: I cast Detect Magic, just in case.
...
```

## Troubleshooting

**"No mapping found for directories"**: Update `TRANSCRIPT_*_USERNAME` in your .env file to match your audio filenames.

**Need different settings for different sessions?**: Use `ENV_FILE=.env.session2 make run`
