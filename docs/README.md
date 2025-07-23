# Transcript Precombobulator

Audio transcription tool for multi-speaker recordings with separate audio files per speaker.

## What It Does

Processes separate audio files and creates organized transcripts:
- **Individual transcripts** - One VTT file per audio file/speaker
- **Combined session transcripts** - Chronological conversation flow across all speakers
- **Character mapping** - You configure which audio files map to which characters

**Requires**: Separate audio files per speaker (tested with Craig Discord bot output)

## Setup

```sh
# Clone and setup
git clone https://github.com/seriouslysean/transcript-precombobulator.git
cd transcript-precombobulator
make setup
```

Requires Python 3.10+ and pyenv.

## Quick Start

1. **Configure character mappings**:
   ```sh
   cp .env.example .env
   # Edit .env to map your audio files to characters
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

Character mapping in `.env`:
```sh
# Map audio files to characters
TRANSCRIPT_1_USERNAME=dm              # From filename 1-dm.flac
TRANSCRIPT_1_PLAYER="DM"              # Display name  
TRANSCRIPT_1_CHARACTER="DM"           # Character name
TRANSCRIPT_1_DESCRIPTION="Dungeon Master"

TRANSCRIPT_2_USERNAME=barbarian       # From filename 2-barbarian.flac
TRANSCRIPT_2_PLAYER="Player 1"
TRANSCRIPT_2_CHARACTER="Barbarian"
TRANSCRIPT_2_DESCRIPTION="Goliath Barbarian"
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
make setup                              # Install dependencies and download Whisper model

# Processing  
make run                                # Process all files in tmp/input/
make run folder=tmp/input/session-name  # Process specific session
make run-single file=path/to/file.flac  # Process single file

# Combination (if needed separately)
make combine-transcripts session=session-name

# Utilities
make clean                              # Clean temporary files
make test                               # Run test suite
```

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

**Need different settings for different campaigns?**: Use `ENV_FILE=.env.campaign2 make run`