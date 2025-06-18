# Transcript Precombobulator

Pre-processes audio files for better Whisper transcription by trimming silence and normalizing audio. Now includes integrated transcript combination functionality that eliminates the need for the separate transcript-recombobulator project.

## Features

- **Voice Activity Detection (VAD)** - Segments audio by detecting speech vs silence
- **Whisper Transcription** - High-quality speech-to-text with confidence scoring
- **Integrated Transcript Combination** - Combines multiple speaker transcripts into chronological session logs
- **Environment-based Configuration** - Easy setup for D&D campaigns and multi-speaker sessions
- **Multiple Output Formats** - VTT files for individual speakers, combined session transcripts

## Setup

```bash
brew install pyenv
git clone --recurse-submodules git@github.com:seriouslysean/transcript-precombobulator.git
cd transcript-precombobulator
make setup
```

## Quick Start

1. **Copy your audio files** to `tmp/input/` (optionally in subdirectories):
   ```bash
   # Example: organize by session date
   mkdir -p tmp/input/2025-06-16
   cp /path/to/session-audio/*.wav tmp/input/2025-06-16/
   ```

2. **Configure your campaign** in `.env`:
   ```bash
   # Copy example configuration
   cp .env.examples .env
   # Edit .env with your character mappings
   ```

3. **Run the complete pipeline**:
   ```bash
   make run    # Process all files and combine transcripts
   # OR process specific files
   make run-single file=tmp/input/2025-06-16/player1.wav
   ```

4. **Find your results**:
   - Individual transcripts: `tmp/output/2025-06-16/[speaker]/[speaker].vtt`
   - Combined session transcript: `tmp/output/2025-06-16/[campaign-name]_transcript.txt`

## Commands

```bash
make setup                          # Install dependencies and download Whisper model
make run                            # Process all WAV files in tmp/input/ and combine
make run-single file=X              # Process single file and combine  
make combine-transcripts            # Combine existing transcripts only
make combine-transcripts session=X  # Combine transcripts for specific session
make test                          # Run test suite
make clean                         # Clean temporary files
```

## Environment Configuration

Copy `.env.examples` to `.env` and configure for your campaign:

```bash
# Campaign settings
CAMPAIGN_NAME="Rime of the Frostmaiden - Session 5"
DEDUPE_STRATEGY=consecutive
INCLUDE_TIMESTAMPS=false
SKIP_FILTERS="[AUDIO OUT],[BLANK_AUDIO]"

# Character mappings (add as many as needed)
TRANSCRIPT_1_DIR="1-player1"
TRANSCRIPT_1_PLAYER="Alice"
TRANSCRIPT_1_ROLE="Player"
TRANSCRIPT_1_CHARACTER="Thorin"
TRANSCRIPT_1_DESCRIPTION="Dwarf Fighter"

TRANSCRIPT_2_DIR="dm-session"
TRANSCRIPT_2_PLAYER="Bob"
TRANSCRIPT_2_ROLE="DM"
TRANSCRIPT_2_CHARACTER="DM"
TRANSCRIPT_2_DESCRIPTION="Dungeon Master"
```

## Directory Structure

The system preserves your input directory organization:

```
tmp/
├── input/
│   ├── 2025-06-16/          # Session directory
│   │   ├── player1.wav      # Audio files
│   │   ├── player2.wav
│   │   └── dm.wav
│   └── 2025-06-23/          # Another session
│       └── *.wav
└── output/
    ├── 2025-06-16/          # Matching output structure
    │   ├── 1-player1/       # Individual transcript directories
    │   │   ├── *.vtt
    │   │   └── *.wav
    │   ├── dm-session/
    │   └── campaign_transcript.txt  # Combined transcript
    └── 2025-06-23/
        └── ...
```

## Examples

Process audio files organized by session date:

```bash
# Organize audio files
mkdir -p tmp/input/2025-06-16
cp /path/to/session/*.wav tmp/input/2025-06-16/

# Process entire session
make run

# Or process specific files
make run-single file=tmp/input/2025-06-16/dm.wav

# Combine transcripts for specific session
make combine-transcripts session=2025-06-16
```

## Integration with AI Tools

The combined transcript is optimized for Large Language Models:

```bash
# Use with your favorite AI tool
cat tmp/output/2025-06-16/campaign_transcript.txt | llm "Summarize this D&D session"
```

## Migration from transcript-recombobulator

This project replaces the separate transcript-recombobulator with integrated functionality:

- ✅ **Environment variables** instead of command-line arguments  
- ✅ **Automatic integration** with transcription workflow
- ✅ **Session organization** with directory structure preservation
- ✅ **All original features** (deduplication, filtering, chunking)
