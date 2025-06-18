# Transcript Precombobulator

## Environment Configuration

The application loads environment variables from a `.env` file by default. To use a different environment file (for example, for testing with example data), set the `ENV_FILE` environment variable to the path of the desired env file:

```sh
export ENV_FILE=.env.example
make combine-transcripts session=example
```

Or, for a one-off command:

```sh
ENV_FILE=.env.example make combine-transcripts session=example
```

This allows you to flexibly switch between real and example output/test data. The `.env.example` file is provided as a reference for the example output in `tmp/output/example/`.

## Example Output

The example output folder is located at `tmp/output/example/` and is structured for testing the transcript combination logic. The `.env.example` file is configured to match this structure.

## Restoring Environment Files

- `.env` should contain your real/production configuration.
- `.env.example` should contain the example/generic configuration for testing.

## Running Transcript Combination

To combine transcripts for the example data:

```sh
ENV_FILE=.env.example make combine-transcripts session=example
```

---

# Features

- **Voice Activity Detection (VAD)** - Segments audio by detecting speech vs silence
- **Whisper Transcription** - High-quality speech-to-text with confidence scoring
- **Integrated Transcript Combination** - Combines multiple speaker transcripts into chronological session logs
- **Environment-based Configuration** - Easy setup for D&D campaigns and multi-speaker sessions
- **Multiple Output Formats** - VTT files for individual speakers, combined session transcripts

## Setup

```sh
brew install pyenv
git clone --recurse-submodules git@github.com:seriouslysean/transcript-precombobulator.git
cd transcript-precombobulator
make setup
```

## Quick Start

1. **Copy your audio files** to `tmp/input/` (optionally in subdirectories):
   ```sh
   # Example: organize by session name
   mkdir -p tmp/input/example
   cp /path/to/sample-audio/*.wav tmp/input/example/
   ```

2. **Configure your campaign** in `.env.example`:
   ```sh
   # Copy example configuration
   cp .env.example .env
   # Edit .env with your character mappings if needed
   ```

3. **Run the complete pipeline**:
   ```sh
   make run    # Process all files and combine transcripts
   # OR process specific files
   make run-single file=tmp/input/example/player1.wav
   ```

4. **Find your results**:
   - Individual transcripts: `tmp/output/example/[speaker]/[speaker].vtt`
   - Combined session transcript: `tmp/output/example/example-combined-1.txt`, `example-combined-2.txt`

## Commands

```sh
make setup                          # Install dependencies and download Whisper model
make run                            # Process all WAV files in tmp/input/ and combine
make run-single file=X              # Process single file and combine
make combine-transcripts            # Combine existing transcripts only
make combine-transcripts session=example  # Combine transcripts for example session
make test                          # Run test suite
make clean                         # Clean temporary files
```

## Environment Configuration

Copy `.env.example` to `.env` and configure for your campaign:

```sh
# Campaign settings
CAMPAIGN_NAME="Example Campaign"
DEDUPE_STRATEGY=consecutive
INCLUDE_TIMESTAMPS=false
SKIP_FILTERS="[AUDIO OUT],[BLANK_AUDIO]"

# Character mappings (add as many as needed)
TRANSCRIPT_1_DIR="player1-barbarian"
TRANSCRIPT_1_PLAYER="Player 1"
TRANSCRIPT_1_ROLE="Player"
TRANSCRIPT_1_CHARACTER="Barbarian"
TRANSCRIPT_1_DESCRIPTION="Goliath Barbarian"

TRANSCRIPT_2_DIR="player0-dm"
TRANSCRIPT_2_PLAYER="DM"
TRANSCRIPT_2_ROLE="DM"
TRANSCRIPT_2_CHARACTER="DM"
TRANSCRIPT_2_DESCRIPTION="Dungeon Master"
```

## Directory Structure

The system preserves your input directory organization:

```
tmp/
├── input/
│   └── example/            # Example session directory
│       ├── player1.wav     # Audio files
│       ├── player2.wav
│       └── dm.wav
└── output/
    └── example/            # Matching output structure
        ├── player1-barbarian/
        │   └── player1-barbarian.vtt
        ├── player0-dm/
        │   └── player0-dm.vtt
        ├── example-combined-1.txt
        └── example-combined-2.txt
```

## Examples

Process audio files organized by session name:

```sh
# Organize audio files
mkdir -p tmp/input/example
cp /path/to/sample/*.wav tmp/input/example/

# Process entire session
make run

# Or process specific files
make run-single file=tmp/input/example/player1.wav

# Combine transcripts for example session
ENV_FILE=.env.example make combine-transcripts session=example
```

## Integration with AI Tools

The combined transcript is optimized for Large Language Models:

```sh
# Use with your favorite AI tool
cat tmp/output/example/example-combined-1.txt | llm "Summarize this D&D session"
```

## Migration from transcript-recombobulator

This project replaces the separate transcript-recombobulator with integrated functionality:

- ✅ **Environment variables** instead of command-line arguments
- ✅ **Automatic integration** with transcription workflow
- ✅ **Session organization** with directory structure preservation
- ✅ **All original features** (deduplication, filtering, chunking)
