# Transcript Precombobulator

Audio transcription tool for multi-speaker recordings with separate audio files per speaker.

**Requires**: Separate audio files per speaker (tested with Craig Discord bot output)

## Quick Start

```sh
make setup
cp .env.example .env
# Edit .env to map audio files to characters
make run folder=tmp/input/your-session
```

## Testing with Sample Audio

For testing and development, set up sample audio files:

```sh
# 1. Add sample audio files to samples/ directory
# Example: samples/jfk.wav

# 2. Create sample files in proper session structure
make create-sample-files

# 3. Test with sample environment
ENV_FILE=.env.jfk-sample make run folder=tmp/input/jfk-sample

# Individual testing steps
ENV_FILE=.env.jfk-sample make run-single file=tmp/input/jfk-sample/jfk_padded.wav
ENV_FILE=.env.jfk-sample make combine-transcripts
```

## Documentation

See [docs/README.md](docs/README.md) for complete usage instructions.