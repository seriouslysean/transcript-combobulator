# Transcript Combination Examples

This document shows how to use the integrated transcript combination functionality in the transcript-precombobulator project.

## Quick Start

The transcript combination feature is automatically integrated into the transcription workflow. After running `make run` or `make run-single`, you can combine all generated transcripts with:

```bash
make combine-transcripts session=example
```

This uses your environment configuration from `.env.example` to automatically map transcript directories to character information.

## Environment Configuration

Configure your character mappings in `.env.example`:

```bash
# Campaign settings
CAMPAIGN_NAME="Example Campaign"
DEDUPE_STRATEGY=consecutive
INCLUDE_TIMESTAMPS=false
SKIP_FILTERS="[AUDIO OUT],[BLANK_AUDIO]"

# Character mappings
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

# Add more as needed...
```

## Complete Workflow

1. **Process audio files:**

```bash
   # Copy your audio files to tmp/input/example/
   cp /path/to/sample-audio/*.wav tmp/input/example/

   # Run the complete transcription pipeline
   make run
```

2. **This automatically:**
   - Segments audio using Voice Activity Detection
   - Transcribes each segment with Whisper
   - Generates VTT files for each speaker
   - Combines all transcripts into a single chronological file

3. **Output files:**
   - Individual transcripts: `tmp/output/example/[speaker]/[speaker].vtt`
   - Combined transcript: `tmp/output/example/example-combined-1.txt`, `example-combined-2.txt`

## Manual Combination

You can also run just the combination step:

```bash
ENV_FILE=.env.example make combine-transcripts session=example
```

## Example Output

The combined transcript includes a summary and chronologically ordered dialogue:

```
Summary:
Player 1 - Barbarian - Goliath Barbarian
DM - DM - Dungeon Master

TRANSCRIPT:
DM: The wind howls through the ruined village. You can barely see through the snow.
Barbarian: That's a 16 on my save.
DM: Blue lights flicker in the distance. The cold is biting, even for Icewind Dale.
Barbarian: I barely feel the cold. Used to it.
...
```

## Configuration Options

### Deduplication Strategies
- `false`: No deduplication
- `consecutive`: Remove consecutive duplicate messages
- `unique`: Keep only the first occurrence of each unique message

### Content Filtering
Use `SKIP_FILTERS` to remove unwanted content:
- `"[AUDIO OUT],[BLANK_AUDIO]"` - Remove audio issues
- `"/\[.*\]/"` - Remove all bracketed content (regex)

### Timestamps
Set `INCLUDE_TIMESTAMPS=true` to include timing information in the output.

## Integration with AI Tools

The combined transcript format is optimized for use with Large Language Models for:
- Session summaries
- Character development tracking
- Plot point extraction
- Campaign note generation

## Comparison with Original Tool

This integrated approach replaces the need for the separate `transcript-recombobulator` project by:
- Using environment variables instead of command-line arguments
- Integrating directly into the transcription workflow
- Maintaining all the same functionality and options
- Providing better organization within a single project structure
