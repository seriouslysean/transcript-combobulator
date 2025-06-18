# Transcript Combination Examples

This document shows how to use the integrated transcript combination functionality in the transcript-precombobulator project.

## Quick Start

The transcript combination feature is automatically integrated into the transcription workflow. After running `make run` or `make run-single`, you can combine all generated transcripts with:

```bash
make combine-transcripts
```

This uses your environment configuration from `.env` to automatically map transcript directories to character information.

## Environment Configuration

Configure your character mappings in `.env`:

```bash
# Campaign settings
CAMPAIGN_NAME="Rime of the Frostmaiden - Icewind Dale"
DEDUPE_STRATEGY=consecutive
INCLUDE_TIMESTAMPS=false
SKIP_FILTERS="[AUDIO OUT],[BLANK_AUDIO]"

# Character mappings
TRANSCRIPT_1_DIR="1-dezfrost"
TRANSCRIPT_1_PLAYER="Dez"
TRANSCRIPT_1_ROLE="Player"
TRANSCRIPT_1_CHARACTER="Frodrick"
TRANSCRIPT_1_DESCRIPTION="Goliath Barbarian"

TRANSCRIPT_2_DIR="2-burger_bear"
TRANSCRIPT_2_PLAYER="Ozzy Backlund"
TRANSCRIPT_2_ROLE="Player"
TRANSCRIPT_2_CHARACTER="William"
TRANSCRIPT_2_DESCRIPTION="Human Druid"

TRANSCRIPT_3_DIR="3-nilbits"
TRANSCRIPT_3_PLAYER="Sean"
TRANSCRIPT_3_ROLE="DM"
TRANSCRIPT_3_CHARACTER="DM"
TRANSCRIPT_3_DESCRIPTION="Dungeon Master"

# Add more as needed...
```

## Complete Workflow

1. **Process audio files:**
   ```bash
   # Copy your audio files to tmp/input/
   cp /path/to/session-audio/*.wav tmp/input/
   
   # Run the complete transcription pipeline
   make run
   ```

2. **This automatically:**
   - Segments audio using Voice Activity Detection
   - Transcribes each segment with Whisper
   - Generates VTT files for each speaker
   - Combines all transcripts into a single chronological file

3. **Output files:**
   - Individual transcripts: `tmp/output/[speaker]/[speaker].vtt`
   - Combined transcript: `tmp/output/[campaign-name]_transcript.txt`

## Manual Combination

You can also run just the combination step:

```bash
make combine-transcripts
```

## Example Output

The combined transcript includes a summary and chronologically ordered dialogue:

```
Summary:
Dez - Frodrick - Goliath Barbarian
Ozzy Backlund - William - Human Druid
Sean - DM - Dungeon Master
Justin - Clemeth Fandango - Halfling Rogue
Trevor - Argentum - Dragonborn Fighter

TRANSCRIPT:
DM: The narrow alley feels like a tomb. Cephick's headless corpse lies in a spreading pool of frost...
Frodrick: I feel fine, just so you guys know. Feel real good.
William: I mean, I basically, we almost died.
DM: Give me a wisdom saving throw.
Frodrick: A wisdom saving throw would be a 14.
```

## Configuration Options

### Deduplication Strategies
- `false`: No deduplication
- `consecutive`: Remove consecutive duplicate messages
- `unique`: Keep only the first occurrence of each unique message

### Content Filtering
Use `SKIP_FILTERS` to remove unwanted content:
- `"[AUDIO OUT],[BLANK_AUDIO]"` - Remove audio issues
- `"/\\[.*\\]/"` - Remove all bracketed content (regex)

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