# Technical Implementation

Implementation details for developers and advanced users.

## Architecture

- **Voice Activity Detection (VAD)** - Silero VAD model segments audio by speech detection
- **Whisper Transcription** - OpenAI Whisper for speech-to-text with confidence scoring
- **Username-based Directory Mapping** - Handles dynamic user positions (Discord Craig bot)
- **Environment-based Configuration** - All settings via .env files

## Processing Pipeline

1. **Audio Conversion** - FLAC/MP3/etc → 16kHz WAV (optimal for Whisper)
2. **Voice Activity Detection** - Segments audio by speech vs silence
3. **Individual Transcription** - Each segment → VTT with timestamps
4. **User Combination** - All segments per user → combined VTT file
5. **Session Combination** - All users → chronological session transcript
6. **Smart Chunking** - Large sessions split, small ones kept intact

## Configuration Options

### Audio Processing
- `SAMPLE_RATE=16000` - Whisper optimal sample rate
- `TRANSCRIPTION_MODE=vad` - Use VAD for segmentation
- `VAD_THRESHOLD=0.5` - Speech detection sensitivity (0.0-1.0)
- `VAD_MIN_SPEECH_DURATION=0.5` - Minimum speech segment length
- `VAD_MIN_SILENCE_DURATION=2.0` - Minimum silence to split
- `PADDING_SECONDS=0.2` - Audio padding around segments

### Whisper Configuration
- `WHISPER_MODEL=large-v3-turbo` - Model size/speed tradeoff
- `WHISPER_DEVICE=cpu` - Processing device
- `WHISPER_LANGUAGE=en` - Language code
- `WHISPER_TEMPERATURE=0.0` - Deterministic output
- `WHISPER_WORD_TIMESTAMPS=false` - Disabled for performance
- `WHISPER_CONFIDENCE_THRESHOLD=50.0` - Filter low-confidence segments

### Combination Settings
- `DEDUPE_STRATEGY=consecutive` - Remove duplicate messages
- `INCLUDE_TIMESTAMPS=false` - Include timing in output
- `SKIP_FILTERS="[AUDIO OUT],[BLANK_AUDIO]"` - Content filtering
- `CHUNKS=2` - Split large transcripts into N parts

## Username Mapping System

Handles dynamic user positions (Discord Craig bot assigns numbers by join order):

```bash
# Audio file: 3-nilbits.flac
# Directory: 3-nilbits_16khz/
# Username extraction: nilbits
# Environment mapping: TRANSCRIPT_2_USERNAME="nilbits"
```

Patterns supported:
- `{number}-{username}_16khz` (e.g., "3-nilbits_16khz")
- Direct username match (directory name in username mapping)

## Performance Optimizations

- **Word timestamps disabled** - Prevents hanging on some segments
- **Segment-based processing** - Avoids memory accumulation
- **Immediate file writing** - Prevents large memory buffers
- **Model reuse** - Load Whisper model once per session

## File Organization

```
tmp/
├── input/                     # Source audio files
│   └── session-name/
│       └── {number}-{username}.{ext}
├── output/                    # Generated files
│   └── session-name/
│       ├── {number}-{username}_16khz/      # User-specific
│       │   ├── {username}_combined.vtt     # User transcript
│       │   ├── *_segment_*.wav             # VAD segments
│       │   └── *_mapping.json              # Segment metadata
│       └── {session}-combined-*.txt        # Session transcripts
```

## Error Handling

- Graceful segment skipping on transcription errors
- Clear error messages for configuration issues
- Progressive logging for long-running processes
- Automatic fallback for unsupported audio formats

## Memory Management

- Process segments individually (no accumulation)
- Immediate VTT file writing after each segment
- Clean up temporary files after combination
- Monitor Whisper model memory usage

## Integration Points

- Environment variable override: `ENV_FILE=.env.custom make run`
- Session-specific processing: `make combine-transcripts session=name`
- Single file processing: `make run-single file=path/to/file`
- Resumable workflow: Individual make targets for each step

## Dependencies

- **Python 3.10+** - Core runtime
- **pyenv** - Python version management
- **OpenAI Whisper** - Speech recognition
- **Silero VAD** - Voice activity detection
- **PyTorch** - Neural network backend
- **soundfile** - Audio file I/O
- **webvtt-py** - VTT file handling