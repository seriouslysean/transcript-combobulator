# AI Agent Development Guidelines

This document contains concrete rules and learnings for AI agent-based development with this codebase.

## Core Principles

### DRY, SIMPLE, KISS
- **Don't Repeat Yourself**: Always reuse existing Makefile targets and functions
- **Keep It Simple**: Prefer simple solutions over complex ones
- **Keep It Simple, Stupid**: Avoid over-engineering

### Use Existing Tools
- **Always use the Makefile**: Never duplicate logic that exists in Makefile targets
- **Respect environment variables**: Use `.env` files and `ENV_FILE` overrides
- **Understand ALL available tools**: Study the Makefile before implementing new functionality

## File and Environment Management

### Environment Configuration
- **Main environment**: `.env` for production configuration
- **Test environment**: `.env.sample` for sample/test data
- **Environment override**: Use `ENV_FILE=.env.sample make command` pattern
- **Never modify main .env**: Use override pattern for testing

### File Naming Conventions
- **Audio files**: `3-nilbits.flac` (number-username pattern)
- **Converted files**: `3-nilbits_16khz.wav` (preserve original with suffix)
- **User directories**: `3-nilbits_16khz/` (matches converted filename)
- **User transcripts**: `nilbits_combined.vtt` (username extracted from pattern)
- **Session transcripts**: `session-combined-1.txt`, `session-combined-2.txt`

### Directory Structure
```
tmp/
├── input/
│   └── session-name/           # Session folders
│       ├── 3-nilbits.flac      # Original audio
│       └── 1-dezfrost.flac     # Numbers may vary
└── output/
    └── session-name/           # Matching output structure
        ├── 3-nilbits_16khz/    # User-specific folders
        │   ├── nilbits_combined.vtt
        │   └── segments/
        └── session-combined-*.txt
```

## Username-Based Mapping

### Dynamic User Positions
- **Problem**: Discord Craig bot assigns numbers based on join order
- **Solution**: Map usernames, not directory numbers
- **Pattern**: `3-nilbits_16khz` → extract `nilbits` → map to `TRANSCRIPT_*_USERNAME`

### Environment Variables
```sh
# Use USERNAME not DIR
TRANSCRIPT_1_USERNAME="nilbits"    # ✅ Correct
TRANSCRIPT_1_DIR="3-nilbits"       # ❌ Fragile
```

## Processing Pipeline

### Complete Workflow
1. **Audio Conversion**: FLAC/MP3 → 16kHz WAV
2. **Voice Activity Detection**: Create segments
3. **Individual Transcription**: Each segment → VTT
4. **User Combination**: All segments → user VTT
5. **Session Combination**: All users → chronological transcript
6. **Smart Chunking**: Split large sessions intelligently

### Memory Management
- **Process segments individually**: Never accumulate all segments in memory
- **Write immediately**: Save VTT files after each segment
- **Clean up**: Remove temporary files after combination

## Transcription Optimizations

### Performance Fixes
- **Disable word timestamps**: `WHISPER_WORD_TIMESTAMPS=false`
- **Reason**: Word timestamp alignment causes hanging on some segments
- **Impact**: Significant performance improvement, no functional loss

### Logging Best Practices
- **Essential progress only**: Model loading, segment counts, progress indicators
- **No verbose debug**: Avoid per-segment text output unless debugging
- **Flush stdout**: Ensure real-time progress display

## Chunking Logic

### Smart Chunking Rules
- **Minimum threshold**: Don't split if fewer than 5 entries per chunk
- **Single file fallback**: Create one file if content is too short
- **Respect CHUNKS setting**: But override when inappropriate

### Implementation
```python
if len(all_entries) < chunks or len(all_entries) < min_entries_per_chunk:
    actual_chunks = 1  # Override CHUNKS setting
else:
    actual_chunks = chunks
```

## Testing Patterns

### Use Sample Files
- **Setup samples**: Add audio files to `samples/` directory, then `make create-sample-files`
- **Test with JFK samples**: Uses `tmp/input/jfk-sample/` session structure
- **JFK sample environment**: `ENV_FILE=.env.jfk-sample make command`
- **Environment override**: Always use environment files for different datasets

### Debugging Commands
```sh
# Test individual steps
make process-vad file=path/to/file.wav
make transcribe-segments file=path/to/file.wav
make combine-transcripts session=session-name

# Test with JFK samples
ENV_FILE=.env.jfk-sample make run folder=tmp/input/jfk-sample
ENV_FILE=.env.jfk-sample make run-single file=tmp/input/jfk-sample/jfk_padded.wav
```

## Code Quality Rules

### Error Handling
- **Graceful degradation**: Skip problematic segments, don't crash
- **Clear error messages**: Help users understand what to fix
- **Logging**: Use appropriate log levels (INFO for progress, WARNING for issues)

### Documentation
- **Update docs**: Keep README current with functionality
- **Include examples**: Show concrete usage patterns
- **Explain patterns**: Document why username mapping is needed

## Common Pitfalls

### Environment Issues
- **❌ Modifying main .env**: Always use override pattern
- **❌ Hardcoding values**: Use environment variables
- **❌ Ignoring existing settings**: Check current configuration first

### Memory Problems
- **❌ Accumulating segments**: Process individually
- **❌ Large model state**: Monitor memory usage
- **❌ Not cleaning up**: Remove temporary files

### File Handling
- **❌ Assuming file structure**: Check what actually exists
- **❌ Hardcoding paths**: Use configurable paths
- **❌ Not handling edge cases**: Empty files, missing directories

## Testing Methodology

### Development Flow
1. **Create sample files**: `make create-sample-files`
2. **Test single file**: `make run-single file=tmp/input/jfk_1.wav`
3. **Test combination**: `ENV_FILE=.env.sample make combine-transcripts session=jfk_1`
4. **Test full workflow**: `make run folder=tmp/input/session`

### Validation Points
- **VAD segment count**: Should appear in logs
- **User VTT creation**: Check individual combined files
- **Session combination**: Verify chronological order
- **Chunking logic**: Ensure appropriate splitting

## Session-Specific Processing

### Session Parameter Usage
- **Target specific sessions**: `make combine-transcripts session=session-name`
- **Avoid processing all files**: Use session parameter to filter
- **Environment isolation**: Different .env files for different sessions

### Directory Organization
- **Preserve input structure**: Output mirrors input directory organization
- **Session-based output**: Each session gets its own output directory
- **User-specific folders**: Individual users get subdirectories


## Performance Considerations

### Optimization Strategies
- **Batch processing**: Process multiple files in sequence
- **Memory management**: Clear intermediate data
- **Model reuse**: Load Whisper model once per session
- **Efficient I/O**: Write files immediately, don't buffer

### Monitoring
- **Progress logging**: Show real-time progress
- **Error tracking**: Log but don't crash on individual failures
- **Performance metrics**: Track processing time per segment

## Integration Guidelines

### AI Tool Integration
- **Optimized format**: Combined transcripts work well with language models
- **Clear speaker identification**: Character names in output
- **Chronological order**: Preserve conversation flow
- **Reasonable chunk sizes**: Balance readability and context

### Workflow Integration
- **Makefile-driven**: All operations through make targets
- **Environment-based**: Configuration through .env files
- **Session-organized**: Natural organization by recording session
- **Resume-friendly**: Individual steps can be re-run independently