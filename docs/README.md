# Transcript Precombobulator

Pre-processes audio files for better Whisper transcription by trimming silence and normalizing audio.

## Setup

```bash
brew install pyenv
git clone --recurse-submodules git@github.com:seriouslysean/transcript-precombobulator.git
cd transcript-precombobulator
make setup

# Always use make commands to ensure proper environment
make test   # Test VAD on sample file
make run    # Process all files
```
