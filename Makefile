.PHONY: setup run run-single clean clean-all test setup-whisper install lint create-sample-files test-segment regenerate-vtt process-vad transcribe-segments create-test-files combine-transcripts convert-audio bump-patch bump-minor bump-major

ROOT_DIR := $(shell pwd)

# Helper to run commands in virtual environment
VENV_CMD = cd $(ROOT_DIR) && . .venv/bin/activate &&

# Make all commands silent by default
.SILENT:

# Ensure Python environment is set up
setup:
	if ! command -v pyenv &> /dev/null; then \
		echo "pyenv is not installed. Please install it first."; \
		exit 1; \
	fi
	cd $(ROOT_DIR) && pyenv install --skip-existing 3.10
	cd $(ROOT_DIR) && python3.10 -m venv .venv
	$(VENV_CMD) pip install --upgrade pip
	$(VENV_CMD) pip install -e .
	$(MAKE) setup-whisper

# Run the transcription on a single file or all audio files in a folder
run:
	if [ -n "$(file)" ]; then \
		if [ ! -f "$(file)" ]; then \
			echo "File not found: $(file)"; \
			exit 1; \
		fi; \
		echo "Processing single file: $$(basename $(file))..."; \
		$(MAKE) run-single file=$(file); \
		if [ -n "$$ENV_FILE" ]; then \
			$(MAKE) combine-transcripts ENV_FILE=$$ENV_FILE; \
		else \
			$(MAKE) combine-transcripts; \
		fi; \
	elif [ -n "$(folder)" ]; then \
		target_dir="$(folder)"; \
		if [ ! -d "$$target_dir" ]; then \
			echo "Directory not found: $$target_dir"; \
			exit 1; \
		fi; \
		session_name=$$(basename "$$target_dir"); \
		$(VENV_CMD) ENV_FILE=$(ENV_FILE) PYTHONPATH=$(ROOT_DIR) python tools/process_batch.py "$$target_dir" --session "$$session_name"; \
	else \
		target_dir="$(ROOT_DIR)/tmp/input"; \
		if [ ! -d "$$target_dir" ]; then \
			echo "Directory not found: $$target_dir"; \
			exit 1; \
		fi; \
		$(VENV_CMD) ENV_FILE=$(ENV_FILE) PYTHONPATH=$(ROOT_DIR) python tools/process_batch.py "$$target_dir"; \
	fi

# Run transcription on a single file (convert -> VAD -> transcribe, but no combine)
run-single:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make run-single file=path/to/file.wav"; \
		exit 1; \
	fi
	echo "Processing $$(basename $(file))..."
	$(VENV_CMD) ENV_FILE=$(ENV_FILE) PYTHONPATH=$(ROOT_DIR) python tools/process_single_file.py $(file)

# Process audio with VAD to generate segments
process-vad:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make process-vad file=path/to/file.wav"; \
		exit 1; \
	fi
	echo "Processing $$(basename $(file)) with VAD..."
	$(VENV_CMD) python -c "from pathlib import Path; from src.process_audio import process_audio; process_audio(Path('$(file)'))"

# Transcribe existing segments for a file
transcribe-segments:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make transcribe-segments file=path/to/file.wav"; \
		exit 1; \
	fi
	echo "Transcribing segments for $$(basename $(file))..."
	$(VENV_CMD) python -c "from pathlib import Path; from src.transcribe import transcribe_segments; transcribe_segments(Path('$(file)'))"


# Clean up generated files
clean:
	cd $(ROOT_DIR) && find tmp -type f -not -path "tmp/input/*" -not -name ".gitkeep" -delete
	cd $(ROOT_DIR) && find tmp -type d -empty -not -path "tmp/input" -delete
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	cd $(ROOT_DIR) && rm -rf .pytest_cache/ .coverage .mypy_cache/

# Clean everything and recreate directory structure
clean-all:
	cd $(ROOT_DIR) && find tmp -type f -not -name ".gitkeep" -delete
	cd $(ROOT_DIR) && find tmp -type d -empty -not -path "tmp/input" -delete
	cd $(ROOT_DIR) && mkdir -p tmp/input tmp/output tmp/transcriptions
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	cd $(ROOT_DIR) && rm -rf .pytest_cache/ .coverage .mypy_cache/

# Install dependencies
install:
	$(VENV_CMD) pip install -e .

# Run all tests
test:
	$(VENV_CMD) python -m pytest tests/ -v
	$(MAKE) clean

# Check code style
lint:
	$(VENV_CMD) flake8 .
	$(VENV_CMD) mypy .

# Setup whisper and download model
setup-whisper:
	$(VENV_CMD) python -c "from pathlib import Path; from dotenv import load_dotenv; import os; load_dotenv(); model = os.getenv('WHISPER_MODEL', 'large-v2'); print(f'Downloading {model} model...'); from tools.setup_whisper import setup_whisper; setup_whisper(model, Path('models'))"

# Create sample files for running the transcription
create-sample-files:
	if [ ! -d "$(ROOT_DIR)/samples" ]; then \
		echo "Samples directory not found at samples/"; \
		exit 1; \
	fi
	$(VENV_CMD) python tools/create_sample_files.py

# Test whisper transcription on a segment
test-segment:
	@if [ "$(file)" = "" ]; then \
		$(VENV_CMD) tools/test_whisper.py; \
	else \
		$(VENV_CMD) tools/test_whisper.py $(file); \
	fi

# Regenerate VTT file with confidence threshold
regenerate-vtt:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make regenerate-vtt file=path/to/file.wav [threshold=50]"; \
		exit 1; \
	fi
	if [ -z "$(threshold)" ]; then \
		$(VENV_CMD) python -c "from pathlib import Path; from src.whisper import regenerate_vtt_for_audio; regenerate_vtt_for_audio(Path('$(file)'))"; \
	else \
		$(VENV_CMD) python -c "from pathlib import Path; from src.whisper import regenerate_vtt_for_audio; regenerate_vtt_for_audio(Path('$(file)'), confidence_threshold=$(threshold))"; \
	fi

# Create test files for running the test suite
create-test-files:
	if [ ! -d "$(ROOT_DIR)/samples" ]; then \
		echo "Samples directory not found at samples/"; \
		exit 1; \
	fi
	$(VENV_CMD) python tools/create_sample_files.py --prefix test_ --copies 1 --padded-copies 3

# Convert audio files to 16kHz WAV format
convert-audio:
	if [ -z "$(input)" ]; then \
		echo "Converting all audio files in tmp/input/ to 16kHz WAV format..."; \
		$(VENV_CMD) python -c "from pathlib import Path; from src.audio_utils import validate_audio_file, needs_conversion, convert_to_wav, get_audio_info_summary; import sys; input_dir = Path('$(ROOT_DIR)/tmp/input'); output_dir = Path('$(ROOT_DIR)/tmp/output'); audio_files = list(input_dir.rglob('*')); audio_files = [f for f in audio_files if f.is_file() and f.suffix.lower() in {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac', '.opus'}]; print(f'Found {len(audio_files)} audio files'); [print(f'Processing: {get_audio_info_summary(f)}') for f in audio_files]; [convert_to_wav(f, output_dir / f.relative_to(input_dir).parent / f'{f.stem}_16khz.wav') if needs_conversion(f) else print(f'Skipping {f.name} (already 16kHz WAV)') for f in audio_files]; print('Conversion complete!')"; \
	else \
		echo "Converting $(input) to 16kHz WAV format..."; \
		$(VENV_CMD) python -c "from pathlib import Path; from src.audio_utils import validate_audio_file, needs_conversion, convert_to_wav, get_audio_info_summary; input_file = Path('$(input)').resolve(); print(f'Input: {get_audio_info_summary(input_file)}'); input_root = Path('$(ROOT_DIR)/tmp/input').resolve(); output_root = Path('$(ROOT_DIR)/tmp/output').resolve(); rel_path = input_file.relative_to(input_root) if input_file.is_relative_to(input_root) else Path(input_file.name); output_file = output_root / rel_path.parent / f'{input_file.stem}_16khz.wav'; convert_to_wav(input_file, output_file) if needs_conversion(input_file) else print('No conversion needed'); print(f'Output: {output_file}')"; \
	fi

# Combine all VTT transcripts in output directory using environment configuration
combine-transcripts:
	echo "Combining all transcripts using environment configuration..."
	if [ -n "$(session)" ]; then \
		$(VENV_CMD) ENV_FILE=$(ENV_FILE) python -c "from pathlib import Path; from src.combine import combine_transcripts_from_env; combine_transcripts_from_env(Path('$(ROOT_DIR)/tmp/output'), '$(session)')"; \
	else \
		$(VENV_CMD) ENV_FILE=$(ENV_FILE) python -c "from pathlib import Path; from src.combine import combine_transcripts_from_env; combine_transcripts_from_env(Path('$(ROOT_DIR)/tmp/output'))"; \
	fi

