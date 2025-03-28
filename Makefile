.PHONY: setup run run-single clean clean-all test setup-whisper install lint create-sample-files test-segment regenerate-vtt process-vad transcribe-segments create-test-files

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

# Run the transcription on all WAV files in tmp/input
run:
	count=$$(ls -1 $(ROOT_DIR)/tmp/input/*.wav 2>/dev/null | wc -l); \
	if [ $$count -eq 0 ]; then \
		echo "No WAV files found in tmp/input/"; \
		exit 1; \
	fi
	for file in $(ROOT_DIR)/tmp/input/*.wav; do \
		$(MAKE) run-single file=$$file; \
	done

# Run transcription on a single file
run-single:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make run-single file=path/to/file.wav"; \
		exit 1; \
	fi
	echo "Processing $$(basename $(file))..."
	$(MAKE) process-vad file=$(file)
	$(MAKE) transcribe-segments file=$(file)

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
