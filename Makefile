.PHONY: setup run run-single clean clean-all test setup-whisper install lint create-sample-files test-segment

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
	cd $(ROOT_DIR) && git submodule update --init --recursive
	cd $(ROOT_DIR) && pyenv install --skip-existing 3.10
	cd $(ROOT_DIR) && python3.10 -m venv .venv
	$(VENV_CMD) pip install --upgrade pip
	$(VENV_CMD) pip install -e .

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
	$(VENV_CMD) python -c "from pathlib import Path; from src.transcribe import transcribe_audio; transcribe_audio(Path('$(file)'))"

# Clean up generated files
clean:
	cd $(ROOT_DIR) && find tmp -type f -not -path "tmp/input/*" -not -name ".gitkeep" -delete
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	cd $(ROOT_DIR) && rm -rf .pytest_cache/ .coverage .mypy_cache/

# Clean everything and recreate directory structure
clean-all:
	cd $(ROOT_DIR) && find tmp -type f -not -name ".gitkeep" -delete
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

# Setup whisper and download sample files
setup-whisper:
	$(VENV_CMD) python tools/setup_whisper.py --model large-v3 --samples

# Create sample files for running the transcription
create-sample-files:
	if [ ! -f "$(ROOT_DIR)/deps/whisper.cpp/samples/jfk.wav" ]; then \
		echo "JFK sample file not found. Run 'make setup-whisper' first."; \
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
