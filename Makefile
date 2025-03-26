.PHONY: setup run clean test setup-whisper install lint create-sample-files

ROOT_DIR := $(shell pwd)

# Ensure Python environment is set up
setup:
	@if ! command -v pyenv &> /dev/null; then \
		echo "pyenv is not installed. Please install it first."; \
		exit 1; \
	fi
	cd $(ROOT_DIR) && git submodule update --init --recursive
	cd $(ROOT_DIR) && pyenv install --skip-existing 3.10
	cd $(ROOT_DIR) && python3.10 -m venv .venv
	cd $(ROOT_DIR) && . .venv/bin/activate && pip install --upgrade pip
	cd $(ROOT_DIR) && . .venv/bin/activate && pip install -e .

# Run the main script on all WAV files in tmp/input
run:
	@if [ ! -f "$(ROOT_DIR)/tmp/input/jfk_1.wav" ]; then \
		echo "No input files found in tmp/input/"; \
		echo "Run 'make create-sample-files' to create sample files."; \
		exit 1; \
	fi
	@for file in $(ROOT_DIR)/tmp/input/*.wav; do \
		echo "\nProcessing $$(basename $$file)..."; \
		cd $(ROOT_DIR) && . .venv/bin/activate && python -c "from pathlib import Path; from src.transcribe import transcribe_audio; transcribe_audio(Path('$$file'))"; \
	done

# Clean up generated files
clean:
	cd $(ROOT_DIR) && find tmp -type f ! -name ".gitkeep" -delete
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	cd $(ROOT_DIR) && rm -rf .pytest_cache/ .coverage .mypy_cache/

# Clean everything and recreate directory structure
clean-all:
	cd $(ROOT_DIR) && rm -rf tmp/*
	cd $(ROOT_DIR) && mkdir -p tmp/input tmp/output tmp/transcriptions
	cd $(ROOT_DIR) && touch tmp/input/.gitkeep tmp/output/.gitkeep tmp/transcriptions/.gitkeep

# Install dependencies
install:
	cd $(ROOT_DIR) && . .venv/bin/activate && pip install -e .

# Run all tests
test:
	cd $(ROOT_DIR) && . .venv/bin/activate && python -m pytest tests/ -v
	$(MAKE) clean

# Check code style
lint:
	cd $(ROOT_DIR) && . .venv/bin/activate && flake8 .
	cd $(ROOT_DIR) && . .venv/bin/activate && mypy .

# Setup whisper and download sample files
setup-whisper:
	cd $(ROOT_DIR) && . .venv/bin/activate && python tools/setup_whisper.py --model large-v3 --samples

# Create sample files for running the transcription
create-sample-files:
	@if [ ! -f "$(ROOT_DIR)/deps/whisper.cpp/samples/jfk.wav" ]; then \
		echo "JFK sample file not found. Run 'make setup-whisper' first."; \
		exit 1; \
	fi
	cd $(ROOT_DIR) && . .venv/bin/activate && python tools/create_sample_files.py
