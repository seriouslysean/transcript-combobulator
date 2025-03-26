.PHONY: setup run clean test setup-whisper install lint

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

# Run the main script
run:
	@if [ ! -f "$(ROOT_DIR)/tmp/input/jfk.wav" ]; then \
		echo "No input file found in tmp/input/jfk.wav"; \
		exit 1; \
	fi
	cd $(ROOT_DIR) && . .venv/bin/activate && python transcribe.py

# Clean up generated files but keep models and .gitkeep files
clean:
	cd $(ROOT_DIR) && find tmp/output -type f ! -name '.gitkeep' -delete
	cd $(ROOT_DIR) && find tmp/transcriptions -type f ! -name '.gitkeep' -delete
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	cd $(ROOT_DIR) && rm -rf .pytest_cache/
	cd $(ROOT_DIR) && rm -rf .coverage
	cd $(ROOT_DIR) && rm -rf .mypy_cache/

# Clean everything including models
clean-all: clean
	cd $(ROOT_DIR) && find tmp/models -type f ! -name '.gitkeep' -delete

# Install dependencies
install:
	cd $(ROOT_DIR) && . .venv/bin/activate && pip install -e .

# Run all tests
test:
	cd $(ROOT_DIR) && . .venv/bin/activate && python -m pytest tests/ -v

# Check code style
lint:
	cd $(ROOT_DIR) && . .venv/bin/activate && flake8 .
	cd $(ROOT_DIR) && . .venv/bin/activate && mypy .

# Setup whisper and download sample files
setup-whisper:
	cd $(ROOT_DIR) && . .venv/bin/activate && python tools/setup_whisper.py --model large-v3 --samples
