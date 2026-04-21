.PHONY: setup run run-single clean clean-all test setup-whisper install lint \
        create-sample-files test-segment regenerate-vtt process-vad \
        transcribe-segments create-test-files combine-transcripts convert-audio

ROOT_DIR := $(shell pwd)

# Run commands inside the virtualenv with our src/ on sys.path
VENV_CMD = cd $(ROOT_DIR) && . .venv/bin/activate && PYTHONPATH=$(ROOT_DIR)
PY = $(VENV_CMD) ENV_FILE=$(ENV_FILE) python

.SILENT:

# ── Setup ──
setup:
	if ! command -v pyenv &> /dev/null; then \
		echo "pyenv is not installed. Please install it first."; \
		exit 1; \
	fi
	cd $(ROOT_DIR) && pyenv install --skip-existing 3.10
	cd $(ROOT_DIR) && python3.10 -m venv .venv
	$(VENV_CMD) pip install --upgrade pip
	$(VENV_CMD) pip install -e ".[dev]"
	$(MAKE) setup-whisper

install:
	$(VENV_CMD) pip install -e ".[dev]"

setup-whisper:
	$(PY) tools/setup_whisper.py

# ── Primary pipeline ──
# Full pipeline: single file or folder (defaults to tmp/input/)
run:
	if [ -n "$(file)" ]; then \
		if [ ! -f "$(file)" ]; then echo "File not found: $(file)"; exit 1; fi; \
		echo "Processing single file: $$(basename $(file))..."; \
		$(MAKE) run-single file=$(file); \
		$(MAKE) combine-transcripts ENV_FILE=$(ENV_FILE); \
	elif [ -n "$(folder)" ]; then \
		if [ ! -d "$(folder)" ]; then echo "Directory not found: $(folder)"; exit 1; fi; \
		session_name=$$(basename "$(folder)"); \
		$(PY) tools/process_batch.py "$(folder)" --session "$$session_name"; \
	else \
		target_dir="$(ROOT_DIR)/tmp/input"; \
		if [ ! -d "$$target_dir" ]; then echo "Directory not found: $$target_dir"; exit 1; fi; \
		$(PY) tools/process_batch.py "$$target_dir"; \
	fi

# Convert -> VAD -> transcribe for one file (no combine)
run-single:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make run-single file=path/to/file.wav"; \
		exit 1; \
	fi
	echo "Processing $$(basename $(file))..."
	$(PY) tools/process_single_file.py $(file)

# Just the VAD step
process-vad:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make process-vad file=path/to/file.wav"; \
		exit 1; \
	fi
	echo "Processing $$(basename $(file)) with VAD..."
	$(PY) -c "from pathlib import Path; from src.vad import process_audio; process_audio(Path('$(file)'))"

# Just the transcribe step (requires mapping JSON from a prior VAD run)
transcribe-segments:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make transcribe-segments file=path/to/file.wav"; \
		exit 1; \
	fi
	echo "Transcribing segments for $$(basename $(file))..."
	$(PY) -c "from pathlib import Path; from src.transcribe import transcribe_segments; transcribe_segments(Path('$(file)'))"

# Just the combine step
combine-transcripts:
	echo "Combining all transcripts using environment configuration..."
	if [ -n "$(session)" ]; then \
		$(PY) -c "from pathlib import Path; from src.combine import combine_transcripts_from_env; combine_transcripts_from_env(Path('$(ROOT_DIR)/tmp/output'), '$(session)')"; \
	else \
		$(PY) -c "from pathlib import Path; from src.combine import combine_transcripts_from_env; combine_transcripts_from_env(Path('$(ROOT_DIR)/tmp/output'))"; \
	fi

# Convert audio to 16kHz mono WAV (walks tmp/input/ if no input= given)
convert-audio:
	if [ -z "$(input)" ]; then \
		echo "Converting all audio files in tmp/input/ to 16kHz WAV format..."; \
		$(PY) tools/convert_audio.py; \
	else \
		echo "Converting $(input) to 16kHz WAV format..."; \
		$(PY) tools/convert_audio.py --input "$(input)"; \
	fi

# Re-run whisper on a file with a confidence threshold; rewrite its VTT
regenerate-vtt:
	if [ -z "$(file)" ]; then \
		echo "No file specified. Usage: make regenerate-vtt file=path/to/file.wav [threshold=50]"; \
		exit 1; \
	fi
	if [ -z "$(threshold)" ]; then \
		$(PY) -c "from pathlib import Path; from src.whisper import regenerate_vtt_for_audio; regenerate_vtt_for_audio(Path('$(file)'))"; \
	else \
		$(PY) -c "from pathlib import Path; from src.whisper import regenerate_vtt_for_audio; regenerate_vtt_for_audio(Path('$(file)'), confidence_threshold=$(threshold))"; \
	fi

# ── Test / dev helpers ──
create-sample-files:
	if [ ! -d "$(ROOT_DIR)/samples" ]; then \
		echo "Samples directory not found at samples/"; \
		exit 1; \
	fi
	$(PY) tools/create_sample_files.py

create-test-files:
	if [ ! -d "$(ROOT_DIR)/samples" ]; then \
		echo "Samples directory not found at samples/"; \
		exit 1; \
	fi
	$(PY) tools/create_sample_files.py --prefix test_ --copies 1 --padded-copies 3

test:
	$(VENV_CMD) python -m pytest tests/ -v
	$(MAKE) clean

# Lint: mypy only (flake8 is not a dependency).
# Type annotations are not enforced repo-wide; mypy will warn on obvious bugs.
lint:
	$(VENV_CMD) mypy src tools

test-segment:
	@if [ "$(file)" = "" ]; then \
		$(VENV_CMD) python tools/test_whisper.py; \
	else \
		$(VENV_CMD) python tools/test_whisper.py $(file); \
	fi

# ── Cleanup ──
clean:
	cd $(ROOT_DIR) && find tmp -type f -not -path "tmp/input/*" -not -name ".gitkeep" -delete
	cd $(ROOT_DIR) && find tmp -type d -empty -not -path "tmp/input" -delete
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	cd $(ROOT_DIR) && rm -rf .pytest_cache/ .coverage .mypy_cache/

clean-all:
	cd $(ROOT_DIR) && find tmp -type f -not -name ".gitkeep" -delete
	cd $(ROOT_DIR) && find tmp -type d -empty -not -path "tmp/input" -delete
	cd $(ROOT_DIR) && mkdir -p tmp/input tmp/output tmp/transcriptions
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	cd $(ROOT_DIR) && rm -rf .pytest_cache/ .coverage .mypy_cache/
