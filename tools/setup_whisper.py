#!/usr/bin/env python3
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return True if successful."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    return True

def setup_whisper(model_name):
    """Set up whisper.cpp and download model."""
    root_dir = Path(__file__).parent.parent
    whisper_dir = root_dir / 'deps' / 'whisper.cpp'

    print("Initializing whisper.cpp submodule...")
    if not run_command(['git', 'submodule', 'update', '--init', '--recursive'], cwd=root_dir):
        return False

    print("\nBuilding whisper.cpp...")
    if not run_command(['make'], cwd=whisper_dir):
        return False

    print("\nDownloading model...")
    if not run_command(['./models/download-ggml-model.sh', model_name], cwd=whisper_dir):
        return False

    # Create tmp directories if they don't exist
    tmp_dir = root_dir / 'tmp'
    output_dir = tmp_dir / 'output'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download sample files
    print("\nDownloading sample files...")
    samples_dir = whisper_dir / 'samples'
    if not samples_dir.exists():
        if not run_command(['bash', 'download-jfk.sh'], cwd=samples_dir):
            return False

    return True

def main():
    parser = argparse.ArgumentParser(description='Setup whisper.cpp and download model')
    parser.add_argument('--model', default='large-v3', help='Model to download')
    parser.add_argument('--samples', action='store_true', help='Download sample files')
    args = parser.parse_args()

    if setup_whisper(args.model):
        print("\nSetup completed successfully!")
    else:
        print("\nSetup failed!")

if __name__ == '__main__':
    main()
