import sys
from pathlib import Path
from src.audio_utils import needs_conversion, convert_to_wav
from src.process_audio import process_audio
from src.transcribe import transcribe_segments
from src.logging_config import setup_logging

def main(input_path):
    input_file = Path(input_path).resolve()
    input_root = Path('tmp/input').resolve()
    output_root = Path('tmp/output').resolve()
    setup_logging()
    print(f"Step 1: Converting {input_file.name} if needed...")
    if needs_conversion(input_file):
        rel_path = input_file.relative_to(input_root) if input_file.is_relative_to(input_root) else Path(input_file.name)
        output_file = output_root / rel_path.parent / f"{input_file.stem}_16khz.wav"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        convert_to_wav(input_file, output_file)
    else:
        output_file = input_file
        print("No conversion needed")
    print(f"Step 2: Processing VAD on {output_file.name}...")
    process_audio(output_file)
    print(f"Step 3: Transcribing segments...")
    transcribe_segments(output_file)
    print("Step 4: All processing complete for this file")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: process_single_file.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
