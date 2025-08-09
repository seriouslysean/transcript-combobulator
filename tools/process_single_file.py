
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
        from src.config import get_output_path_for_input, INPUT_DIR, OUTPUT_DIR
        print(f"Input file: {input_file}")
        print(f"Input DIR: {INPUT_DIR}")
        print(f"Input relative: {input_file.relative_to(INPUT_DIR) if input_file.is_relative_to(INPUT_DIR) else 'Not relative'}")
        output_dir = get_output_path_for_input(input_file)
        print(f"Output dir: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Place the converted WAV file inside the output directory with the same basename as the input (no _16khz)
        output_file = output_dir / f"{input_file.stem}.wav"
        print(f"Output file: {output_file}")
        convert_to_wav(input_file, output_file)
    else:
        # If no conversion is needed, copy the file to the output directory with .wav extension if not already
        from src.config import get_output_path_for_input
        output_dir = get_output_path_for_input(input_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{input_file.stem}.wav"
        if not output_file.exists():
            import shutil
            shutil.copy(input_file, output_file)
        print("No conversion needed, copied to output directory")
    print(f"Step 2: Processing VAD on {output_file.name}...")
    output_dir, segments = process_audio(output_file)
    print(f"Step 3: Transcribing segments...")
    transcribe_segments(output_file, input_file)
    print("Step 4: All processing complete for this file")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: process_single_file.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
