#!/usr/bin/env python3
from pathlib import Path
from src.process_audio import process_audio, ProcessingError
import json
from src.config import INPUT_DIR, OUTPUT_DIR

def main():
    """Process JFK sample file from whisper.cpp samples."""
    input_file = Path('deps/whisper.cpp/samples/jfk.wav')
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run 'make setup-whisper' to download sample files.")
        return

    print(f"\nProcessing {input_file.name}...")

    try:
        # Process audio with VAD
        processed_path = process_audio(input_file)
        print(f"✓ Audio processed: {processed_path.name}")

        # Load the mapping file
        mapping_file = OUTPUT_DIR / f"{input_file.stem}_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            segments = mapping.get('segments', [])
            print(f"Found {len(segments)} segments in {mapping_file.name}")

            # Log where whisper.cpp would run
            for i, segment in enumerate(segments):
                segment_wav = Path(segment['segment_file'])
                print(f"[WOULD RUN] whisper.cpp on segment {i}: {segment_wav}")
        else:
            print(f"No mapping file found at {mapping_file}")

    except ProcessingError as e:
        print(f"Error processing {input_file.name}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {input_file.name}: {e}")

    print(f"Completed {input_file.name}")

if __name__ == '__main__':
    main()
