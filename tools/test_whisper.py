#!/usr/bin/env python3
"""Test Whisper transcription."""

import json
import statistics
from pathlib import Path
import sys
from typing import Dict, Any, List

from src.whisper_cpp import transcribe_segment
from src.config import OUTPUT_DIR

def format_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def process_file(audio_path: Path, show_transcription: bool = False) -> Dict[str, Any]:
    """Process a single file and return stats."""

    # Create test output directory
    test_output_dir = audio_path.parent / 'whisper_test_output'
    test_output_dir.mkdir(exist_ok=True)

    print(f"\nProcessing: {audio_path.name}")
    if show_transcription:
        print("-" * 40)

    try:
        # Transcribe the segment
        segments = transcribe_segment(audio_path)

        # Save results as JSON
        result = {
            'file': audio_path.name,
            'segments': segments,
            'stats': {
                'num_segments': len(segments),
                'total_duration': sum(s['end'] - s['start'] for s in segments),
                'avg_segment_duration': statistics.mean(s['end'] - s['start'] for s in segments) if segments else 0
            }
        }

        json_output = test_output_dir / f"{audio_path.stem}_test.json"
        with open(json_output, 'w') as f:
            json.dump(result, f, indent=2)

        # Generate VTT output
        vtt_output = test_output_dir / f"{audio_path.stem}_test.vtt"
        with open(vtt_output, 'w') as f:
            f.write('WEBVTT\n\n')
            for segment in segments:
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")

        if show_transcription:
            print("\nTranscription:")
            for segment in segments:
                print(f"\n[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}]")
                print(f"Text: {segment['text']}")

        print(f"\nResults saved to:")
        print(f"JSON: {json_output}")
        print(f"VTT: {vtt_output}")

        return result

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    """Main entry point."""
    # Look for segment files in output directory
    output_dir = Path('tmp/output')
    if not output_dir.exists():
        print("Error: Output directory not found")
        sys.exit(1)

    if len(sys.argv) > 1:
        # Use provided filename as a pattern to search in output directory
        pattern = sys.argv[1]
        audio_files = list(output_dir.glob(f'*{pattern}*_segment_*.wav'))
        if not audio_files:
            print(f"No segment files found matching pattern '{pattern}' in output directory")
            sys.exit(1)
        show_transcription = True  # Show detailed output for specific files
    else:
        # Process all files in output directory
        audio_files = list(output_dir.glob('*_segment_*.wav'))
        if not audio_files:
            print("No segment files found in output directory")
            sys.exit(1)
        show_transcription = False  # Only show summary for bulk processing

    print(f"\nFound {len(audio_files)} segment file(s) to analyze")
    print("-" * 80)

    # Process each file
    results = []
    for audio_path in audio_files:
        result = process_file(audio_path, show_transcription)
        if result:
            results.append(result)

    # Generate summary report
    if results:
        total_segments = sum(r['stats']['num_segments'] for r in results)
        total_duration = sum(r['stats']['total_duration'] for r in results)
        avg_duration = statistics.mean(r['stats']['avg_segment_duration'] for r in results)

        print("\nSummary:")
        print(f"Total files processed: {len(results)}")
        print(f"Total segments: {total_segments}")
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Average segment duration: {avg_duration:.2f}s")

if __name__ == '__main__':
    main()
