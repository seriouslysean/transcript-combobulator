"""Whisper.cpp integration for audio transcription."""

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from datetime import timedelta
import os
import json

from src.config import (
    WHISPER_CPP_PATH,
    WHISPER_MODEL_PATH,
    WHISPER_PROMPT,
    WHISPER_BEAM_SIZE,
    WHISPER_ENTROPY_THOLD,
    WHISPER_MAX_CONTEXT,
    WHISPER_TEMPERATURE,
    WHISPER_WORD_THOLD,
    WHISPER_LANGUAGE,
)

class WhisperCppError(Exception):
    """Base exception for whisper.cpp-related errors."""
    pass

def parse_timestamp(timestamp: str) -> float:
    """Parse timestamp format into seconds.

    Handles both VTT format (HH:MM:SS.mmm) and JSON format (HH:MM:SS,mmm).
    """
    try:
        # Try VTT format first (HH:MM:SS.mmm)
        h, m, s = timestamp.split(':')
        s, ms = s.split('.')
        return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000
    except ValueError:
        try:
            # Try JSON format (HH:MM:SS,mmm)
            h, m, s = timestamp.split(':')
            s, ms = s.split(',')
            return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000
        except ValueError:
            # If no milliseconds, treat as whole seconds
            h, m, s = timestamp.split(':')
            return float(h) * 3600 + float(m) * 60 + float(s)

def format_timestamp(seconds: float) -> str:
    """Format seconds into VTT timestamp format (HH:MM:SS.mmm)."""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def parse_vtt_timestamp(timestamp: str) -> float:
    """Parse VTT timestamp format (HH:MM:SS.mmm) into seconds."""
    h, m, s = timestamp.split(':')
    s, ms = s.split('.')
    return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000

def parse_vtt_file(vtt_path: Path) -> List[Dict[str, Any]]:
    """Parse a VTT file into a list of segments with timestamps and text."""
    segments = []
    current_segment = None

    with open(vtt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip WEBVTT header
            if line == 'WEBVTT':
                continue

            # Parse timestamp line
            if '-->' in line:
                if current_segment:
                    segments.append(current_segment)
                start, end = line.split(' --> ')
                current_segment = {
                    'start': parse_vtt_timestamp(start),
                    'end': parse_vtt_timestamp(end),
                    'text': ''
                }
            # Parse text line
            elif current_segment:
                current_segment['text'] += line + ' '

    if current_segment:
        segments.append(current_segment)

    return segments

def transcribe_segment(
    audio_path: Path,
    output_path: Optional[Path] = None,
    offset: float = 0.0
) -> List[Dict[str, Any]]:
    """Transcribe a single audio segment using whisper.cpp.

    Args:
        audio_path: Path to the audio file to transcribe
        output_path: Optional path to save the VTT file (only used for final output)
        offset: Time offset in seconds to add to timestamps

    Returns:
        List of segments with timestamps and text

    Raises:
        WhisperCppError: If transcription fails
    """
    if not audio_path.exists():
        raise WhisperCppError(f"Audio file not found: {audio_path}")

    if not WHISPER_MODEL_PATH.exists():
        raise WhisperCppError(f"Whisper model not found: {WHISPER_MODEL_PATH}")

    if not WHISPER_CPP_PATH.exists():
        raise WhisperCppError(f"Whisper.cpp binary not found: {WHISPER_CPP_PATH}")

    whisper_cli = WHISPER_CPP_PATH / 'whisper-cli'
    if not whisper_cli.exists():
        raise WhisperCppError(f"whisper-cli not found in {WHISPER_CPP_PATH}")

    # Build whisper.cpp command
    cmd = [
        str(whisper_cli),
        '-m', str(WHISPER_MODEL_PATH),
        '-f', str(audio_path),
        '--output-json',  # Output JSON instead of VTT
        '--print-colors',
        '--output-file', str(audio_path.with_suffix('')),  # Use audio path as base
        '--beam-size', str(WHISPER_BEAM_SIZE),
        '--entropy-thold', str(WHISPER_ENTROPY_THOLD),
        '--temperature', str(WHISPER_TEMPERATURE),
        '--max-context', str(WHISPER_MAX_CONTEXT),
        '--word-thold', str(WHISPER_WORD_THOLD),
        '--no-fallback',
        '--language', WHISPER_LANGUAGE
    ]

    # Only add prompt if it's not empty
    if WHISPER_PROMPT:
        cmd.extend(['--prompt', WHISPER_PROMPT])

    try:
        # Run whisper.cpp
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise WhisperCppError(f"whisper.cpp failed: {result.stderr}")

        # Parse the JSON output
        temp_json = audio_path.with_suffix('.json')
        with open(temp_json) as f:
            json_data = json.load(f)

        # Clean up temporary JSON file
        temp_json.unlink()

        # Extract segments and add offset to timestamps
        segments = []
        for segment in json_data['transcription']:
            # Convert timestamp to seconds
            start = parse_timestamp(segment['timestamps']['from'])
            end = parse_timestamp(segment['timestamps']['to'])

            segments.append({
                'start': start + offset,
                'end': end + offset,
                'text': segment['text'].strip()
            })

        return segments

    except subprocess.CalledProcessError as e:
        raise WhisperCppError(f"Failed to run whisper.cpp: {e}") from e
    except Exception as e:
        raise WhisperCppError(f"Failed to transcribe audio: {e}") from e

def transcribe_audio_segments(
    segments: List[Dict[str, Any]],
    output_path: Path
) -> List[Dict[str, Any]]:
    """Transcribe multiple audio segments and combine their results.

    Args:
        segments: List of segments with audio paths and timestamps
        output_path: Path to save the final VTT file

    Returns:
        List of all transcribed segments with timestamps and text
    """
    all_segments = []

    for i, segment in enumerate(segments, 1):
        audio_path = Path(segment['audio_path'])
        offset = segment['start']
        print(f"Transcribing segment {i}/{len(segments)}")

        # Transcribe this segment
        segment_results = transcribe_segment(audio_path, offset=offset)
        all_segments.extend(segment_results)

    # Sort segments by start time
    all_segments.sort(key=lambda x: x['start'])

    # Write combined VTT file
    with open(output_path, 'w') as f:
        f.write('WEBVTT\n\n')
        for segment in all_segments:
            # Format timestamps in VTT format (HH:MM:SS.mmm)
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text']}\n\n")

    return all_segments

def transcribe_audio(audio_path: Path, output_path: Path, prompt: str = "") -> None:
    """Transcribe audio file using whisper.cpp."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get whisper settings from environment
    beam_size = int(os.getenv('WHISPER_BEAM_SIZE', '8'))
    entropy_thold = float(os.getenv('WHISPER_ENTROPY_THOLD', '2.4'))
    temperature = float(os.getenv('WHISPER_TEMPERATURE', '0.0'))
    max_context = int(os.getenv('WHISPER_MAX_CONTEXT', '128'))
    word_thold = float(os.getenv('WHISPER_WORD_THOLD', '0.6'))
    confidence_threshold = int(os.getenv('WHISPER_CONFIDENCE_THRESHOLD', '50'))

    # Build whisper.cpp command
    cmd = [
        'deps/whisper.cpp/build/bin/whisper-cli',
        '-m', os.getenv('WHISPER_MODEL_PATH', 'deps/whisper.cpp/models/ggml-large-v3.bin'),
        '-f', str(audio_path),
        '--output-vtt',
        '--print-colors',
        '--print-special',
        '--output-file', str(output_path.with_suffix('')),
        '--beam-size', str(beam_size),
        '--entropy-thold', str(entropy_thold),
        '--temperature', str(temperature),
        '--max-context', str(max_context),
        '--word-thold', str(word_thold),
        '--no-fallback',
        '--language', os.getenv('WHISPER_LANGUAGE', 'en')
    ]

    # Add prompt if provided
    if prompt:
        cmd.extend(['--prompt', prompt])

    # Run whisper.cpp
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"whisper.cpp failed: {result.stderr}")

        # Filter out low-confidence transcriptions
        if confidence_threshold > 0:
            with open(output_path) as f:
                content = f.read()

            # Parse VTT content and filter by confidence
            filtered_content = []
            current_segment = []
            for line in content.split('\n'):
                if line.strip() and not line.startswith('WEBVTT'):
                    if '-->' in line:
                        if current_segment:
                            # Check confidence of previous segment
                            confidence = 0
                            for seg_line in current_segment:
                                if '[_TT_' in seg_line:
                                    match = re.search(r'\[_TT_(\d+)\]', seg_line)
                                    if match:
                                        confidence = int(match.group(1))
                                        break

                            # Only keep segment if confidence is above threshold
                            if confidence >= confidence_threshold:
                                filtered_content.extend(current_segment)
                            current_segment = []
                    current_segment.append(line)

            # Add any remaining segment
            if current_segment:
                confidence = 0
                for seg_line in current_segment:
                    if '[_TT_' in seg_line:
                        match = re.search(r'\[_TT_(\d+)\]', seg_line)
                        if match:
                            confidence = int(match.group(1))
                            break

                if confidence >= confidence_threshold:
                    filtered_content.extend(current_segment)

            # Write filtered content back to file
            with open(output_path, 'w') as f:
                f.write('WEBVTT\n\n')
                f.write('\n'.join(filtered_content))

    except Exception as e:
        raise RuntimeError(f"Failed to transcribe audio: {e}")
