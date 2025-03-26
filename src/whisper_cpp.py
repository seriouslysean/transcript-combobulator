"""Whisper.cpp integration for audio transcription."""

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from datetime import timedelta

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
        output_path: Optional path to save the VTT file
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

    # If no output path specified, use the same name as input but with .vtt extension
    if output_path is None:
        output_path = audio_path.with_suffix('.vtt')

    # Build whisper.cpp command
    cmd = [
        str(whisper_cli),
        '-m', str(WHISPER_MODEL_PATH),
        '-f', str(audio_path),
        '--output-vtt',
        '--print-colors',
        '--output-file', str(output_path.with_suffix('')),
        '--beam-size', str(WHISPER_BEAM_SIZE),
        '--entropy-thold', str(WHISPER_ENTROPY_THOLD),
        '--temperature', str(WHISPER_TEMPERATURE),
        '--max-context', str(WHISPER_MAX_CONTEXT),
        '--word-thold', str(WHISPER_WORD_THOLD),
        '--no-fallback',
        '--language', WHISPER_LANGUAGE,
        '--prompt', WHISPER_PROMPT
    ]

    try:
        # Run whisper.cpp
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise WhisperCppError(f"whisper.cpp failed: {result.stderr}")

        # Parse the VTT file
        segments = parse_vtt_file(output_path)

        # Add offset to timestamps
        for segment in segments:
            segment['start'] += offset
            segment['end'] += offset

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
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")

    return all_segments
