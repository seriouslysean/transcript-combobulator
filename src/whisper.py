"""Whisper integration for audio transcription."""

import whisper
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from datetime import timedelta
import json
import os
import torch
import sys

from src.logging_config import get_logger
logger = get_logger(__name__)

from src.config import (
    WHISPER_PROMPT,
    WHISPER_TEMPERATURE,
    WHISPER_LANGUAGE,
    OUTPUT_DIR,
    TRANSCRIPTIONS_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_MODELS_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_LOGPROB_THRESHOLD,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_WORD_TIMESTAMPS,
    WHISPER_CONDITION_ON_PREVIOUS,
    WHISPER_CONFIDENCE_THRESHOLD,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_DURATION,
    VAD_MIN_SILENCE_DURATION,
    PADDING_SECONDS,
)

class WhisperError(Exception):
    """Base exception for whisper-related errors."""
    pass

def get_whisper_config() -> Dict[str, str]:
    """Get whisper configuration from environment variables.

    Returns:
        Dict[str, str]: Configuration dictionary with device
    """
    # Get device from environment
    device = os.getenv('WHISPER_DEVICE', 'cpu')
    if device not in ['cpu', 'cuda']:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

    return {'device': device}

def format_timestamp(seconds: float) -> str:
    """Format seconds into VTT timestamp format (HH:MM:SS.mmm)."""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def transcribe_segment(
    audio_path: Path,
    output_path: Optional[Path] = None,
    offset: float = 0.0,
    model: Optional[whisper.Whisper] = None
) -> List[Dict[str, Any]]:
    """Transcribe a single audio segment using Whisper.

    Args:
        audio_path: Path to the audio file to transcribe
        output_path: Optional path to save the VTT file (only used for final output)
        offset: Time offset in seconds to add to timestamps
        model: Optional pre-loaded Whisper model to use

    Returns:
        List of segments with timestamps and text

    Raises:
        WhisperError: If transcription fails
    """
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    try:
        # Get whisper configuration
        config = get_whisper_config()

        # Load model if not provided
        if model is None:
            model = whisper.load_model(
                WHISPER_MODEL,
                device=config['device'],
                download_root=str(WHISPER_MODELS_DIR)
            )

        # Transcribe the audio
        result = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            temperature=WHISPER_TEMPERATURE,
            initial_prompt=WHISPER_PROMPT,
            word_timestamps=False,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            fp16=False  # Force FP32
        )

        # Process segments
        segments = []
        for segment in result["segments"]:
            # Add offset to timestamps
            start = segment["start"] + offset
            end = segment["end"] + offset

            # Get text and calculate confidence from log probability
            text = segment["text"].strip()
            avg_logprob = segment.get('avg_logprob', 0)
            confidence = min(100, max(0, (1 + avg_logprob) * 100))

            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "confidence": confidence
            })

        # Write VTT file if output path is provided
        if output_path and segments:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in segments:
                    if segment["text"].strip():  # Only write non-empty segments
                        start_time = format_timestamp(segment["start"])
                        end_time = format_timestamp(segment["end"])
                        f.write(f"{start_time} --> {end_time}\n{segment['text'].strip()}\n\n")

        return segments

    except Exception as e:
        raise WhisperError(f"Failed to transcribe segment: {e}") from e

def transcribe_audio_segments(
    segments: List[tuple[Path, float]],
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Transcribe multiple audio segments in sequence.

    Args:
        segments: List of tuples containing (segment_path, start_time)
        output_path: Optional path to save the combined VTT file

    Returns:
        List of transcribed segments with timestamps and text
    """
    total_segments = len(segments)
    logger.info(f"Loading Whisper model...")
    
    # Load model once for all segments
    model = load_whisper_model()
    
    logger.info(f"VAD found {total_segments} segments")

    # Create individual VTT files for each segment, then combine
    individual_vtts = []
    all_segments = []
    
    for i, (segment_path, start_time) in enumerate(segments, 1):
        logger.info(f"Processing segment {i:03d}/{total_segments}...")
        try:
            # Create individual VTT file for this segment
            segment_vtt_path = segment_path.with_suffix('.vtt')
            
            # Transcribe segment and save to individual VTT
            segments_result = transcribe_segment(segment_path, segment_vtt_path, start_time, model)
            individual_vtts.append(segment_vtt_path)
            all_segments.extend(segments_result)
            
        except Exception as e:
            logger.warning(f"Failed to transcribe segment {segment_path}: {e}")
            continue

    # Combine all individual VTT files into final output
    if output_path and individual_vtts:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as combined_file:
            combined_file.write("WEBVTT\n\n")
            
            for vtt_path in individual_vtts:
                if vtt_path.exists():
                    with open(vtt_path, "r", encoding="utf-8") as individual_file:
                        content = individual_file.read()
                        # Skip the WEBVTT header and add content
                        if content.startswith("WEBVTT"):
                            content = content[6:].lstrip()
                        combined_file.write(content)
                        if not content.endswith("\n\n"):
                            combined_file.write("\n")
        
        logger.info(f"User transcript saved: {output_path.name}")
        
        # Clean up individual VTT files
        for vtt_path in individual_vtts:
            if vtt_path.exists():
                vtt_path.unlink()

    # Sort segments by start time for return value
    all_segments.sort(key=lambda x: x["start"])
    return all_segments

def filter_by_confidence(segments: List[Dict[str, Any]], confidence_threshold: float = 50.0) -> List[Dict[str, Any]]:
    """Filter segments by confidence score.

    Args:
        segments: List of segments with confidence scores
        confidence_threshold: Minimum confidence score (0-100)

    Returns:
        List of segments that meet the confidence threshold
    """
    return [segment for segment in segments if segment.get("confidence", 0.0) >= confidence_threshold]

def transcribe_audio(audio_path: Path, output_path: Path, prompt: str = "") -> List[Dict[str, Any]]:
    """Transcribe audio file directly using Whisper without VAD.

    Args:
        audio_path: Path to the audio file to transcribe
        output_path: Path to save the VTT file
        prompt: Optional prompt to guide transcription

    Returns:
        List of transcribed segments with timestamps and text

    Raises:
        WhisperError: If transcription fails
    """
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    try:
        # Get whisper configuration
        config = get_whisper_config()

        # Load model
        model = whisper.load_model(
            WHISPER_MODEL,
            device=config['device'],
            download_root=str(WHISPER_MODELS_DIR)
        )

        # Transcribe the audio
        result = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            temperature=WHISPER_TEMPERATURE,
            initial_prompt=prompt or WHISPER_PROMPT,
            word_timestamps=False,
            condition_on_previous_text=True,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            fp16=False  # Force FP32
        )

        # Process segments
        segments = []
        for segment in result["segments"]:
            text = segment["text"].strip()
            avg_logprob = segment.get('avg_logprob', 0)
            confidence = min(100, max(0, (1 + avg_logprob) * 100))

            # Log the transcribed text
            if text:
                logger.info(f"  [{format_timestamp(segment['start'])} -> {format_timestamp(segment['end'])}] {text}")

            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
                "confidence": confidence
            })

        # Write VTT file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{start_time} --> {end_time}\n{text}\n\n")

        return segments

    except Exception as e:
        raise WhisperError(f"Failed to transcribe audio: {e}") from e

def regenerate_vtt_with_confidence(
    json_path: Path,
    output_vtt: Path,
    confidence_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Regenerate VTT file from JSON with optional confidence filtering.

    Args:
        json_path: Path to the JSON file containing segments
        output_vtt: Path to save the VTT file
        confidence_threshold: Optional confidence threshold (0-100)

    Returns:
        List of segments that meet the confidence threshold

    Raises:
        WhisperError: If regeneration fails
    """
    if not json_path.exists():
        raise WhisperError(f"JSON file not found: {json_path}")

    try:
        # Load segments from JSON
        with open(json_path) as f:
            segments = json.load(f)

        # Filter by confidence if threshold is provided
        if confidence_threshold is not None:
            segments = filter_by_confidence(segments, confidence_threshold)

        # Write VTT file
        output_vtt.parent.mkdir(parents=True, exist_ok=True)
        with open(output_vtt, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{start_time} --> {end_time}\n{text}\n\n")

        return segments

    except Exception as e:
        raise WhisperError(f"Failed to regenerate VTT: {e}") from e

def regenerate_vtt_for_audio(
    audio_path: Path,
    confidence_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Regenerate VTT file for an audio file with optional confidence filtering.

    Args:
        audio_path: Path to the audio file
        confidence_threshold: Optional confidence threshold (0-100)

    Returns:
        List of segments that meet the confidence threshold

    Raises:
        WhisperError: If regeneration fails
    """
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    # Define paths
    json_path = audio_path.with_suffix(".json")
    output_vtt = audio_path.with_suffix(".vtt")

    # First transcribe the audio to get JSON
    segments = transcribe_audio(audio_path, output_vtt)

    # Save segments to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    # Regenerate VTT with confidence filtering if threshold is provided
    if confidence_threshold is not None:
        segments = regenerate_vtt_with_confidence(json_path, output_vtt, confidence_threshold)

    return segments

def load_whisper_model(model_name: str = None) -> whisper.Whisper:
    """Load whisper model with proper configuration.

    Args:
        model_name: Optional model name to use. If None, uses WHISPER_MODEL env var
                   or falls back to large-v2

    Returns:
        whisper.Whisper: Loaded model

    Raises:
        WhisperError: If model file doesn't exist
    """
    if not model_name:
        model_name = os.getenv('WHISPER_MODEL', 'large-v2')

    # Check if model file exists
    model_path = WHISPER_MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise WhisperError(f"Model file not found: {model_path}. Please run setup-whisper first.")

    # Set WHISPER_MODELS_DIR environment variable
    os.environ['WHISPER_MODELS_DIR'] = str(WHISPER_MODELS_DIR)

    # Get configuration
    config = get_whisper_config()

    # Load model
    model = whisper.load_model(
        model_name,
        device=config['device'],
        download_root=str(WHISPER_MODELS_DIR)
    )

    return model


def transcribe_segments(audio_segments: List[Dict[str, Any]], model_name: str = None) -> List[Dict[str, Any]]:
    """Transcribe audio segments using Whisper.

    Args:
        audio_segments: List of audio segment dictionaries with paths and metadata
        model_name: Optional model name to use

    Returns:
        List[Dict[str, Any]]: List of transcribed segments with text and metadata
    """
    if not audio_segments:
        return []

    try:
        # Load model once for all segments
        model = load_whisper_model(model_name)

        # Process each segment
        results = []
        for segment in audio_segments:
            # Transcribe the segment
            result = model.transcribe(
                str(segment['path']),
                language=WHISPER_LANGUAGE,
                temperature=WHISPER_TEMPERATURE,
                initial_prompt=WHISPER_PROMPT if WHISPER_PROMPT else None,
                word_timestamps=False,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=1.2,
                fp16=False  # Force FP32
            )

            # Calculate confidence score from logprobs
            if 'segments' in result and result['segments']:
                # Average probability across all segments
                avg_logprob = sum(s.get('avg_logprob', 0) for s in result['segments']) / len(result['segments'])
                # Convert log probability to confidence percentage (0-100)
                confidence = min(100, max(0, (1 + avg_logprob) * 100))
            else:
                confidence = 0

            # Add transcription and confidence to segment data
            segment['text'] = result['text'].strip()
            segment['confidence'] = confidence
            results.append(segment)

        return results

    except Exception as e:
        raise WhisperError(f"Error transcribing segments: {e}")
