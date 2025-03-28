"""Audio processing utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.vad import process_audio as process_vad, VADError
from src.config import OUTPUT_DIR, SAMPLE_RATE

class ProcessingError(Exception):
    """Base exception for audio processing errors."""
    pass

def process_audio(input_path: Path) -> tuple[Path, List[Dict[str, Any]]]:
    """Process audio file and save mapping information.

    Args:
        input_path: Path to the input audio file.

    Returns:
        tuple[Path, List[Dict[str, Any]]]: Path to output directory and list of segments.

    Raises:
        ProcessingError: If processing fails.
    """
    try:
        output_dir, processed_segments = process_vad(input_path)

        # Save timestamp mapping - preserve test_ prefix if present
        stem = input_path.stem
        mapping = {
            'original_file': str(input_path),
            'sample_rate': SAMPLE_RATE,
            'segments': processed_segments,
            'created_at': datetime.now().isoformat()
        }

        mapping_path = output_dir / f"{stem}_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        return output_dir, processed_segments

    except VADError as e:
        raise ProcessingError(f"VAD processing failed: {e}") from e
    except Exception as e:
        raise ProcessingError(f"Failed to process audio: {e}") from e
