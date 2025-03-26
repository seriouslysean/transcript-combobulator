"""Audio processing utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.vad import process_audio as process_vad, VADError
from src.config import OUTPUT_DIR, SAMPLE_RATE

class ProcessingError(Exception):
    """Base exception for audio processing errors."""
    pass

def process_audio(input_path: Path) -> Path:
    """Process audio file and save mapping information.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Path: Path to the processed audio file.

    Raises:
        ProcessingError: If processing fails.
    """
    try:
        output_path, processed_segments = process_vad(input_path)

        # Save timestamp mapping
        mapping = {
            'original_file': str(input_path),
            'processed_file': str(output_path),
            'sample_rate': SAMPLE_RATE,
            'segments': processed_segments,
            'created_at': datetime.now().isoformat()
        }

        mapping_path = output_path.parent / f"{input_path.stem}_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        return output_path

    except VADError as e:
        raise ProcessingError(f"VAD processing failed: {e}") from e
    except Exception as e:
        raise ProcessingError(f"Failed to process audio: {e}") from e
