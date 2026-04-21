"""Transcript combination utilities.

Reads per-user VTT files, tags each line with a speaker label, sorts by
timestamp, deduplicates, and writes a combined session transcript.
"""

import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import OUTPUT_DIR
from src.logging_config import get_logger

logger = get_logger(__name__)

_PUNCT_TRANS = str.maketrans('', '', string.punctuation)
_TIME_RE = re.compile(
    r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})'
)


class CombineError(Exception):
    """Base exception for transcript combination errors."""


@dataclass
class TranscriptEntry:
    """A single timestamped line spoken by one speaker."""
    timestamp: str
    start_time: float
    end_time: float
    speaker: str
    content: str  # original text, for display
    dedup_key: str  # lowercased/punctuation-stripped, for comparison only


@dataclass
class TranscriptConfig:
    """Mapping from a per-user VTT file to its display metadata."""
    name: str
    label: str
    description: str
    transcript_path: Path


def _normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used only for dedup comparison."""
    return re.sub(r'\s+', ' ', text.strip().lower().translate(_PUNCT_TRANS))


def should_skip_content(content: str, skip_filters: list[str]) -> bool:
    """True if content matches any literal or /regex/ filter."""
    for pattern in skip_filters:
        if pattern.startswith('/') and pattern.endswith('/'):
            if re.search(pattern[1:-1], content):
                return True
        elif pattern in content:
            return True
    return False


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """HH:MM:SS.mmm -> seconds."""
    h, m, sec_ms = timestamp.split(':')
    s, _, ms = sec_ms.partition('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000 if ms else 0)


def parse_vtt_file(
    file_path: Path,
    speaker_label: str,
    skip_filters: Optional[list[str]] = None,
) -> list[TranscriptEntry]:
    """Parse a VTT file into TranscriptEntry objects, preserving original text."""
    skip_filters = skip_filters or []

    try:
        content = file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise CombineError(f"Transcript file not found: {file_path}")
    except Exception as e:
        raise CombineError(f"Failed to read transcript file {file_path}: {e}") from e

    entries: list[TranscriptEntry] = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        match = _TIME_RE.match(lines[i].strip())
        if not match:
            i += 1
            continue

        start_str, end_str = match.group(1), match.group(2)
        i += 1
        content_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            content_lines.append(lines[i].strip())
            i += 1

        if not content_lines:
            continue

        text = ' '.join(content_lines)
        if should_skip_content(text, skip_filters):
            continue
        dedup_key = _normalize_for_dedup(text)
        if not dedup_key:
            continue

        entries.append(
            TranscriptEntry(
                timestamp=f"{start_str} --> {end_str}",
                start_time=parse_timestamp_to_seconds(start_str),
                end_time=parse_timestamp_to_seconds(end_str),
                speaker=speaker_label,
                content=text,
                dedup_key=dedup_key,
            )
        )

    return entries


def combine_transcripts(
    transcript_configs: list[TranscriptConfig],
    output_path: Path,
    skip_filters: Optional[list[str]] = None,
    include_timestamps: bool = True,
    chunks: int = 1,
) -> list[Path]:
    """Combine per-user VTTs into one (or more) chunked session transcripts.

    Dedup is always global across all speakers by (speaker, normalized_text).
    """
    skip_filters = skip_filters or []

    logger.info(f"Combining {len(transcript_configs)} transcript files...")
    all_entries: list[TranscriptEntry] = []
    for config in transcript_configs:
        logger.info(f"Processing {config.transcript_path.name} for {config.label}...")
        entries = parse_vtt_file(config.transcript_path, config.label, skip_filters)
        logger.info(f"  - Found {len(entries)} entries")
        all_entries.extend(entries)

    all_entries.sort(key=lambda e: e.start_time)
    logger.info(f"Total combined entries: {len(all_entries)}")

    seen: set[tuple[str, str]] = set()
    deduped: list[TranscriptEntry] = []
    for entry in all_entries:
        key = (entry.speaker, entry.dedup_key)
        if key in seen:
            logger.debug(f"DUPLICATE SKIP: {entry.speaker}: '{entry.dedup_key}'")
            continue
        seen.add(key)
        deduped.append(entry)

    summary_lines = ["Summary:"]
    seen_summary: set[str] = set()
    for config in transcript_configs:
        line = f"{config.name} - {config.label} - {config.description}"
        if line not in seen_summary:
            summary_lines.append(line)
            seen_summary.add(line)
    summary = '\n'.join(summary_lines)

    min_per_chunk = 5
    actual_chunks = 1 if len(deduped) < max(chunks, min_per_chunk) else chunks
    chunk_size = len(deduped) // actual_chunks
    output_files: list[Path] = []

    for chunk_idx in range(actual_chunks):
        start = chunk_idx * chunk_size
        end = len(deduped) if chunk_idx == actual_chunks - 1 else start + chunk_size
        chunk_entries = deduped[start:end]
        chunk_path = (
            output_path
            if actual_chunks == 1
            else output_path.with_name(
                f"{output_path.stem}-{chunk_idx + 1}{output_path.suffix}"
            )
        )
        try:
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(summary + '\n\n')
                for entry in chunk_entries:
                    if include_timestamps:
                        f.write(
                            f"{entry.speaker}: {entry.content} [{entry.timestamp}]\n"
                        )
                    else:
                        f.write(f"{entry.speaker}: {entry.content}\n")
            output_files.append(chunk_path)
            logger.info(f"Generated: {chunk_path}")
        except Exception as e:
            raise CombineError(f"Failed to write output file {chunk_path}: {e}") from e

    return output_files


def _load_username_mapping() -> dict[str, dict[str, str]]:
    """Build username -> {name, label, description} from TRANSCRIPT_N_* env vars."""
    mapping: dict[str, dict[str, str]] = {}
    i = 1
    while True:
        username = os.getenv(f'TRANSCRIPT_{i}_USERNAME', '').strip().strip('"')
        if not username:
            break

        name = (
            os.getenv(f'TRANSCRIPT_{i}_NAME', '')
            or os.getenv(f'TRANSCRIPT_{i}_PLAYER', '')
        ).strip().strip('"')
        label = (
            os.getenv(f'TRANSCRIPT_{i}_LABEL', '')
            or os.getenv(f'TRANSCRIPT_{i}_CHARACTER', '')
        ).strip().strip('"')
        description = os.getenv(f'TRANSCRIPT_{i}_DESCRIPTION', '').strip().strip('"')

        if all([name, label, description]):
            mapping[username] = {'name': name, 'label': label, 'description': description}
        else:
            logger.warning(f"Incomplete configuration for TRANSCRIPT_{i}_* variables")
        i += 1

    return mapping


def combine_transcripts_from_env(
    base_dir: Path,
    session_subdir: Optional[str] = None,
) -> list[Path]:
    """Combine transcripts using TRANSCRIPT_N_* env vars for speaker mapping."""
    search_dir = base_dir / session_subdir if session_subdir else base_dir
    output_dir = search_dir

    vtt_files = list(search_dir.glob("**/*.vtt"))
    if not vtt_files:
        raise CombineError(f"No VTT files found in {search_dir}")

    username_mapping = _load_username_mapping()
    if not username_mapping:
        raise CombineError(
            "No transcript mappings found. Define TRANSCRIPT_N_USERNAME, "
            "TRANSCRIPT_N_NAME, TRANSCRIPT_N_LABEL, TRANSCRIPT_N_DESCRIPTION."
        )

    transcript_configs: list[TranscriptConfig] = []
    unmapped_dirs: list[str] = []
    for vtt_file in vtt_files:
        parent_dir = vtt_file.parent.name
        matches = [u for u in username_mapping if u in parent_dir]
        if not matches:
            unmapped_dirs.append(parent_dir)
            continue
        if len(matches) > 1:
            raise CombineError(
                f"Ambiguous username match for directory '{parent_dir}': {matches}. "
                "Make TRANSCRIPT_*_USERNAME values more specific."
            )
        m = username_mapping[matches[0]]
        transcript_configs.append(
            TranscriptConfig(
                name=m['name'],
                label=m['label'],
                description=m['description'],
                transcript_path=vtt_file,
            )
        )

    if unmapped_dirs:
        raise CombineError(
            f"No mapping found for directories: {unmapped_dirs}. "
            "Update TRANSCRIPT_*_USERNAME environment variables."
        )
    if not transcript_configs:
        raise CombineError("No transcript configurations created.")

    include_timestamps = (
        os.getenv('INCLUDE_TIMESTAMPS', 'false').strip('"').lower() == 'true'
    )
    skip_filters_str = os.getenv(
        'SKIP_FILTERS', '[AUDIO OUT],[BLANK_AUDIO]'
    ).strip('"')
    skip_filters = [f.strip() for f in skip_filters_str.split(',') if f.strip()]
    chunks = int(os.getenv('CHUNKS', '1').strip('"'))

    output_filename = (
        f"{session_subdir}-combined.txt" if session_subdir
        else f"{output_dir.name}-combined.txt"
    )
    output_path = output_dir / output_filename

    return combine_transcripts(
        transcript_configs=transcript_configs,
        output_path=output_path,
        skip_filters=skip_filters,
        include_timestamps=include_timestamps,
        chunks=chunks,
    )
