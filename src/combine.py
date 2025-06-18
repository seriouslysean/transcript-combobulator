"""Transcript combination utilities."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from src.config import OUTPUT_DIR

# Load environment variables
load_dotenv()


class CombineError(Exception):
    """Base exception for transcript combination errors."""
    pass


@dataclass
class TranscriptEntry:
    """Represents a single transcript entry."""
    timestamp: str
    start_time: float
    end_time: float
    character: str
    content: str
    
    
@dataclass
class TranscriptConfig:
    """Configuration for a transcript file."""
    player_name: str
    role: str
    character_name: str
    character_description: str
    transcript_path: Path


def normalize_content(text: str) -> str:
    """Normalize content by trimming and removing excessive spaces and newlines.
    
    Args:
        text: The content text to normalize.
        
    Returns:
        Normalized content.
    """
    return re.sub(r'\s+', ' ', text.strip())


def should_skip_content(content: str, skip_filters: List[str]) -> bool:
    """Check if the content matches any of the skip filters.
    
    Args:
        content: The content to check.
        skip_filters: List of strings or regex patterns to filter out.
        
    Returns:
        True if the content matches any filter, false otherwise.
    """
    for filter_pattern in skip_filters:
        if filter_pattern.startswith('/') and filter_pattern.endswith('/'):
            # Regex pattern
            pattern = filter_pattern[1:-1]
            if re.search(pattern, content):
                return True
        else:
            # String match
            if filter_pattern in content:
                return True
    return False


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """Convert VTT timestamp to seconds.
    
    Args:
        timestamp: Timestamp string in format HH:MM:SS.mmm
        
    Returns:
        Time in seconds as float.
    """
    parts = timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split('.')
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
    
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_vtt_file(file_path: Path, character_name: str, dedupe: str = "false", 
                  skip_filters: Optional[List[str]] = None) -> List[TranscriptEntry]:
    """Parse VTT file content and return timestamped entries.
    
    Args:
        file_path: Path to the VTT file.
        character_name: Name of the character/speaker.
        dedupe: Deduplication strategy ("false", "consecutive", "unique").
        skip_filters: List of strings or regex patterns to filter out content.
        
    Returns:
        List of parsed transcript entries.
        
    Raises:
        CombineError: If parsing fails.
    """
    if skip_filters is None:
        skip_filters = []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        raise CombineError(f"Transcript file not found: {file_path}")
    except Exception as e:
        raise CombineError(f"Failed to read transcript file {file_path}: {e}")

    # Parse VTT format
    time_regex = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})')
    entries = []
    last_content = None
    unique_contents = set()
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for timestamp line
        match = time_regex.match(line)
        if match:
            start_time_str = match.group(1)
            end_time_str = match.group(2)
            timestamp = f"{start_time_str} --> {end_time_str}"
            
            # Get content from next non-empty lines
            i += 1
            content_lines = []
            while i < len(lines) and lines[i].strip():
                content_lines.append(lines[i].strip())
                i += 1
                
            if content_lines:
                text_content = ' '.join(content_lines)
                normalized_content = normalize_content(text_content)
                
                # Skip content if it matches any filters
                if should_skip_content(normalized_content, skip_filters):
                    continue
                    
                # Apply deduplication
                skip_duplicate = False
                if dedupe == "consecutive" and last_content == normalized_content:
                    skip_duplicate = True
                elif dedupe == "unique" and normalized_content in unique_contents:
                    skip_duplicate = True
                    
                if not skip_duplicate and normalized_content:
                    start_seconds = parse_timestamp_to_seconds(start_time_str)
                    end_seconds = parse_timestamp_to_seconds(end_time_str)
                    
                    entry = TranscriptEntry(
                        timestamp=timestamp,
                        start_time=start_seconds,
                        end_time=end_seconds,
                        character=character_name,
                        content=normalized_content
                    )
                    entries.append(entry)
                    
                    last_content = normalized_content
                    unique_contents.add(normalized_content)
        else:
            i += 1
            
    return entries


def combine_transcripts(transcript_configs: List[TranscriptConfig], 
                       output_path: Path,
                       dedupe: str = "false",
                       skip_filters: Optional[List[str]] = None,
                       include_timestamps: bool = True,
                       chunks: int = 1) -> List[Path]:
    """Combine multiple transcript files into a single output file.
    
    Args:
        transcript_configs: List of transcript configurations.
        output_path: Path for the output file.
        dedupe: Deduplication strategy.
        skip_filters: List of content filters.
        include_timestamps: Whether to include timestamps in output.
        chunks: Number of chunks to split the output into.
        
    Returns:
        List of paths to the generated output files.
        
    Raises:
        CombineError: If combination fails.
    """
    if skip_filters is None:
        skip_filters = []
        
    all_entries = []
    
    print(f"\nCombining {len(transcript_configs)} transcript files...")
    
    # Parse all transcript files
    for config in transcript_configs:
        print(f"Processing {config.transcript_path.name} for {config.character_name}...")
        
        entries = parse_vtt_file(
            config.transcript_path,
            config.character_name,
            dedupe,
            skip_filters
        )
        
        print(f"  - Found {len(entries)} entries")
        all_entries.extend(entries)
    
    # Sort by start time
    all_entries.sort(key=lambda x: x.start_time)
    
    print(f"Total combined entries: {len(all_entries)}")
    
    # Create summary
    summary_lines = ["Summary:"]
    for config in transcript_configs:
        if config.role == "DM":
            summary_lines.append(
                f"{config.player_name} - {config.role} - {config.character_description}"
            )
        else:
            summary_lines.append(
                f"{config.player_name} - {config.character_name} - {config.character_description}"
            )
    summary = '\n'.join(summary_lines)
    
    # Split into chunks
    chunk_size = len(all_entries) // chunks
    output_files = []
    
    for chunk_idx in range(chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if chunk_idx < chunks - 1 else len(all_entries)
        chunk_entries = all_entries[start_idx:end_idx]
        
        # Generate output content
        content_lines = [summary, ""]
        
        if chunks > 1:
            content_lines.append(f"FILE {chunk_idx + 1} of {chunks}")
            content_lines.append("")
        
        content_lines.append("TRANSCRIPT:")
        
        for entry in chunk_entries:
            if include_timestamps:
                line = f"[{entry.timestamp}] {entry.character}: {entry.content}"
            else:
                line = f"{entry.character}: {entry.content}"
            content_lines.append(line)
        
        # Determine output file path
        if chunks > 1:
            stem = output_path.stem
            suffix = output_path.suffix
            chunk_path = output_path.parent / f"{stem}-{chunk_idx + 1}{suffix}"
        else:
            chunk_path = output_path
            
        # Write output file
        try:
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
            output_files.append(chunk_path)
            print(f"Generated: {chunk_path}")
        except Exception as e:
            raise CombineError(f"Failed to write output file {chunk_path}: {e}")
    
    return output_files


def combine_transcripts_from_env(base_dir: Path, session_subdir: Optional[str] = None) -> List[Path]:
    """Combine transcripts using environment variable configuration.
    
    Args:
        base_dir: Base output directory to search for transcripts.
        session_subdir: Optional subdirectory (e.g., "2025-06-16") to search within.
        
    Returns:
        List of paths to generated output files.
        
    Raises:
        CombineError: If combination fails or env vars are missing.
    """
    # Determine search directory
    if session_subdir:
        search_dir = base_dir / session_subdir
        output_dir = search_dir
    else:
        search_dir = base_dir
        output_dir = base_dir
    
    # Find VTT files directly in the session directory
    vtt_files = list(search_dir.glob("*.vtt"))
    if not vtt_files:
        raise CombineError(f"No VTT files found in {search_dir}")
    
    # Load mapping configuration from environment using indexed variables
    directory_mapping = {}
    
    # Find all transcript configurations by looking for TRANSCRIPT_N_DIR patterns
    i = 1
    while True:
        directory = os.getenv(f'TRANSCRIPT_{i}_DIR', '').strip().strip('"')
        if not directory:
            break
            
        player_name = os.getenv(f'TRANSCRIPT_{i}_PLAYER', '').strip().strip('"')
        role = os.getenv(f'TRANSCRIPT_{i}_ROLE', '').strip().strip('"')
        character_name = os.getenv(f'TRANSCRIPT_{i}_CHARACTER', '').strip().strip('"')
        character_description = os.getenv(f'TRANSCRIPT_{i}_DESCRIPTION', '').strip().strip('"')
        
        if all([player_name, role, character_name, character_description]):
            directory_mapping[directory] = {
                'player_name': player_name,
                'role': role,
                'character_name': character_name,
                'character_description': character_description
            }
        else:
            print(f"Warning: Incomplete configuration for TRANSCRIPT_{i}_* variables")
        
        i += 1
    
    if not directory_mapping:
        raise CombineError("No transcript mappings found. Define environment variables like TRANSCRIPT_1_DIR, TRANSCRIPT_1_PLAYER, etc.")
    
    # Create transcript configurations using mapping
    transcript_configs = []
    unmapped_dirs = []
    
    for vtt_file in vtt_files:
        # Extract directory name from VTT filename (remove .vtt extension)
        file_basename = vtt_file.stem
        
        if file_basename in directory_mapping:
            mapping = directory_mapping[file_basename]
            config = TranscriptConfig(
                player_name=mapping['player_name'],
                role=mapping['role'],
                character_name=mapping['character_name'],
                character_description=mapping['character_description'],
                transcript_path=vtt_file
            )
            transcript_configs.append(config)
        else:
            unmapped_dirs.append(file_basename)
    
    if unmapped_dirs:
        raise CombineError(f"No mapping found for directories: {unmapped_dirs}. Please update TRANSCRIPT_* environment variables.")
    
    if not transcript_configs:
        raise CombineError("No transcript configurations created. Check TRANSCRIPT_* environment variables.")
    
    # Get other settings from environment
    campaign_name = os.getenv('CAMPAIGN_NAME', 'D&D Campaign').strip('"')
    dedupe = os.getenv('DEDUPE_STRATEGY', 'consecutive').strip('"')
    include_timestamps = os.getenv('INCLUDE_TIMESTAMPS', 'false').strip('"').lower() == 'true'
    skip_filters_str = os.getenv('SKIP_FILTERS', '[AUDIO OUT],[BLANK_AUDIO]').strip('"')
    skip_filters = [f.strip() for f in skip_filters_str.split(',') if f.strip()]
    chunks = int(os.getenv('CHUNKS', '1').strip('"'))
    
    # Generate output filename based on session directory name
    if session_subdir:
        output_filename = f"{session_subdir}-combined.txt"
    else:
        # Use current directory name as fallback
        output_filename = f"{output_dir.name}-combined.txt"
    output_path = output_dir / output_filename
    
    return combine_transcripts(
        transcript_configs=transcript_configs,
        output_path=output_path,
        dedupe=dedupe,
        skip_filters=skip_filters,
        include_timestamps=include_timestamps,
        chunks=chunks
    )


def combine_transcripts_for_directory(input_dir: Path, 
                                    output_path: Optional[Path] = None,
                                    dedupe: str = "consecutive",
                                    skip_filters: Optional[List[str]] = None,
                                    include_timestamps: bool = False) -> List[Path]:
    """Combine all VTT files in a directory into a single transcript.
    
    Args:
        input_dir: Directory containing VTT files and subdirectories.
        output_path: Path for combined output. If None, uses default naming.
        dedupe: Deduplication strategy.
        skip_filters: List of content filters.
        include_timestamps: Whether to include timestamps.
        
    Returns:
        List of paths to generated output files.
        
    Raises:
        CombineError: If combination fails.
    """
    if skip_filters is None:
        skip_filters = ["[AUDIO OUT]", "[BLANK_AUDIO]"]
    
    # Find all VTT files in subdirectories
    vtt_files = list(input_dir.glob("*/*.vtt"))
    
    if not vtt_files:
        raise CombineError(f"No VTT files found in {input_dir}")
    
    # Create transcript configurations
    transcript_configs = []
    for vtt_file in vtt_files:
        # Extract character name from directory or filename
        character_name = vtt_file.parent.name
        
        config = TranscriptConfig(
            player_name=character_name,
            role="Player",
            character_name=character_name,
            character_description="",
            transcript_path=vtt_file
        )
        transcript_configs.append(config)
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_dir / f"combined_transcript_{timestamp}.txt"
    
    return combine_transcripts(
        transcript_configs,
        output_path,
        dedupe=dedupe,
        skip_filters=skip_filters,
        include_timestamps=include_timestamps
    )