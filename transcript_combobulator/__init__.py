"""Audio processing and transcription pipeline.

Import submodules directly (``from transcript_combobulator.vad import process_audio``). This
package init deliberately imports nothing: the batch parent process only needs
config, combine, and telemetry, and must not pull torch and whisper into
memory just to render a progress table.
"""
