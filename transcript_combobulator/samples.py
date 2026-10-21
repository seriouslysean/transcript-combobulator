#!/usr/bin/env python3
"""Create sample/test audio files from inputs in samples/ for development and testing."""

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

import soundfile as sf
import numpy as np


def create_copies(
    input_path: Path,
    output_dir: Path,
    num_copies: int = 5,
    prefix: str = "",
) -> None:
    """Duplicate input_path into output_dir, numbered ``{prefix}{stem}_N{suffix}``.

    ``num_copies=1`` produces a single unnumbered copy (``{prefix}{stem}{suffix}``).
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = input_path.stem, input_path.suffix

    if num_copies == 1:
        dest = output_dir / f"{prefix}{stem}{suffix}"
        shutil.copy2(input_path, dest)
        print(f"Created {dest}")
        return

    for i in range(num_copies):
        dest = output_dir / f"{prefix}{stem}_{i + 1}{suffix}"
        shutil.copy2(input_path, dest)
        print(f"Created {dest}")


def create_padded_audio(
    input_path: Path,
    output_path: Path,
    num_copies: int = 3,
    silence_duration: float = 4.0,
) -> None:
    """Write a file that repeats input_path ``num_copies`` times with silence between."""
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    wav, sr = sf.read(str(input_path), dtype='float32', always_2d=True)
    silence = np.zeros((int(silence_duration * sr), wav.shape[1]), dtype=np.float32)

    parts = [wav]
    for _ in range(num_copies - 1):
        parts.extend((silence, wav))
    padded_wav = np.concatenate(parts, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), padded_wav, sr)
    print(f"Created padded version: {output_path}")


def create_sample_files(
    prefix: str = "",
    copies: int = 5,
    padded_copies: int = 3,
    session: str = "jfk-sample",
) -> None:
    """Create numbered copies + a padded file for every WAV in samples/.

    Test files (prefix starting with ``test_``) go to ``tmp/input/``.
    Sample files go to ``tmp/input/{session}/``.
    """
    root_dir = Path(__file__).parent.parent
    samples_dir = root_dir / 'samples'

    if not samples_dir.exists():
        print(f"Error: Samples directory not found at {samples_dir}")
        print("Please create the samples/ directory and add your audio files.")
        return

    wav_files = list(samples_dir.glob('*.wav'))
    if not wav_files:
        print(f"Error: No WAV files found in {samples_dir}")
        return

    input_dir = (
        root_dir / 'tmp' / 'input'
        if prefix.startswith("test_")
        else root_dir / 'tmp' / 'input' / session
    )
    print(f"Creating sample files in {input_dir}")

    for sample_file in wav_files:
        create_copies(sample_file, input_dir, num_copies=copies, prefix=prefix)
        create_padded_audio(
            input_path=sample_file,
            output_path=input_dir / f"{prefix}{sample_file.stem}_padded{sample_file.suffix}",
            num_copies=padded_copies,
        )


# A multi-speaker sample from the AMI Meeting Corpus (CC BY 4.0; samples/ami/NOTICE.md):
# meeting EN2002b, one headset per speaker, 240-540 s. That window was picked from the
# corpus's word-level annotations as the one with the most crosstalk where all four
# speak: 98-126 s of speech each, 124 s with two or more talking, three silences of 3 s+,
# 85 annotated laughs. It is fetched, not committed, and checked byte for byte.
AMI_MEETING = 'EN2002b'
AMI_URL = 'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.Headset-{n}.wav'
AMI_SHA256 = (  # headsets 0-3 as served on 2026-10-21, 57,179,180 bytes each
    'c1d29d1fe2abc238ee98464fb71b0f33c9f92d8be14fce2e9857341b8f22420c',
    'ebf3eea1d9459b31f191593893ff30e57aed6c23da5c843162e8a0e77d1fae87',
    '2f01166f3895c987e91f4ad47426ac38ac80e900d4b6ee4da125112d7292b197',
    '68f495547ef9674d1d6848c12d7358829cc6e6d302268c0c51c3e3c470b7f692',
)
AMI_START_SECONDS = 240
AMI_SECONDS = 300
AMI_SESSION = 'ami-en2002b'


class SampleError(Exception):
    """A sample download is missing or does not match its checksum."""


def download_verified(url: str, dest: Path, sha256: str) -> Path:
    """Fetch url to dest unless dest already has this sha256; never leave a bad file behind."""
    if dest.is_file() and _sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as out:
            while block := response.read(1 << 20):
                digest.update(block)
                out.write(block)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise SampleError(f'cannot download {url}: {e}') from e
    if digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise SampleError(f'{url} does not match its pinned sha256; the source changed, so the reference no longer applies')
    part.replace(dest)
    return dest


def _sha256(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def create_ami_sample(root_dir: Path | None = None) -> Path:
    """Fetch the AMI headsets into tmp/cache/ami/ and cut the excerpt to tmp/input/ami-en2002b/."""
    root_dir = root_dir or Path(__file__).parent.parent
    cache = root_dir / 'tmp' / 'cache' / 'ami'
    out_dir = root_dir / 'tmp' / 'input' / AMI_SESSION
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, sha in enumerate(AMI_SHA256):
        url = AMI_URL.format(meeting=AMI_MEETING, n=n)
        src = download_verified(url, cache / f'{AMI_MEETING}.Headset-{n}.wav', sha)
        dest = out_dir / f'{n + 1}-ami_{"abcd"[n]}.flac'
        if dest.is_file():
            continue  # a rewrite would change the file's mtime and re-run the whole pipeline
        part = dest.with_name(dest.stem + '.part.flac')
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-y', '-ss', str(AMI_START_SECONDS), '-t', str(AMI_SECONDS),
             '-i', str(src), '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', str(part)],
            check=True, timeout=600,
        )
        part.replace(dest)
        print(f'Created {dest}')
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="combobulator samples", description="Build sample and test audio from samples/")
    parser.add_argument('--prefix', default="", help='Prefix for output files (e.g. "test_")')
    parser.add_argument('--copies', type=int, default=5)
    parser.add_argument('--padded-copies', type=int, default=3)
    parser.add_argument('--session', default="jfk-sample")
    parser.add_argument('--ami', action='store_true', help=f'Fetch the AMI {AMI_MEETING} excerpt instead (network)')
    args = parser.parse_args(argv)
    if args.ami:
        try:
            create_ami_sample()
        except SampleError as e:
            raise SystemExit(f'Error: {e}') from None
        return
    create_sample_files(
        prefix=args.prefix,
        copies=args.copies,
        padded_copies=args.padded_copies,
        session=args.session,
    )
