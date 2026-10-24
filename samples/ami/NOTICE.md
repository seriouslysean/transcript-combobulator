# AMI sample: source, licence and what it is for

## Source and licence

The audio for `make sample-ami` and the transcript in `EN2002b-reference.txt` come from the
**AMI Meeting Corpus** (https://groups.inf.ed.ac.uk/ami/corpus/), meeting EN2002b, the
individual headset recordings 0 to 3. The corpus is released under the
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)
licence ([corpus licence page](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)). The
licence's disclaimer of warranties applies, and the corpus creators do not endorse this
project.

**Changes:**
- The audio is not stored in this repository. `make sample-ami` downloads the four headset
  WAVs from the corpus mirror and checks each against a pinned SHA-256
  (`transcript_combobulator/samples.py`).
- It keeps 240 to 540 s of each recording and writes 16 kHz mono FLAC.
- `EN2002b-reference.txt` is not AMI's transcript. It is this project's output on that
  excerpt: `.env.ami-sample` with whisper `large-v3-turbo` on CPU, run on 2026-10-21.

## Why this excerpt

AMI's word-level annotations rank every 5-minute window of the 4-headset meetings. In this one
all four people speak, 98 to 126 s each, and two or more talk at once for 124 s. It has three
silences of 3 s or more, and 85 annotated laughs.

The jfk sample is one 11 s sentence on six copies of one voice. It cannot show a change.

## What it can and cannot tell you

| Use | Works? |
|---|---|
| Did a change that should keep the words leave them alone (a runtime, a refactor, a dependency bump)? | Yes. `make check-sample` passes only on identical output. Two runs with the same settings were byte-identical. |
| Is a settings change better or worse? | No. The check fails on any change in words, better or worse. |
| Tune VAD for Discord tracks | No. The headsets pick up each other's speech. |

**Why a failed check doesn't mean worse.** On 2026-10-21 every settings change failed the
check: `PADDING_SECONDS` 0.4, `VAD_THRESHOLD` 0.2, 0.8 and 0.95, `VAD_MIN_SPEECH_DURATION`
1.0, and `VAD_MIN_SILENCE_DURATION` 0.5.

**Which were really worse.** Scored against AMI's human transcript of the same 300 s:
- Only `VAD_THRESHOLD` 0.2 was clearly worse: 30.3% of words differ, against 25.7% for the
  default.
- Higher thresholds scored better, because they drop speech heard from the other headsets.

**About the bleed.** It roughly doubles what whisper writes on these tracks, and Discord
tracks normally do not have it.
