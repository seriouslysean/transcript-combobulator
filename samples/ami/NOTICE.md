# AMI sample: source, licence and what it is for

## Source and licence

The audio for `make sample-ami`, and the words in `EN2002b-human.txt`, come from the
**AMI Meeting Corpus** (https://groups.inf.ed.ac.uk/ami/corpus/), meeting EN2002b: the
individual headset recordings 0 to 3, and the manual annotations
(`ami_public_manual_1.6.2.zip`). The corpus and its annotations are released under the
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
- `EN2002b-human.txt` is derived from AMI's word-level transcript by
  `combobulator samples --ami-human`, which downloads the annotations and checks a pinned
  SHA-256. `EN2002b-human.json` records the source, each line's AMI word ids, and the speaker
  mapping. Changes from AMI's annotation:
  - Window: only words that start and end inside 240 to 540 s. Times are seconds from 240.
  - Speakers: each AMI agent is labelled by its headset channel from `meetings.xml`
    (channel 0 is Speaker A, and so on), the same order as the audio files.
  - Words: truncated words, vocal sounds, gaps, disfluency markers and transcription-error
    marks are dropped. Punctuation is joined to the word before it.
  - Turns: one line per AMI segment, sorted by start time.
  - Scoring (`compare --wer`) applies whisper's English normaliser to both sides.

## Why this excerpt

AMI's word-level annotations rank every 5-minute window of the 4-headset meetings. In this one
all four people speak, 98 to 126 s each, and two or more talk at once for 124 s. It has three
silences of 3 s or more, and 85 annotated laughs.

The jfk sample is one 11 s sentence on six copies of one voice. It cannot show a change.

## What it can and cannot tell you

| Use | Works? |
|---|---|
| Did a change that should keep the words leave them alone (a runtime, a refactor, a dependency bump)? | Yes. `make check-sample` passes only on identical output. Two runs with the same settings were byte-identical. |
| Is a settings change better or worse? | Only as a rough sign. `check-sample` prints WER against `EN2002b-human.txt` (no gate). One 5-minute excerpt cannot rank settings. |
| Tune VAD for Discord tracks | No. The headsets pick up each other's speech. |

**Why a failed check doesn't mean worse.** On 2026-10-21 every settings change failed the
check: `PADDING_SECONDS` 0.4, `VAD_THRESHOLD` 0.2, 0.8 and 0.95, `VAD_MIN_SPEECH_DURATION`
1.0, and `VAD_MIN_SILENCE_DURATION` 0.5. Their WER is in the table below.

WER against `EN2002b-human.txt`, rerun on 2026-10-23 with whisper `large-v3-turbo` on CPU,
each row one change from `.env.ami-sample`:

| Setting | WER | WER without words another speaker said |
|---|---|---|
| `.env.ami-sample` as is | 46.0% | 29.0% |
| `PADDING_SECONDS=0.4` | 45.5% | 27.8% |
| `VAD_THRESHOLD=0.2` | 57.6% | 31.8% |
| `VAD_THRESHOLD=0.8` | 36.1% | 25.3% |
| `VAD_THRESHOLD=0.95` | 29.2% | 23.9% |
| `VAD_MIN_SPEECH_DURATION=1.0` | 43.2% | 28.0% |
| `VAD_MIN_SILENCE_DURATION=0.5` | 38.0% | 27.4% |

Most of the spread is bleed: a higher threshold drops the quieter speech from other
headsets. The column without those words moves far less, and one excerpt can't say whether
the gain carries over to Discord tracks.

**About the bleed.** Each headset picks up the other speakers. Whisper writes much of that
speech into the wrong speaker's lines, and Discord tracks normally do not have it. WER counts
those words as insertions. `compare --wer` also reports how many inserted words another
speaker's reference line says within 1 s (`other_speaker_words`), and the WER without them.
That count is only a sign: a common word can match by chance.

