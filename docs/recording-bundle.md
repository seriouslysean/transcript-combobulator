# Recording bundle, schema version 1

A recording bundle is a folder that hands one recording's audio from a fetcher (Craig, a local-file import, a future single-track source) to a transcriber. It is source-neutral: a consumer reads `manifest.json` and never needs to know which source produced the bundle.

## Layout

```
<bundle>/
  manifest.json
  <audio files>        any names and subfolders the producer chooses; the manifest lists them
```

The folder holds no source credentials. A Craig access key, for example, appears nowhere in the manifest, in file names or in any other file.

## manifest.json

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema_version` | integer | yes | `1` |
| `recording_id` | string | yes | The source's identifier for the recording. It must not be secret. |
| `source` | string | no | The producer's name for the source, such as `craig`. It is provenance only, and consumers never branch on it. |
| `started_at` | string or null | no | When the recording started, in ISO 8601 with an offset, such as `2026-10-18T23:04:11.000Z`. Use null when the source doesn't say. |
| `tracks` | array, at least 1 | yes | One entry per audio file, in the source's order. |

Each entry in `tracks`:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `track_id` | string | yes | Unique within the bundle. |
| `file` | string | yes | The path relative to the bundle folder, using `/` separators. |
| `sha256` | string | yes | The SHA-256 of the file's bytes, as 64 lowercase hex digits. |
| `speaker_key` | string or null | yes | The identity of the one person on this track, namespaced by source, such as `discord:<user id>`. It is stable across recordings when the source provides that. Null means the speaker is unknown, and the track may hold more than one person. |
| `username` | string or null | yes | The speaker's name exactly as the source reports it. It is not a sanitized file stem. Null when the source has none. |

## Rules

1. **Shared timeline.** Every track starts at the recording's time zero, with leading silence kept, so timestamps from different tracks can be interleaved.
2. **Paths stay inside.** A `file` value is never absolute, never contains a `..` segment, and never resolves through a symlink to a location outside the bundle. The consumer checks this as well as the producer.
3. **Unique stems.** No two `file` values share a base name without its extension, compared case-insensitively. Consumers may name per-track output after the stem, and macOS folders ignore case.
4. **Unknown stays unknown.** A producer never guesses a `speaker_key` or a `username`, and a consumer never infers a speaker from track order or from file names. A bundle with one track, `speaker_key: null` and `username: null` is valid. A consumer that needs one known speaker per track rejects the bundle and names the track; it does not guess.
5. **Complete or absent.** The producer writes every audio file first and `manifest.json` last, to a temporary file in the bundle folder followed by an atomic rename. A folder without `manifest.json` is incomplete and is not a bundle. The manifest appearing is the event a watcher reacts to.
6. **Immutable once published.** A producer doesn't change a published bundle. A rerun that finds a bundle verifies it and leaves it alone.
7. **Verify before use.** A consumer checks every `sha256` before it reads the audio.
8. **Versioning.** A consumer rejects a `schema_version` it doesn't know and ignores fields it doesn't know. Adding an optional field keeps the version at 1. Any other change raises it.
9. **Audio format.** The file extension names the format. Producers for version 1 write FLAC, and a consumer rejects a format it can't read.
10. **Only what transcription needs.** Server, channel and requester names, and free-text notes, are left out.

## Examples

A Craig recording with three speakers (synthetic values):

```json
{
  "schema_version": 1,
  "recording_id": "AbCdEf12345",
  "source": "craig",
  "started_at": "2026-10-18T23:04:11.000Z",
  "tracks": [
    {"track_id": "1", "file": "1-alice_dm.flac", "sha256": "<64 hex>", "speaker_key": "discord:100000000000000001", "username": "alice.dm"},
    {"track_id": "2", "file": "2-bob.flac", "sha256": "<64 hex>", "speaker_key": "discord:100000000000000002", "username": "bob"},
    {"track_id": "3", "file": "3-carol.flac", "sha256": "<64 hex>", "speaker_key": "discord:100000000000000003", "username": "carol"}
  ]
}
```

A single mixed track whose speakers are unknown, such as a downloaded video:

```json
{
  "schema_version": 1,
  "recording_id": "video-20261018-a",
  "started_at": null,
  "tracks": [
    {"track_id": "1", "file": "audio.flac", "sha256": "<64 hex>", "speaker_key": null, "username": null}
  ]
}
```

## Notes for consumers

- **Match on `speaker_key` first.** Usernames change when people rename, so `speaker_key` is the primary match and the raw `username` is the fallback. Neither is a file stem, such as Craig's sanitized `<n>-<name>`.
- **Session date.** The consumer derives a local session date from `started_at`, in a time zone it chooses, such as the host's with an override. A start at 01:00 UTC falls on the previous evening in the Americas.

## Assumptions and open decisions

- **Assumption, unverified:** Craig gives each user at most one track per recording. A consumer may reject duplicate speaker keys, but the format doesn't require keys to be unique.

## What combobulator accepts

This page is the canonical copy of the format; producers cite its schema version. `combobulator session-env --bundle DIR` checks a bundle against it (`transcript_combobulator.bundle.BUNDLE_SCHEMA_VERSION`) before writing the session's env, and refuses with a message that names the file or track at fault.

- **Schema version 1 only.** Any other `schema_version` is refused. Unknown fields are ignored.
- **Checked before use:**
  - every `file` stays inside the bundle and exists;
  - every `sha256` matches;
  - stems are unique, ignoring case;
  - no two tracks share a `speaker_key`.
- **Flat folders only, for now.** `combobulator run` reads the audio files at the top of a folder. So a `file` in a subfolder is refused, and so is an audio file in the folder that the manifest does not list, which `run` would otherwise transcribe as an extra speaker.
- **Formats `run` reads.** `.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg`, `.aac` and `.opus`. Version 1 producers write FLAC.
- **Unknown speakers.** A track with `speaker_key` and `username` null gets a provisional speaker row, marked as possibly several people. It is transcribed under one label, since combobulator does not separate voices within a track.
