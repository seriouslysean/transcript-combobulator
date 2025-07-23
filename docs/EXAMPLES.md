# Examples

Working examples are included in `tmp/output/example/` to demonstrate output format and character mapping.

## View Example Output

```bash
ENV_FILE=.env.example make combine-transcripts session=example
```

This processes the example VTT files using the character mappings in `.env.example`.

## Example Character Mapping

From `.env.example`:
```bash
TRANSCRIPT_1_USERNAME=dm
TRANSCRIPT_1_PLAYER="DM"
TRANSCRIPT_1_CHARACTER="DM"
TRANSCRIPT_1_DESCRIPTION="Dungeon Master"

TRANSCRIPT_2_USERNAME=barbarian
TRANSCRIPT_2_PLAYER="Player 1"
TRANSCRIPT_2_CHARACTER="Barbarian"
TRANSCRIPT_2_DESCRIPTION="Goliath Barbarian"
```

## Example Output Structure

```
tmp/output/example/
├── dm/
│   └── dm.vtt                    # Individual DM transcript
├── barbarian/
│   └── barbarian.vtt             # Individual player transcript
├── druid/
│   └── druid.vtt
├── rogue/
│   └── rogue.vtt
├── example-combined-1.txt        # Combined session transcript (part 1)
└── example-combined-2.txt        # Combined session transcript (part 2)
```

## Example Combined Output

```
Summary:
DM - DM - Dungeon Master
Player 1 - Barbarian - Goliath Barbarian
Player 2 - Druid - Human Druid

FILE 1 of 2

TRANSCRIPT:
DM: The wind howls through the ruined village.
Barbarian: That's a 16 on my save.
Druid: I cast Detect Magic, just in case.
...
```
