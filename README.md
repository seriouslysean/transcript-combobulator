# Transcript Precombobulator

## Environment Configuration

The application loads environment variables from a `.env` file by default. To use a different environment file (for example, for testing with example data), set the `ENV_FILE` environment variable to the path of the desired env file:

```sh
export ENV_FILE=.env.example
make combine-transcripts session=example
```

Or, for a one-off command:

```sh
ENV_FILE=.env.example make combine-transcripts session=example
```

This allows you to flexibly switch between real and example output/test data. The `.env.example` file is provided as a reference for the example output in `tmp/output/example/`.

## Example Output

The example output folder is located at `tmp/output/example/` and is structured for testing the transcript combination logic. The `.env.example` file is configured to match this structure.

## Restoring Environment Files

- `.env` should contain your real/production configuration.
- `.env.example` should contain the example/generic configuration for testing.

## Running Transcript Combination

To combine transcripts for the example data:

```sh
ENV_FILE=.env.example make combine-transcripts session=example
```

---

For more details, see the documentation in `docs/README.md`.
