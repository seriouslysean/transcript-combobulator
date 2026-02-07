# Parallelization Plan

## Goal
Significantly reduce total processing time when transcribing multiple files by utilizing multi-core CPUs (and potentially multiple GPUs if available).

## Recommended Approach: Shell-based Parallelism (`xargs`)

This approach is the least intrusive to the codebase and leverages standard Unix tools.

### Implementation Steps

1.  **Modify `Makefile`**:
    *   Target: `run`
    *   Replace the existing `for` loop that iterates over `$$files`.
    *   Use `xargs` with the `-P` flag to specify the number of parallel processes.

### Code Example

```makefile
# Old Loop
# for file in $$files; do \
#     $(MAKE) run-single file=$$file; \
# done;

# New Parallel Execution (Use 4 cores)
echo "$$files" | tr '\n' '\0' | xargs -0 -n 1 -P 4 -I {} \
    $(MAKE) run-single file="{}" ENV_FILE="$(ENV_FILE)"
```

### Considerations

*   **Output Interleaving**: The output from multiple parallel processes will be mixed in the console. This is generally acceptable for a CLI tool, but buffering or per-job logging might be desired later.
*   **Resource Usage**:
    *   **CPU**: High.
    *   **RAM**: High. Each process loads its own instance of the Whisper model (approx 1-3GB per instance depending on model size). Ensure the machine has enough RAM (e.g., 4 processes * 2GB = 8GB overhead).
    *   **GPU/MPS**: If using GPU acceleration, multiple processes fighting for VRAM might cause OOM errors. On macOS (MPS), it usually manages well, but performance gains might plateau if the Neural Engine is saturated.

## Alternative: Python-based `multiprocessing`

If `xargs` proves too brittle or resource-heavy:

1.  Create `tools/process_batch.py`.
2.  Use `concurrent.futures.ProcessPoolExecutor`.
3.  Allows for smarter resource management (e.g., a shared queue, better error handling).

```python
# Conceptual Python implementation
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_single_file, f) for f in files]
```

```