# Signature Method Benchmark Suite

A small orchestrator-and-adapter suite for benchmarking signature, log-signature,
branched-signature, differentiation, and signature-kernel implementations.

The repository is intended to produce steady-state kernel-throughput measurements.
Compilation, dependency startup, path generation, and input conversion are outside
the timed region. Batched results report the latency of the complete batched call;
the batch size is always retained in the data and plot identity.

## Benchmark protocol

- Inputs use `float32` across Python, JAX, Torch, and Julia adapters.
- Stochastic fBM inputs are generated from the configured `seed`. Each adapter
  resets the same deterministic input sequence, so libraries and backends receive
  matching data for a given task.
- JAX kernels are JIT-compiled during warmup and synchronized with
  `block_until_ready()` during measurement.
- Each timed iteration is retained in `samples_ms`.
- `t_ms` is the sample median; `t_ms_mean` and `t_ms_std` are also recorded.
- Adapter errors stop the run while preserving every completed result for resume.
- CPU/GPU, method, path representation, and batch variants remain distinct in
  every plot.

The allocation column is a diagnostic only. Python `tracemalloc` does not measure
native or GPU allocations and should not be used for cross-language memory claims.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Rust 1.89 or newer to build the pinned `signature-rs` Python binding
- A CUDA-capable environment for GPU registry entries
- Julia 1.10 only when the Julia adapter is enabled

The tracked `uv.lock` and Julia `Manifest.toml` pin the software environments.
Git-based Python dependencies are pinned to commit revisions in `pyproject.toml`.
`signature-py` is not currently published on PyPI, so its first installation
builds the pinned Rust workspace source; subsequent runs reuse `uv`'s build cache.
`log-signatures-pytorch` 0.1.11 is pure Python and runs on Python 3.12, despite
wheel metadata requiring Python 3.13. Its registry entry therefore pins the exact
PyPI wheel and SHA-256 digest; the orchestrator installs it once into the ignored
`.adapter_packages/` cache using a package-specific compatibility override.

## Run

Run the configured benchmark sweep:

```bash
uv run src/orchestrator.py
```

Run the small smoke configuration:

```bash
uv run src/orchestrator.py config/smoke.yaml
```

Continue an interrupted run using its saved configurations:

```bash
uv run src/orchestrator.py --resume runs/benchmark_TIMESTAMP
```

`results.csv` is appended and flushed after every successful adapter task.
`completed_tasks.txt` records which task IDs are safe to skip. On resume, any
uncommitted final rows are discarded and only missing tasks are executed.

Run benchmarks and generate every plot:

```bash
uv run run_benchmarks.py
```

Generate plots from a specific run or CSV:

```bash
uv run src/plotting.py runs/benchmark_TIMESTAMP --plot-type all
uv run src/plotting.py path/to/results.csv --plot-type heatmap
```

Available plot types are `line`, `heatmap`, `speedup`, `profile`, and `box`.
Line and speedup layouts are derived from all operations actually present in the
CSV, including branched signatures and signature kernels.

## Configuration

`config/benchmark_sweep.yaml` defines the main parameter grid:

```yaml
path_kind: fbm
seed: 20260529
Ns: [200, 400, 800, 1600]
Ds: [2, 4, 8, 12, 16, 20]
Ms: [2, 3]
operations:
  - signature
  - logsignature
  - branchedsignature_nonplanar
  - branchedsignature_planar
backends: [cpu, gpu]
batch_size: 256
repeats: 10
runs_dir: runs
```

`config/libraries_registry.yaml` controls which adapters are active. The current
working registry enables CPU-only `iisignature`, `RoughPy`, and `signature-rs`,
plus CPU/GPU `log-signatures-pytorch` and `pysiglib`; other adapters remain
available in `adapters/` and should be enabled deliberately for a comparison run.

RoughPy and `signature-rs` expose only single-path APIs, so their method labels
identify the simple Python `batch_loop` used to process the common batched
workload. These measurements are public-API batch throughput, including Python
loop overhead, rather than native batched-kernel throughput. `iisignature` accepts
the batch natively, while JAX/Torch adapters use native batching or compiled
vectorization. `log-signatures-pytorch` uses its native batched API in float32:
eager mode on CPU and `torch.compile(mode="reduce-overhead")` on GPU, with
compilation in warmup and CUDA synchronization in every measured call. The
`chen-signatures` adapter skips batch sizes above one.

RoughPy's float32 increments and algebra context are prepared outside timing, but
each measured call constructs a fresh stream so RoughPy's interval cache cannot
turn repetitions into cache lookups. The `signature-rs` builder and contiguous
float32 input are prepared outside timing; its binding still performs its internal
NaN check, input copy, basis construction, and log-signature calculation during
each measured call.

Focused configurations are provided for branched signatures and signature
kernels:

- `config/bnrde_sweep.yaml`
- `config/signature_kernel_sweep.yaml`
- `config/smoke.yaml`

## Run artifacts

Each run creates `runs/benchmark_TIMESTAMP/` containing:

- `results.csv`
- `completed_tasks.txt`
- `benchmark_sweep.yaml`
- `libraries_registry.yaml`
- `metadata.json`
- `uv.lock`

`metadata.json` records the timestamp, git commit and dirty state, Python/platform
information, Julia and uv versions, visible GPU/driver information when available,
and relevant backend/thread environment variables.

The CSV contains:

```text
task_id,N,d,m,batch_size,seed,path_kind,operation,backend,language,library,
method,path_type,t_ms,t_ms_mean,t_ms_std,samples_ms,alloc_bytes
```

For a batched signature calculation, `t_ms` is the latency of one call processing
`batch_size` paths. For a signature-kernel calculation, one call produces a
`batch_size × batch_size` kernel matrix. Do not interpret these values as
single-path latency.

## Adding an adapter

Python adapters subclass `common.BenchmarkAdapter` and return a callable whose
body contains only the operation being measured. Register the adapter in
`config/libraries_registry.yaml` with its optional dependency group and supported
operations.

Every asynchronous backend must synchronize before returning from the callable.
Input conversion, basis construction, and JIT construction belong in setup unless
the experiment is explicitly defined as end-to-end.

## Tests

```bash
uv run --with pytest pytest -q
```

The tests cover path generation, adapter wiring, signature-kernel wiring, and plot
generation. A paper run should additionally be inspected for complete task
coverage and stable timing distributions before figures are produced.
