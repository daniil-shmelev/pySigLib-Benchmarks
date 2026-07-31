# Signature Method Benchmark Suite

Benchmarks signature, log-signature, differentiation, branched-signature, and
signature-kernel implementations through a common adapter interface.

The suite measures steady-state `float32` kernel latency. Compilation, process
startup, path generation, and input conversion are outside the timed region.
Batched timings cover the complete batch, not one path.

Sweeps default to three untimed warmup iterations. The paper benchmark uses
one warmup, three measured calls, and reports the minimum measured runtime.
The signature paper sweep uses `N=10000`; the other paper sweeps use `N=1000`.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Rust 1.89+ for
`signature-rs`, and CUDA for GPU adapters.

Run the default signature-family sweep:

```bash
uv run src/orchestrator.py
```

Run a quick CPU smoke test:

```bash
uv run src/orchestrator.py config/smoke.yaml
```

Run every problem-specific paper benchmark:

```bash
uv run run_paper_benchmark.py
```

Each problem can also be run independently:

```bash
uv run run_paper_signatures.py
uv run run_paper_logsignatures.py
uv run run_paper_branched_signatures.py
uv run run_paper_signature_kernels.py
```

Each runner records benchmark and plotting wall time in a JSON summary under
`runs/` and generates heatmaps for its result directory.

Run a sweep with a saved or restricted library registry:

```bash
uv run src/orchestrator.py \
    --registry runs/benchmark_TIMESTAMP/libraries_registry.yaml \
    config/benchmark_sweep.yaml
```

Resume an interrupted run:

```bash
uv run src/orchestrator.py --resume runs/benchmark_TIMESTAMP
```

Generate heatmaps from a run:

```bash
uv run src/plotting.py runs/benchmark_TIMESTAMP
```

## Sweep configurations

| Config | Scope | Batch | Repeats |
| --- | --- | ---: | ---: |
| `config/benchmark_sweep.yaml` | Signature family; `N=256,512,1024`, `d=2,4,8,16`, `m=2,3` | 256 | 5 |
| `config/paper_signatures_sweep.yaml` | Forward signatures and `sig_backprop`; `N=10000` | 256 | 3 |
| `config/paper_logsignatures_sweep.yaml` | Forward log-signatures and backprop; `N=1000` | 256 | 3 |
| `config/paper_branched_signatures_sweep.yaml` | Forward planar/non-planar branched signatures and backprop; `N=1000` | 256 | 3 |
| `config/paper_signature_kernel_sweep.yaml` | Forward signature kernels and backprop; `N=1000` | 32 x 32 | 3 |
| `config/signature_kernel_sweep.yaml` | Signature kernels; `N=200,400,800`, `d=2,4,8,16` | 32 × 32 | 10 |
| `config/combined_sweep.yaml` | All operations on the kernel-safe grid | 32 | 10 |
| `config/bnrde_sweep.yaml` | Focused GPU log/branched-signature sweep | 256 | 10 |
| `config/smoke.yaml` | Minimal CPU check | 1 | 3 |

Edit `config/libraries_registry.yaml` to enable adapters and select their
backends. A sweep may use `libraries` to select registry entries. Edit the
paper sweep YAML to change its package list or parameter grid.

## Enabled libraries

| Library | Backends | Operations |
| --- | --- | --- |
| `iisignature` | CPU | signature, logsignature, backprop |
| `roughpy` | CPU | signature, logsignature |
| `signature-rs` | CPU | logsignature |
| `log-signatures-pytorch` | CPU, GPU | signature, logsignature, backprop |
| `signatory` | CPU, GPU | signature, logsignature, backprop |
| `pathsig` | GPU | signature, logsignature, backprop |
| `pysiglib` | CPU, GPU | signature, logsignature, planar/non-planar branched signature, signature kernel, backprop |
| `polysigkernel` | CPU, GPU | signature kernel, backprop |
| `stochastax` | CPU, GPU | signature, logsignature, planar/non-planar branched signature, backprop |
| `signax` | CPU, GPU | signature, logsignature, backprop |
| `tensordev` | CPU, GPU | signature, logsignature, backprop |
| `keras_sig` | CPU, GPU | signature, backprop |
| `chen-signatures` | CPU | signature, logsignature, backprop |

Important comparison notes:

- `roughpy`, `signature-rs`, and `chen-signatures` use explicit Python batch
  loops because their public APIs are single-path. Their timings include that
  loop overhead.
- JAX kernels compile during warmup and synchronize every measured call.
  `log-signatures-pytorch` uses eager CPU execution and compiled GPU execution.
- `pathsig` and `signatory` report compact Lyndon-basis log-signatures.
  `tensordev` reports an expanded word-basis tensor logarithm, so its
  log-signature output size differs.

The lockfiles pin the environments. `signature-py` builds from pinned Rust
source; `log-signatures-pytorch` uses a pinned compatibility wheel; and
`pathsig` and `signatory` build their extensions once and reuse the uv cache.

## Protocol and output

- Seeded standard Brownian inputs are cached with SHA-256 sidecars and reused
  across matching libraries and backends.
- Python tasks reuse persistent worker processes. Most adapters use one process
  per backend; adapters whose native state affects later measurements can
  recycle the worker at shape boundaries. Framework imports, CUDA
  initialization, and compatible host/device inputs are reused, while results
  are still committed one task at a time for safe resume.
- Every measured iteration is stored in `samples_ms`. `t_ms` is the median by
  default and the minimum for the paper benchmark.
- Backprop benchmarks construct the forward graph outside the timed region and
  time only the pullback applied to a deterministic random cotangent.
- Runtime errors and OOMs are stored in `failed_tasks.csv`,
  after which the next comparison cell is attempted.
- Results keep library, method, backend, path representation, and batch size
  distinct.
- Adapter failures are recorded in `failed_tasks.csv` and the sweep continues
  with a fresh worker. Known unsupported tasks are recorded separately.

Each `runs/benchmark_TIMESTAMP/` directory contains:

- `results.csv`, `completed_tasks.txt`, `skipped_tasks.csv`, and `failed_tasks.csv`
- the exact sweep, registry, lockfile, and cached inputs
- environment and hardware metadata
- `git_status.txt` and `git_diff.patch` when the checkout is dirty

For signature operations, `t_ms` is the latency for `batch_size` paths. For
signature kernels, it is the latency for a `batch_size × batch_size` matrix.
`alloc_bytes` is only the average net Python heap change visible to
`tracemalloc`; it excludes native, framework-pool, and device memory.

## Development

Python adapters subclass `common.BenchmarkAdapter` and return a callable
containing only the measured operation. Register new adapters in
`config/libraries_registry.yaml`; asynchronous kernels must synchronize before
returning.

Python adapters default to one worker process per backend. An adapter whose
native state changes later measurements can set `WORKER_SCOPE = "shape"` to
restart only when `(N, d, batch_size, path_kind, seed)` changes. PathSig uses
this scope; the JAX adapters retain one process for the complete backend sweep.

Run the tests with:

```bash
uv run --with pytest pytest -q
```
