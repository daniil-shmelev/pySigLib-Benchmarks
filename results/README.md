# Paper results

This directory contains finalized benchmark outputs selected for the paper.
Files are copied here manually after a run has been checked and approved.

Use one directory per benchmark group:

```text
results/
  signatures/
    results.csv
    heatmaps/
      signature_cpu.png
      signature_gpu.png
  logsignatures/
  branched_signatures/
  signature_kernels/
```

Copy `results.csv` without modifying it. Copy the corresponding finalized
heatmaps from the run's `plots/heatmap/` directory. Unlike `runs/`, everything
under this directory is tracked by Git.
