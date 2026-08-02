"""Heatmap plotting utilities for signature benchmarks"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

SeriesKey = Tuple[str, str, str, str, int]

plt.rcParams.update({
    "axes.edgecolor": "#4B5563",
    "axes.labelcolor": "#1F2937",
    "axes.labelsize": 15,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlecolor": "#111827",
    "axes.titlesize": 16,
    "axes.titleweight": "semibold",
    "figure.facecolor": "white",
    "font.family": "serif",
    "font.size": 13,
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "grid.color": "#D1D5DB",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "lines.linewidth": 2.0,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "savefig.facecolor": "white",
    "xtick.color": "#374151",
    "xtick.labelsize": 12,
    "ytick.color": "#374151",
    "ytick.labelsize": 12,
})


def _format_number(value: float) -> str:
    """Format a number without scientific notation."""
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(value))))
    decimal_places = max(0, min(6, 2 - magnitude))
    formatted = f"{value:.{decimal_places}f}"
    return formatted.rstrip("0").rstrip(".") if decimal_places else formatted


NUMBER_FORMATTER = FuncFormatter(lambda value, _: _format_number(value))


def _operation_label(operation: str) -> str:
    """Return a publication-friendly operation name."""
    labels = {
        "signature": "Signature",
        "sig_backprop": "Signature backprop",
        "logsignature": "Log signature",
        "logsignature_backprop": "Log signature backprop",
        "branchedsignature_nonplanar": "Non-planar branched signature",
        "branchedsignature_nonplanar_backprop": "Non-planar branched backprop",
        "branchedsignature_planar": "Planar branched signature",
        "branchedsignature_planar_backprop": "Planar branched backprop",
        "signaturekernel": "Signature kernel",
        "signaturekernel_backprop": "Signature kernel backprop",
    }
    return labels.get(operation, operation.replace("_", " ").capitalize())


def load_results(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Load benchmark results from CSV file.

    Args:
        csv_path: Path to results CSV file

    Returns:
        List of result dictionaries
    """
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "N": int(row["N"]),
                "d": int(row["d"]),
                "m": int(row["m"]),
                "batch_size": int(row.get("batch_size") or 1),
                "seed": int(row.get("seed") or 0),
                "path_kind": row["path_kind"].strip(),
                "operation": row["operation"].strip(),
                "backend": row.get("backend", "").strip(),
                "language": row.get("language", "").strip(),
                "library": row["library"].strip(),
                "method": row.get("method", "").strip(),
                "path_type": row.get("path_type", "").strip(),
                "t_ms": float(row["t_ms"]),
            })
    return rows


def _series_key(row: Dict[str, Any]) -> SeriesKey:
    """Identify a measured implementation without collapsing its variants."""
    return (
        row["library"],
        row.get("backend", ""),
        row.get("method", ""),
        row.get("path_type", ""),
        row.get("batch_size", 1),
    )


def _series_label(key: SeriesKey, *, include_backend: bool = True) -> str:
    """Return a readable label while identity details remain in the series key."""
    library, backend, _, _, _ = key
    label = library
    if include_backend and backend:
        label += f" [{backend}]"
    return label


def _ordered_operations(rows: List[Dict[str, Any]]) -> List[str]:
    """Use a stable conventional order while retaining every observed operation."""
    observed = {row["operation"] for row in rows}
    preferred = [
        "signature",
        "sig_backprop",
        "logsignature",
        "logsignature_backprop",
        "branchedsignature_nonplanar",
        "branchedsignature_nonplanar_backprop",
        "branchedsignature_planar",
        "branchedsignature_planar_backprop",
        "signaturekernel",
        "signaturekernel_backprop",
    ]
    return [op for op in preferred if op in observed] + sorted(observed - set(preferred))


def _rows_for_operation(
    rows: List[Dict[str, Any]],
    operation: Optional[str],
    backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return rows for the requested operation and backend selection."""
    return [
        row
        for row in rows
        if (
            (operation is None or row["operation"] == operation)
            and (backend is None or row.get("backend", "") == backend)
        )
    ]


def get_latest_run(runs_dir: Path = Path("runs")) -> Optional[Path]:
    """
    Find the most recent benchmark run directory.

    Args:
        runs_dir: Directory containing benchmark runs

    Returns:
        Path to latest run directory, or None if no runs found
    """
    if not runs_dir.exists():
        return None

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("benchmark_")]
    if not run_dirs:
        return None

    # Sort by directory name (which includes timestamp)
    return sorted(run_dirs)[-1]


def _format_heatmap_param_axis(
    params: List[Tuple[int, int]],
) -> Tuple[List[str], str, str, bool]:
    """
    Format heatmap row labels, omitting dimensions fixed across the panel.

    Returns labels, y-axis label, title suffix, and whether row ticks are needed.
    """
    Ns = sorted({n for n, _ in params})
    Ds = sorted({d for _, d in params})
    show_N = len(Ns) > 1
    show_d = len(Ds) > 1

    labels = []
    for n, d in params:
        if show_N and show_d:
            labels.append(f"$N={_format_number(n)}, d={d}$")
        elif show_N:
            labels.append(_format_number(n))
        elif show_d:
            labels.append(str(d))
        else:
            labels.append("")

    fixed_parts = []
    if not show_N and Ns:
        fixed_parts.append(f"N={_format_number(Ns[0])}")
    if not show_d and Ds:
        fixed_parts.append(f"d={Ds[0]}")

    if show_N and show_d:
        ylabel = "Parameters"
    elif show_N:
        ylabel = "$N$"
    elif show_d:
        ylabel = "$d$"
    else:
        ylabel = ""

    fixed_title = f"${', '.join(fixed_parts)}$" if fixed_parts else ""
    return labels, ylabel, fixed_title, show_N or show_d


def _format_heatmap_library_axis(libraries: List[str]) -> Tuple[str, bool]:
    """Return a title suffix and whether x-axis library ticks are needed."""
    if len(libraries) == 1:
        return libraries[0], False
    return "", True


def _safe_filename_part(value: object) -> str:
    """Make a compact filesystem-safe filename component."""
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(value).strip()
    )
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "none"


def _configure_heatmap_axes(
    ax: Any,
    panel: Dict[str, Any],
    show_titles: bool,
) -> None:
    """Apply shared heatmap axis labels and optional title."""
    libraries = panel["libraries"]
    params = panel["params"]

    if panel["show_library_axis"]:
        ax.set_xticks(np.arange(len(libraries)))
        ax.set_xticklabels(libraries, rotation=45, ha="right")
        ax.set_xlabel("")
    else:
        ax.set_xticks([])
        ax.set_xlabel("")

    if panel["show_param_axis"]:
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels(panel["param_labels"], fontsize=12)
        ax.set_ylabel(panel["param_ylabel"])
    else:
        ax.set_yticks([])
        ax.set_ylabel("")

    if show_titles:
        ax.set_title(" | ".join(panel["title_parts"]))


def _annotate_heatmap(
    ax: Any,
    matrix: np.ndarray,
    norm: LogNorm,
    failure_labels: Optional[np.ndarray] = None,
    fontsize: int = 10,
) -> None:
    """Add runtime text labels to heatmap cells."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            failure_label = (
                failure_labels[i, j]
                if failure_labels is not None
                else ""
            )
            if failure_label:
                ax.text(
                    j,
                    i,
                    failure_label,
                    ha="center",
                    va="center",
                    color="#991B1B",
                    fontsize=fontsize,
                    fontweight="semibold",
                )
            elif not np.isnan(matrix[i, j]):
                normalized = float(norm(matrix[i, j])) if matrix[i, j] > 0.0 else 0.0
                color = "black" if normalized > 0.55 else "white"
                ax.text(
                    j,
                    i,
                    _format_number(matrix[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=fontsize,
                )


def _heatmap_color_scale(values: List[float]) -> LogNorm:
    """Choose logarithmic color limits for one heatmap panel."""
    positive_values = [value for value in values if np.isfinite(value) and value > 0.0]
    if not positive_values:
        return LogNorm(vmin=1.0, vmax=10.0)

    data_min = min(positive_values)
    data_max = max(positive_values)
    if data_min == data_max:
        return LogNorm(vmin=data_min / 1.05, vmax=data_max * 1.05)
    return LogNorm(vmin=data_min, vmax=data_max)


def make_heatmap_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    show_titles: bool = True,
    operation: Optional[str] = None,
    backend: Optional[str] = None,
) -> Path:
    """
    Generate heatmap showing performance across all parameter combinations.

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path (defaults to same dir as CSV)
        config: Optional configuration dict
        show_titles: Whether to show subplot titles
        operation: Optional operation to plot in isolation
        backend: Optional backend to plot in isolation

    Returns:
        Path to saved plot
    """
    rows = _rows_for_operation(load_results(csv_path), operation, backend)
    oom_failures = set()
    failed_tasks_path = csv_path.parent / "failed_tasks.csv"
    if failed_tasks_path.exists():
        with failed_tasks_path.open("r", encoding="utf-8", newline="") as f:
            for failure in csv.DictReader(f):
                error_text = " ".join([
                    failure.get("error_type", ""),
                    failure.get("reason", ""),
                ]).lower()
                if not (
                    "oom" in error_text
                    or "out of memory" in error_text
                    or "memoryerror" in error_text
                ):
                    continue
                oom_failures.add((
                    failure.get("operation", "").strip(),
                    failure.get("backend", "").strip(),
                    int(failure["m"]),
                    int(failure["N"]),
                    int(failure["d"]),
                    failure.get("library", "").strip(),
                ))

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Get unique values
    operations = _ordered_operations(rows)
    depths = sorted(set(r["m"] for r in rows))
    backends = sorted(set(r.get("backend", "") for r in rows))
    panels: List[Dict[str, Any]] = []

    for depth_idx, m in enumerate(depths):
        for backend_idx, backend in enumerate(backends):
            axis_row = depth_idx * len(backends) + backend_idx

            for op_idx, operation in enumerate(operations):
                op_rows = [
                    r
                    for r in rows
                    if (
                        r["operation"] == operation
                        and r["m"] == m
                        and r.get("backend", "") == backend
                    )
                ]
                if not op_rows:
                    continue

                series_keys = sorted(set(_series_key(r) for r in op_rows))
                libraries = [
                    _series_label(key, include_backend=False)
                    for key in series_keys
                ]
                params = sorted(set((r["N"], r["d"]) for r in op_rows))
                param_labels, param_ylabel, fixed_params_title, show_param_axis = (
                    _format_heatmap_param_axis(params)
                )
                library_title, show_library_axis = _format_heatmap_library_axis(libraries)

                matrix = np.full((len(params), len(libraries)), np.nan)
                failure_labels = np.full(
                    (len(params), len(libraries)),
                    "",
                    dtype=object,
                )
                result_by_key = {}
                for r in op_rows:
                    result_by_key.setdefault(
                        (r["N"], r["d"], _series_key(r)),
                        r["t_ms"],
                    )
                for row_idx, (N, d) in enumerate(params):
                    for col_idx, series_key in enumerate(series_keys):
                        t_ms = result_by_key.get((N, d, series_key))
                        if t_ms is not None:
                            matrix[row_idx, col_idx] = t_ms
                        elif (
                            operation,
                            backend,
                            m,
                            N,
                            d,
                            series_key[0],
                        ) in oom_failures:
                            failure_labels[row_idx, col_idx] = "OOM"

                finite_values = matrix[np.isfinite(matrix)]
                norm = _heatmap_color_scale([float(v) for v in finite_values])

                title_parts = [_operation_label(operation)]
                if backend:
                    title_parts.append(backend.upper())
                title_parts.append(f"$m={m}$")
                if fixed_params_title:
                    title_parts.append(fixed_params_title)
                if library_title:
                    title_parts.append(library_title)

                panels.append({
                    "axis_row": axis_row,
                    "op_idx": op_idx,
                    "operation": operation,
                    "m": m,
                    "backend": backend,
                    "libraries": libraries,
                    "params": params,
                    "param_labels": param_labels,
                    "param_ylabel": param_ylabel,
                    "show_param_axis": show_param_axis,
                    "show_library_axis": (
                        show_library_axis
                        and axis_row == len(depths) * len(backends) - 1
                    ),
                    "title_parts": title_parts,
                    "matrix": matrix,
                    "failure_labels": failure_labels,
                    "norm": norm,
                })

    # Create a heatmap for each operation/depth/backend tuple. Splitting by
    # backend keeps CPU and GPU comparisons in separate panels.
    num_ops = len(operations)
    num_depths = len(depths)
    num_backends = len(backends)
    max_libraries = max(
        (len(panel["libraries"]) for panel in panels),
        default=1,
    )
    fig, axes = plt.subplots(
        num_depths * num_backends,
        num_ops,
        figsize=(
            max(8.0, 0.9 * max_libraries) * num_ops,
            4.6 * num_depths * num_backends,
        ),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.07},
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for panel in panels:
        ax = axes[panel["axis_row"]][panel["op_idx"]]
        ax.set_visible(True)

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#E5E7EB")
        im = ax.imshow(
            panel["matrix"],
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            norm=panel["norm"],
        )
        _configure_heatmap_axes(ax, panel, show_titles)
        _annotate_heatmap(
            ax,
            panel["matrix"],
            panel["norm"],
            panel["failure_labels"],
        )
        cbar = fig.colorbar(im, ax=ax, location="right")
        cbar.formatter = NUMBER_FORMATTER
        cbar.update_ticks()
        cbar.set_label("Runtime (ms, log scale)")

    if output_path is None:
        output_path = csv_path.parent / "plot_heatmap.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap plot saved to: {output_path}")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate heatmaps from signature benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate heatmaps for the latest run
  python plotting.py

  # Generate heatmaps for a specific run
  python plotting.py runs/benchmark_20251201_221922
        """,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=str,
        help="Path to benchmark run directory or results.csv (defaults to latest run)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for plots (defaults to run directory)",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Disable subplot titles",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_path = Path(args.run_dir)
        if run_path.is_file() and run_path.suffix == ".csv":
            csv_path = run_path
            run_dir = run_path.parent
        elif run_path.is_dir():
            run_dir = run_path
            csv_path = run_dir / "results.csv"
        else:
            print(f"Error: {run_path} is not a valid directory or CSV file")
            sys.exit(1)
    else:
        latest = get_latest_run()
        if not latest:
            print("Error: No benchmark runs found in 'runs/' directory")
            sys.exit(1)
        run_dir = latest
        csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = run_dir

    all_rows = load_results(csv_path)
    operations = _ordered_operations(all_rows)
    backends = sorted(set(row.get("backend", "") for row in all_rows))
    plot_dir = output_dir / "plots" / "heatmap"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating heatmaps by operation and backend in: {plot_dir}\n")

    failed = False
    for operation in operations:
        for backend in backends:
            if not any(
                row["operation"] == operation
                and row.get("backend", "") == backend
                for row in all_rows
            ):
                continue
            output_path = plot_dir / (
                f"{_safe_filename_part(operation)}_"
                f"{_safe_filename_part(backend)}.png"
            )
            try:
                make_heatmap_plot(
                    csv_path,
                    output_path,
                    show_titles=not args.no_titles,
                    operation=operation,
                    backend=backend,
                )
            except Exception as e:
                failed = True
                print(
                    f"Error generating heatmap for {operation} [{backend}]: {e}",
                )

    if failed:
        sys.exit(1)

    print(f"\nDone! Heatmaps saved to: {plot_dir}")
