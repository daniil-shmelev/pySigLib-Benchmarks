"""Plotting utilities for signature benchmarks"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

SeriesKey = Tuple[str, str, str, str, int]


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


def _series_label(key: SeriesKey) -> str:
    """Return a compact, unambiguous plot label for an implementation variant."""
    library, backend, method, path_type, batch_size = key
    label = library
    if backend:
        label += f" [{backend}]"
    details = [value for value in (method, path_type) if value]
    details.append(f"batch={batch_size}")
    return f"{label} ({', '.join(details)})"


def _ordered_operations(rows: List[Dict[str, Any]]) -> List[str]:
    """Use a stable conventional order while retaining every observed operation."""
    observed = {row["operation"] for row in rows}
    preferred = [
        "signature",
        "logsignature",
        "sigdiff",
        "branchedsignature_nonplanar",
        "branchedsignature_planar",
        "signaturekernel",
    ]
    return [op for op in preferred if op in observed] + sorted(observed - set(preferred))


def get_time(
    rows: List[Dict[str, Any]],
    series_key: SeriesKey,
    N: int,
    d: int,
    m: int,
    path_kind: str,
    operation: str
) -> Optional[float]:
    """
    Find timing result for specific configuration.

    Args:
        rows: List of result dictionaries
        series_key: Full implementation identity
        N: Number of points
        d: Dimension
        m: Signature level
        path_kind: Path type
        operation: Operation name

    Returns:
        Time in milliseconds, or None if not found
    """
    for r in rows:
        if (
            r["N"] == N
            and r["d"] == d
            and r["m"] == m
            and r["path_kind"] == path_kind
            and r["operation"] == operation
            and _series_key(r) == series_key
        ):
            return r["t_ms"]
    return None


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
            labels.append(f"N={n}, d={d}")
        elif show_N:
            labels.append(str(n))
        elif show_d:
            labels.append(str(d))
        else:
            labels.append("")

    fixed_parts = []
    if not show_N and Ns:
        fixed_parts.append(f"N={Ns[0]}")
    if not show_d and Ds:
        fixed_parts.append(f"d={Ds[0]}")

    if show_N and show_d:
        ylabel = "Parameters"
    elif show_N:
        ylabel = "N"
    elif show_d:
        ylabel = "d"
    else:
        ylabel = ""

    return labels, ylabel, ", ".join(fixed_parts), show_N or show_d


def _format_heatmap_library_axis(libraries: List[str]) -> Tuple[str, bool]:
    """Return a title suffix and whether x-axis library ticks are needed."""
    if len(libraries) == 1:
        return f"library={libraries[0]}", False
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
        ax.set_xlabel("Library")
    else:
        ax.set_xticks([])
        ax.set_xlabel("")

    if panel["show_param_axis"]:
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels(panel["param_labels"], fontsize=8)
        ax.set_ylabel(panel["param_ylabel"])
    else:
        ax.set_yticks([])
        ax.set_ylabel("")

    if show_titles:
        ax.set_title(f"{', '.join(panel['title_parts'])} - Runtime (ms)")


def _annotate_heatmap(
    ax: Any,
    matrix: np.ndarray,
    threshold: float,
    fontsize: int = 7,
) -> None:
    """Add runtime text labels to heatmap cells."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                color = "black" if matrix[i, j] > threshold else "white"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2g}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=fontsize,
                )


def _heatmap_color_scale(values: List[float]) -> Tuple[float, float, List[float]]:
    """Choose shared color limits and ticks that include readable lower values."""
    if not values:
        return 0.0, 1.0, []

    data_min = min(values)
    data_max = max(values)
    if data_min == data_max:
        padding = abs(data_min) * 0.05 or 1.0
        vmin = data_min - padding
        vmax = data_max + padding
        if data_min >= 0.0:
            vmin = max(0.0, vmin)
        return vmin, vmax, []

    ticks = np.asarray(
        MaxNLocator(nbins=8).tick_values(data_min, data_max),
        dtype=float,
    )
    ticks = ticks[np.isfinite(ticks)]
    if ticks.size < 2:
        return data_min, data_max, []

    vmin = float(ticks[0])
    vmax = float(ticks[-1])
    if data_min >= 0.0:
        vmin = max(0.0, vmin)
        if vmin == 0.0 and data_min < 1.0:
            lower_decimal_tick = np.floor(data_min * 10.0) / 10.0
            if lower_decimal_tick > 0.0:
                vmin = float(lower_decimal_tick)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    if ticks.size and ticks[0] > vmin:
        ticks = np.insert(ticks, 0, vmin)
    return vmin, vmax, [float(tick) for tick in ticks]


def _individual_heatmap_figsize(panels: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Pick a uniform, compact size for standalone heatmap panels."""
    if not panels:
        return 2.25, 2.4

    max_rows = max(panel["matrix"].shape[0] for panel in panels)
    max_cols = max(panel["matrix"].shape[1] for panel in panels)

    width = min(max(2.25, 1.75 + 0.25 * max_cols), 2.6)
    height = min(max(2.4, 0.13 * max_rows + 0.7), 4.2)
    return round(width, 2), round(height, 2)


def _save_individual_heatmap(
    panel: Dict[str, Any],
    output_path: Path,
    vmin: float,
    vmax: float,
    colorbar_ticks: List[float],
    annotation_threshold: float,
    include_colorbar: bool,
    figsize: Tuple[float, float],
) -> None:
    """Save a single heatmap panel using the shared color scale."""
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.08],
        left=0.23,
        right=0.76,
        bottom=0.08,
        top=0.98,
        wspace=0.16,
    )
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    im = ax.imshow(
        panel["matrix"],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    _configure_heatmap_axes(ax, panel, show_titles=False)
    ax.tick_params(axis="both", labelsize=7, length=3)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    if include_colorbar:
        cbar = fig.colorbar(im, cax=cax, ticks=colorbar_ticks or None)
        cbar.set_label("Runtime (ms)")
        cbar.ax.tick_params(labelsize=7, length=3)
        cbar.ax.yaxis.label.set_size(8)
    else:
        cax.set_visible(False)
    _annotate_heatmap(ax, panel["matrix"], annotation_threshold, fontsize=6)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def make_line_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate 3x3 line plot comparison grid (original visualization).

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path (defaults to same dir as CSV)
        config: Optional configuration dict with sweep parameters

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Derive grid parameters from data or config
    if config:
        Ns = sorted(config.get("Ns", []))
        Ds = sorted(config.get("Ds", []))
        Ms = sorted(config.get("Ms", []))
        path_kind = config.get("path_kind", "sin")
    else:
        Ns = sorted(set(r["N"] for r in rows))
        Ds = sorted(set(r["d"] for r in rows))
        Ms = sorted(set(r["m"] for r in rows))
        path_kind = rows[0]["path_kind"]

    # Fixed parameters for each subplot (use max for worst-case scaling)
    N_fixed_for_d = max(Ns)
    N_fixed_for_m = max(Ns)
    d_fixed_for_N = max(Ds)
    d_fixed_for_m = max(Ds)
    m_fixed_for_N = max(Ms)
    m_fixed_for_d = max(Ms)

    # Keep backend/method/path/batch variants distinct.
    series_keys = sorted(set(_series_key(r) for r in rows))
    op_order = _ordered_operations(rows)

    # Rows vary N/d/m; columns are all operations present in the data.
    fig, axes = plt.subplots(
        3,
        len(op_order),
        figsize=(5 * len(op_order), 12),
        sharey="col",
        squeeze=False,
    )

    for row_idx, vary in enumerate(["N", "d", "m"]):
        for col_idx, op in enumerate(op_order):
            ax = axes[row_idx, col_idx]

            # Determine x-axis values and fixed parameters
            if vary == "N":
                xs = Ns
                d_fix = d_fixed_for_N
                m_fix = m_fixed_for_N
                xlabel = "N (number of points)"
            elif vary == "d":
                xs = Ds
                d_fix = None
                m_fix = m_fixed_for_d
                xlabel = "d (dimension)"
            else:  # vary m
                xs = Ms
                d_fix = d_fixed_for_m
                m_fix = None
                xlabel = "m (signature level)"

            plotted_any = False

            # Plot each library
            for series_key in series_keys:
                ys = []
                xs_effective = []

                for x in xs:
                    if vary == "N":
                        N, d, m = x, d_fix, m_fix
                    elif vary == "d":
                        N, d, m = N_fixed_for_d, x, m_fix
                    else:  # vary m
                        N, d, m = N_fixed_for_m, d_fix, x

                    t = get_time(rows, series_key, N, d, m, path_kind, op)
                    if t is not None and t > 0.0:
                        xs_effective.append(x)
                        ys.append(t)

                # Only plot if we have at least 2 points
                if len(xs_effective) >= 2:
                    ax.plot(
                        xs_effective,
                        ys,
                        marker="o",
                        label=_series_label(series_key),
                    )
                    plotted_any = True

            # Hide subplot if no data plotted
            if not plotted_any:
                ax.set_visible(False)
                continue

            # Configure axes
            ax.set_xlabel(xlabel)
            if xs:
                ax.set_xticks(xs)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_ylabel("time (ms)")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # Title with fixed parameters
            title = f"{op}, vary {vary}"
            if vary == "N":
                title += f" (d={d_fixed_for_N}, m={m_fixed_for_N})"
            elif vary == "d":
                title += f" (N={N_fixed_for_d}, m={m_fixed_for_d})"
            else:
                title += f" (N={N_fixed_for_m}, d={d_fixed_for_m})"
            ax.set_title(title)

            # Legend only on top row
            if row_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=8)

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / "plot_line.png"

    fig.savefig(output_path, dpi=300)
    print(f"Line plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_heatmap_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    show_titles: bool = True,
) -> Path:
    """
    Generate heatmap showing performance across all parameter combinations.

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path (defaults to same dir as CSV)
        config: Optional configuration dict
        show_titles: Whether to show titles on combined and individual heatmaps

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Get unique values
    operations = _ordered_operations(rows)
    depths = sorted(set(r["m"] for r in rows))
    backends = sorted(set(r.get("backend", "") for r in rows))
    panels: List[Dict[str, Any]] = []
    global_finite_values: List[float] = []

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
                libraries = [_series_label(key) for key in series_keys]
                params = sorted(set((r["N"], r["d"]) for r in op_rows))
                param_labels, param_ylabel, fixed_params_title, show_param_axis = (
                    _format_heatmap_param_axis(params)
                )
                library_title, show_library_axis = _format_heatmap_library_axis(libraries)

                matrix = np.full((len(params), len(libraries)), np.nan)
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

                finite_values = matrix[np.isfinite(matrix)]
                global_finite_values.extend(float(v) for v in finite_values)

                title_parts = [f"{operation}", f"m={m}"]
                if backend:
                    title_parts.append(f"backend={backend}")
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
                    "show_library_axis": show_library_axis,
                    "title_parts": title_parts,
                    "matrix": matrix,
                })

    vmin, vmax, colorbar_ticks = _heatmap_color_scale(global_finite_values)
    annotation_threshold = (vmin + vmax) / 2

    # Create a heatmap for each operation/depth/backend tuple. Splitting by
    # backend keeps CPU and GPU comparisons in separate panels.
    num_ops = len(operations)
    num_depths = len(depths)
    num_backends = len(backends)
    fig, axes = plt.subplots(
        num_depths * num_backends,
        num_ops,
        figsize=(7 * num_ops, 4.5 * num_depths * num_backends),
        squeeze=False,
        constrained_layout=True,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    images = []
    for panel in panels:
        ax = axes[panel["axis_row"]][panel["op_idx"]]
        ax.set_visible(True)

        im = ax.imshow(
            panel["matrix"],
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(im)
        _configure_heatmap_axes(ax, panel, show_titles)
        _annotate_heatmap(ax, panel["matrix"], annotation_threshold)

    if images:
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        cbar = fig.colorbar(
            images[0],
            ax=visible_axes,
            location="right",
            ticks=colorbar_ticks or None,
        )
        cbar.set_label("Runtime (ms)")

    if output_path is None:
        output_path = csv_path.parent / "plot_heatmap.png"

    individual_dir = output_path.parent / "plot_heatmaps"
    individual_dir.mkdir(parents=True, exist_ok=True)
    colorbar_panel = (
        max(panels, key=lambda panel: (panel["axis_row"], panel["op_idx"]))
        if panels
        else None
    )
    individual_figsize = _individual_heatmap_figsize(panels)
    for panel in panels:
        filename_parts = [
            _safe_filename_part(panel["operation"]),
            f"m-{_safe_filename_part(panel['m'])}",
        ]
        if panel["backend"]:
            filename_parts.append(f"backend-{_safe_filename_part(panel['backend'])}")
        filename = "_".join(filename_parts) + ".png"
        _save_individual_heatmap(
            panel,
            individual_dir / filename,
            vmin,
            vmax,
            colorbar_ticks,
            annotation_threshold,
            panel is colorbar_panel,
            individual_figsize,
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap plot saved to: {output_path}")
    print(f"Individual heatmaps saved to: {individual_dir}")
    plt.close(fig)

    return output_path


def make_speedup_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    baseline: str = "slowest"
) -> Path:
    """
    Generate speedup plot showing relative performance (same layout as line plot).

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict
        baseline: Baseline for speedup calculation ("slowest", "fastest", or library name)

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Derive grid parameters from data or config
    if config:
        Ns = sorted(config.get("Ns", []))
        Ds = sorted(config.get("Ds", []))
        Ms = sorted(config.get("Ms", []))
        path_kind = config.get("path_kind", "sin")
    else:
        Ns = sorted(set(r["N"] for r in rows))
        Ds = sorted(set(r["d"] for r in rows))
        Ms = sorted(set(r["m"] for r in rows))
        path_kind = rows[0]["path_kind"]

    # Fixed parameters for each subplot
    N_fixed_for_d = max(Ns)
    N_fixed_for_m = max(Ns)
    d_fixed_for_N = max(Ds)
    d_fixed_for_m = max(Ds)
    m_fixed_for_N = max(Ms)
    m_fixed_for_d = max(Ms)

    series_keys = sorted(set(_series_key(r) for r in rows))
    op_order = _ordered_operations(rows)

    fig, axes = plt.subplots(
        3,
        len(op_order),
        figsize=(5 * len(op_order), 12),
        squeeze=False,
    )

    for row_idx, vary in enumerate(["N", "d", "m"]):
        for col_idx, op in enumerate(op_order):
            ax = axes[row_idx, col_idx]

            # Determine x-axis values and fixed parameters
            if vary == "N":
                xs = Ns
                d_fix = d_fixed_for_N
                m_fix = m_fixed_for_N
                xlabel = "N (number of points)"
            elif vary == "d":
                xs = Ds
                d_fix = None
                m_fix = m_fixed_for_d
                xlabel = "d (dimension)"
            else:  # vary m
                xs = Ms
                d_fix = d_fixed_for_m
                m_fix = None
                xlabel = "m (signature level)"

            plotted_any = False

            # Calculate speedups
            for series_key in series_keys:
                speedups = []
                xs_effective = []

                for x in xs:
                    if vary == "N":
                        N, d, m = x, d_fix, m_fix
                    elif vary == "d":
                        N, d, m = N_fixed_for_d, x, m_fix
                    else:  # vary m
                        N, d, m = N_fixed_for_m, d_fix, x

                    # Get times for all libraries at this point
                    times = {}
                    for candidate_key in series_keys:
                        t = get_time(rows, candidate_key, N, d, m, path_kind, op)
                        if t is not None and t > 0.0:
                            times[candidate_key] = t

                    if not times:
                        continue

                    # Calculate baseline
                    if baseline == "slowest":
                        baseline_time = max(times.values())
                    elif baseline == "fastest":
                        baseline_time = min(times.values())
                    else:
                        matching_keys = [
                            key
                            for key in times
                            if baseline in (key[0], _series_label(key))
                        ]
                        baseline_time = (
                            times[matching_keys[0]]
                            if len(matching_keys) == 1
                            else max(times.values())
                        )

                    # Calculate speedup for this library
                    if series_key in times:
                        speedup = baseline_time / times[series_key]
                        xs_effective.append(x)
                        speedups.append(speedup)

                if len(xs_effective) >= 2:
                    ax.plot(
                        xs_effective,
                        speedups,
                        marker="o",
                        label=_series_label(series_key),
                    )
                    plotted_any = True

            if not plotted_any:
                ax.set_visible(False)
                continue

            # Add reference line at speedup=1.0
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

            # Configure axes
            ax.set_xlabel(xlabel)
            if xs:
                ax.set_xticks(xs)

            ax.set_ylabel(f"Speedup vs {baseline}")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # Title
            title = f"{op}, vary {vary}"
            if vary == "N":
                title += f" (d={d_fixed_for_N}, m={m_fixed_for_N})"
            elif vary == "d":
                title += f" (N={N_fixed_for_d}, m={m_fixed_for_d})"
            else:
                title += f" (N={N_fixed_for_m}, d={d_fixed_for_m})"
            ax.set_title(title)

            # Legend only on top row
            if row_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=8)

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / f"plot_speedup_{baseline}.png"

    fig.savefig(output_path, dpi=300)
    print(f"Speedup plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_profile_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate performance profile plot showing how often each library is competitive.

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    operations = _ordered_operations(rows)
    series_keys = sorted(set(_series_key(r) for r in rows))

    # Group by complete workload identity.
    benchmarks = {}
    for r in rows:
        key = (
            r["N"],
            r["d"],
            r["m"],
            r["path_kind"],
            r["operation"],
            r["batch_size"],
            r["seed"],
        )
        if key not in benchmarks:
            benchmarks[key] = {}
        benchmarks[key][_series_key(r)] = r["t_ms"]

    # Calculate performance ratios for each benchmark
    num_ops = len(operations)
    fig, axes = plt.subplots(1, num_ops, figsize=(8 * num_ops, 6))
    if num_ops == 1:
        axes = [axes]

    for op_idx, operation in enumerate(operations):
        ax = axes[op_idx]

        # Filter benchmarks for this operation
        op_benchmarks = {k: v for k, v in benchmarks.items() if k[4] == operation}

        if not op_benchmarks:
            ax.set_visible(False)
            continue

        # For each library, calculate ratio to best time for each benchmark
        library_ratios = {series_key: [] for series_key in series_keys}

        for bench_key, times in op_benchmarks.items():
            if not times:
                continue

            best_time = min(times.values())

            for series_key in series_keys:
                if series_key in times:
                    ratio = times[series_key] / best_time
                    library_ratios[series_key].append(ratio)

        # Sort ratios and plot performance profile
        for series_key in series_keys:
            if not library_ratios[series_key]:
                continue

            ratios = sorted(library_ratios[series_key])
            # Y-axis: fraction of benchmarks where ratio <= x
            y_values = np.arange(1, len(ratios) + 1) / len(ratios)

            ax.plot(
                ratios,
                y_values,
                marker="o",
                markersize=4,
                label=_series_label(series_key),
            )

        ax.set_xlabel("Performance ratio (time / best_time)")
        ax.set_ylabel("Fraction of benchmarks")
        ax.set_title(f"{operation} - Performance Profile")
        ax.set_xlim(left=1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add vertical line at ratio=2 (2x slower than best)
        ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / "plot_profile.png"

    fig.savefig(output_path, dpi=300)
    print(f"Performance profile plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_box_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate box plots showing distribution of performance across all benchmarks.

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    operations = _ordered_operations(rows)
    series_keys = sorted(set(_series_key(r) for r in rows))

    # Create subplots for each operation
    num_ops = len(operations)
    fig, axes = plt.subplots(1, num_ops, figsize=(6 * num_ops, 6))
    if num_ops == 1:
        axes = [axes]

    for op_idx, operation in enumerate(operations):
        ax = axes[op_idx]

        # Get data for this operation
        op_rows = [r for r in rows if r["operation"] == operation]

        if not op_rows:
            ax.set_visible(False)
            continue

        # Organize data by library
        data_by_lib = {series_key: [] for series_key in series_keys}
        for r in op_rows:
            data_by_lib[_series_key(r)].append(r["t_ms"])

        # Filter out libraries with no data
        plot_data = [
            data_by_lib[series_key]
            for series_key in series_keys
            if data_by_lib[series_key]
        ]
        plot_labels = [
            _series_label(series_key)
            for series_key in series_keys
            if data_by_lib[series_key]
        ]

        if not plot_data:
            ax.set_visible(False)
            continue

        # Create box plot
        bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{operation} - Distribution Across All Benchmarks")
        ax.grid(True, axis="y", alpha=0.3)

        # Rotate x labels if needed
        if len(plot_labels) > 3:
            ax.set_xticklabels(plot_labels, rotation=45, ha="right")

        # Use log scale if data spans multiple orders of magnitude
        if plot_data:
            all_vals = [v for lib_data in plot_data for v in lib_data]
            if max(all_vals) / min(all_vals) > 100:
                ax.set_yscale("log")

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / "plot_box.png"

    fig.savefig(output_path, dpi=300)
    print(f"Box plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_comparison_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Legacy wrapper for make_line_plot (for backwards compatibility).

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = csv_path.parent / "comparison_3x3.png"
    return make_line_plot(csv_path, output_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from signature benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots for latest run
  python plotting.py --plot-type all

  # Generate heatmap for specific run
  python plotting.py runs/benchmark_20251201_221922 --plot-type heatmap

  # Generate speedup plot with custom baseline
  python plotting.py --plot-type speedup --baseline iisignature

  # List available plot types
  python plotting.py --list-plots
        """
    )

    parser.add_argument(
        "run_dir",
        nargs="?",
        type=str,
        help="Path to benchmark run directory or results.csv (defaults to latest run)"
    )

    parser.add_argument(
        "--plot-type",
        "-t",
        type=str,
        default="all",
        choices=["line", "heatmap", "speedup", "profile", "box", "all"],
        help="Type of plot to generate (default: all)"
    )

    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        default="slowest",
        help="Baseline for speedup plot: 'slowest', 'fastest', or library name (default: slowest)"
    )

    parser.add_argument(
        "--list-plots",
        "-l",
        action="store_true",
        help="List available plot types and exit"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for plots (defaults to run directory)"
    )

    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Disable subplot titles for heatmap plots"
    )

    args = parser.parse_args()

    # List plot types if requested
    if args.list_plots:
        print("Available plot types:")
        print("  line     - 3x3 grid of line plots (original)")
        print("  heatmap  - Heatmap showing all parameter combinations")
        print("  speedup  - Relative performance vs baseline")
        print("  profile  - Performance profile (competitiveness)")
        print("  box      - Box plots showing distribution")
        print("  all      - Generate all plot types")
        sys.exit(0)

    # Determine CSV path
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
        # Use latest run
        latest = get_latest_run()
        if not latest:
            print("Error: No benchmark runs found in 'runs/' directory")
            sys.exit(1)
        run_dir = latest
        csv_path = run_dir / "results.csv"

    # Verify CSV exists
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = run_dir

    # Generate plots
    plot_funcs = {
        "line": (make_line_plot, {}),
        "heatmap": (make_heatmap_plot, {"show_titles": not args.no_titles}),
        "speedup": (make_speedup_plot, {"baseline": args.baseline}),
        "profile": (make_profile_plot, {}),
        "box": (make_box_plot, {}),
    }

    if args.plot_type == "all":
        print(f"\nGenerating all plot types in: {output_dir}\n")
        for plot_name, (plot_func, kwargs) in plot_funcs.items():
            try:
                output_path = output_dir / f"plot_{plot_name}.png"
                if plot_name == "speedup":
                    output_path = output_dir / f"plot_speedup_{args.baseline}.png"
                plot_func(csv_path, output_path, **kwargs)
            except Exception as e:
                print(f"Error generating {plot_name} plot: {e}")
    else:
        plot_func, kwargs = plot_funcs[args.plot_type]
        try:
            output_path = output_dir / f"plot_{args.plot_type}.png"
            if args.plot_type == "speedup":
                output_path = output_dir / f"plot_speedup_{args.baseline}.png"
            plot_func(csv_path, output_path, **kwargs)
        except Exception as e:
            print(f"Error generating {args.plot_type} plot: {e}")
            sys.exit(1)

    print(f"\nDone! Plots saved to: {output_dir}")
