"""Unit tests for plotting label helpers."""

import sys
from pathlib import Path

import matplotlib
from matplotlib.colors import LogNorm

matplotlib.use("Agg")

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from plotting import (
    _format_heatmap_library_axis,
    _format_heatmap_param_axis,
    _format_number,
    _heatmap_color_scale,
    _rows_for_operation,
    _series_key,
    load_results,
    make_heatmap_plot,
)


def test_format_number_avoids_scientific_notation():
    assert _format_number(0.0264) == "0.0264"
    assert _format_number(170.0) == "170"
    assert _format_number(4300.0) == "4300"


def test_heatmap_param_axis_omits_fixed_N():
    labels, ylabel, fixed_title, show_axis = _format_heatmap_param_axis([
        (512, 2),
        (512, 3),
    ])

    assert labels == ["2", "3"]
    assert ylabel == "$d$"
    assert fixed_title == "$N=512$"
    assert show_axis


def test_heatmap_param_axis_omits_fixed_d():
    labels, ylabel, fixed_title, show_axis = _format_heatmap_param_axis([
        (256, 4),
        (512, 4),
    ])

    assert labels == ["256", "512"]
    assert ylabel == "$N$"
    assert fixed_title == "$d=4$"
    assert show_axis


def test_heatmap_param_axis_hides_single_fixed_point():
    labels, ylabel, fixed_title, show_axis = _format_heatmap_param_axis([(512, 4)])

    assert labels == [""]
    assert ylabel == ""
    assert fixed_title == "$N=512, d=4$"
    assert not show_axis


def test_heatmap_library_axis_hides_single_library_tick():
    title, show_axis = _format_heatmap_library_axis(["pysiglib"])

    assert title == "pysiglib"
    assert not show_axis


def test_heatmap_library_axis_shows_multiple_library_ticks():
    title, show_axis = _format_heatmap_library_axis(["iisignature", "pysiglib"])

    assert title == ""
    assert show_axis


def test_heatmap_color_scale_is_logarithmic():
    norm = _heatmap_color_scale([0.26, 1.6])

    assert isinstance(norm, LogNorm)
    assert norm.vmin == 0.26
    assert norm.vmax == 1.6


def test_heatmap_color_scale_ignores_nonpositive_values():
    norm = _heatmap_color_scale([-1.0, 0.0, 0.26, 3.9])

    assert norm.vmin == 0.26
    assert norm.vmax == 3.9


def test_make_heatmap_plot_saves_one_operation_level_figure(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "512,2,2,1,fbm,logsignature,gpu,python,pysiglib,method,path,0.3,0",
            "512,3,2,1,fbm,logsignature,gpu,python,pysiglib,method,path,0.4,0",
            "512,2,2,1,fbm,branchedsignature_planar,gpu,python,pysiglib,method,path,0.6,0",
            "512,3,2,1,fbm,branchedsignature_planar,gpu,python,pysiglib,method,path,0.8,0",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "plot_heatmap.png"

    result_path = make_heatmap_plot(
        csv_path,
        output_path,
        show_titles=False,
        operation="logsignature",
    )

    assert result_path == output_path
    assert output_path.exists()
    assert not (tmp_path / "plot_heatmaps").exists()


def test_plot_series_keep_cpu_and_gpu_variants_distinct(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "32,2,2,256,fbm,signature,cpu,python,pysiglib,signature,jax.Array,1.0,0",
            "32,2,2,256,fbm,signature,gpu,python,pysiglib,signature,jax.Array,0.5,0",
        ]),
        encoding="utf-8",
    )

    rows = load_results(csv_path)

    assert len({_series_key(row) for row in rows}) == 2


def test_rows_for_operation_keeps_functions_separate():
    rows = [
        {"operation": "signature", "backend": "cpu"},
        {"operation": "logsignature", "backend": "cpu"},
        {"operation": "logsignature", "backend": "gpu"},
        {"operation": "branchedsignature_nonplanar", "backend": "gpu"},
    ]

    assert _rows_for_operation(rows, "logsignature") == [
        {"operation": "logsignature", "backend": "cpu"},
        {"operation": "logsignature", "backend": "gpu"},
    ]
    assert _rows_for_operation(rows, "logsignature", "gpu") == [
        {"operation": "logsignature", "backend": "gpu"},
    ]
    assert _rows_for_operation(rows, None) == rows
