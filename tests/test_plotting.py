"""Unit tests for plotting label helpers."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotting
from plotting import (
    _annotate_heatmap,
    _format_heatmap_library_axis,
    _format_heatmap_param_axis,
    _format_number,
    _heatmap_color_scale,
    _heatmap_color_scales_by_operation,
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


def test_heatmap_library_tick_labels_use_enlarged_font():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plotting._configure_heatmap_axes(
        ax,
        {
            "libraries": ["iisignature", "pysiglib"],
            "params": [(1000, 2)],
            "show_library_axis": True,
            "show_param_axis": False,
        },
        show_titles=False,
    )

    assert {label.get_fontsize() for label in ax.get_xticklabels()} == {
        float(plotting.HEATMAP_LIBRARY_TICK_LABEL_FONT_SIZE),
    }
    plt.close(fig)


def test_heatmap_color_scale_is_logarithmic():
    norm = _heatmap_color_scale([0.26, 1.6])

    assert isinstance(norm, LogNorm)
    assert norm.vmin == 0.26
    assert norm.vmax == 1.6


def test_heatmap_color_scale_ignores_nonpositive_values():
    norm = _heatmap_color_scale([-1.0, 0.0, 0.26, 3.9])

    assert norm.vmin == 0.26
    assert norm.vmax == 3.9


def test_heatmap_color_scales_are_shared_by_backend_but_not_operation():
    rows = [
        {"operation": "signaturekernel", "backend": "cpu", "t_ms": 20.0},
        {"operation": "signaturekernel", "backend": "gpu", "t_ms": 0.5},
        {
            "operation": "signaturekernel_backprop",
            "backend": "cpu",
            "t_ms": 200.0,
        },
        {
            "operation": "signaturekernel_backprop",
            "backend": "gpu",
            "t_ms": 10.0,
        },
    ]

    scales = _heatmap_color_scales_by_operation(rows)

    assert scales["signaturekernel"].vmin == 0.5
    assert scales["signaturekernel"].vmax == 20.0
    assert scales["signaturekernel_backprop"].vmin == 10.0
    assert scales["signaturekernel_backprop"].vmax == 200.0


def test_make_heatmap_plot_uses_backend_shared_operation_scale(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "1000,2,3,32,brownian,signaturekernel,cpu,python,cpu_lib,method,path,20,0",
            "1000,4,3,32,brownian,signaturekernel,cpu,python,cpu_lib,method,path,10,0",
            "1000,2,3,32,brownian,signaturekernel,gpu,python,gpu_lib,method,path,1,0",
            "1000,4,3,32,brownian,signaturekernel,gpu,python,gpu_lib,method,path,0.5,0",
        ]),
        encoding="utf-8",
    )
    captured_limits = []

    def capture_limits(
        ax,
        matrix,
        norm,
        failure_labels=None,
        fontsize=plotting.HEATMAP_ANNOTATION_FONT_SIZE,
    ):
        captured_limits.append((norm.vmin, norm.vmax))

    monkeypatch.setattr(plotting, "_annotate_heatmap", capture_limits)

    make_heatmap_plot(
        csv_path,
        tmp_path / "cpu.pdf",
        operation="signaturekernel",
        backend="cpu",
    )
    make_heatmap_plot(
        csv_path,
        tmp_path / "gpu.pdf",
        operation="signaturekernel",
        backend="gpu",
    )

    assert captured_limits == [(0.5, 20.0), (0.5, 20.0)]


def test_annotate_heatmap_labels_oom_failures():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    matrix = np.array([[np.nan, 1.0]])
    failure_labels = np.array([["OOM", ""]], dtype=object)

    _annotate_heatmap(
        ax,
        matrix,
        _heatmap_color_scale([1.0]),
        failure_labels,
    )

    assert [text.get_text() for text in ax.texts] == ["OOM", "1"]
    assert ax.texts[0].get_color() == "#991B1B"
    plt.close(fig)


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
    output_path = tmp_path / "plot_heatmap.pdf"

    result_path = make_heatmap_plot(
        csv_path,
        output_path,
        show_titles=False,
        operation="logsignature",
    )

    assert result_path == output_path
    assert output_path.exists()
    assert not (tmp_path / "plot_heatmaps").exists()


def test_failure_labels_do_not_treat_random_errors_as_oom(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("", encoding="utf-8")
    (tmp_path / "failed_tasks.csv").write_text(
        "\n".join([
            "task_id,library,backend,operation,N,d,m,batch_size,error_type,reason",
            "id,pysiglib,cpu,signature,1000,2,3,32,RuntimeError,zoom worker disconnected",
        ]),
        encoding="utf-8",
    )

    assert plotting._load_failure_labels(csv_path) == {}


def test_make_heatmap_plot_loads_oom_failures(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "10000,2,3,256,brownian,signature,cpu,python,pysiglib,signature,array,100,0",
            "10000,2,3,256,brownian,signature,cpu,python,other,signature,array,120,0",
            "10000,16,3,256,brownian,signature,cpu,python,other,signature,array,140,0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "failed_tasks.csv").write_text(
        "\n".join([
            "task_id,library,backend,operation,N,d,m,batch_size,error_type,reason",
            "id,pysiglib,cpu,signature,10000,16,3,256,BenchmarkWorkerOOM,worker received SIGKILL",
        ]),
        encoding="utf-8",
    )
    captured_labels = []
    original_annotate = plotting._annotate_heatmap

    def capture_labels(
        ax,
        matrix,
        norm,
        failure_labels=None,
        fontsize=plotting.HEATMAP_ANNOTATION_FONT_SIZE,
    ):
        captured_labels.append(failure_labels.copy())
        original_annotate(ax, matrix, norm, failure_labels, fontsize)

    monkeypatch.setattr(plotting, "_annotate_heatmap", capture_labels)

    make_heatmap_plot(csv_path, tmp_path / "heatmap.pdf")

    assert any((labels == "OOM").any() for labels in captured_labels)


def test_make_heatmap_plot_includes_oom_only_library(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "1000,2,3,32,brownian,signaturekernel_backprop,cpu,python,pysiglib,method,path,10,0",
            "1000,4,3,32,brownian,signaturekernel_backprop,cpu,python,pysiglib,method,path,11,0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "failed_tasks.csv").write_text(
        "\n".join([
            "task_id,library,backend,operation,N,d,m,batch_size,error_type,reason",
            "id1,other,cpu,signaturekernel_backprop,1000,2,3,32,BenchmarkWorkerOOM,worker received SIGKILL",
            "id2,other,cpu,signaturekernel_backprop,1000,4,3,32,BenchmarkWorkerOOM,worker received SIGKILL",
        ]),
        encoding="utf-8",
    )
    captured_labels = []

    def capture_labels(
        ax,
        matrix,
        norm,
        failure_labels=None,
        fontsize=plotting.HEATMAP_ANNOTATION_FONT_SIZE,
    ):
        captured_labels.append(failure_labels.copy())

    monkeypatch.setattr(plotting, "_annotate_heatmap", capture_labels)

    make_heatmap_plot(
        csv_path,
        tmp_path / "heatmap.pdf",
        operation="signaturekernel_backprop",
        backend="cpu",
    )

    assert captured_labels[0].shape == (2, 2)
    assert np.count_nonzero(captured_labels[0] == "OOM") == 2


def test_make_heatmap_plot_supports_failure_only_selection(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "1000,2,3,32,brownian,signaturekernel_backprop,gpu,python,sigkernel,method,path,10,0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "failed_tasks.csv").write_text(
        "\n".join([
            "task_id,library,backend,operation,N,d,m,batch_size,error_type,reason",
            "id1,pysiglib,cpu,signaturekernel_backprop,1000,2,3,32,BenchmarkWorkerOOM,worker received SIGKILL",
            "id2,pysiglib,cpu,signaturekernel_backprop,1000,4,3,32,BenchmarkWorkerOOM,worker received SIGKILL",
            "id3,other,cpu,signaturekernel_backprop,1000,2,3,32,BenchmarkWorkerOOM,worker received SIGKILL",
            "id4,other,cpu,signaturekernel_backprop,1000,4,3,32,BenchmarkWorkerOOM,worker received SIGKILL",
        ]),
        encoding="utf-8",
    )
    captured = []

    def reject_colorbar(*args, **kwargs):
        raise AssertionError("Failure-only plot must not show a runtime color bar")

    def capture_labels(
        ax,
        matrix,
        norm,
        failure_labels=None,
        fontsize=plotting.HEATMAP_ANNOTATION_FONT_SIZE,
    ):
        captured.append((matrix.copy(), failure_labels.copy()))

    monkeypatch.setattr(plotting, "_annotate_heatmap", capture_labels)
    monkeypatch.setattr(plotting.plt.Figure, "colorbar", reject_colorbar)
    output_path = tmp_path / "heatmap.pdf"

    make_heatmap_plot(
        csv_path,
        output_path,
        operation="signaturekernel_backprop",
        backend="cpu",
    )

    assert output_path.exists()
    assert len(captured) == 1
    matrix, labels = captured[0]
    assert matrix.shape == (2, 2)
    assert np.isnan(matrix).all()
    assert np.count_nonzero(labels == "OOM") == 4


def test_make_heatmap_plot_labels_cuda_tree_launch_limit(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "1000,2,3,256,brownian,branchedsignature_planar,gpu,python,pysiglib,method,path,10,0",
            "1000,4,3,256,brownian,branchedsignature_planar,gpu,python,pysiglib,method,path,11,0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "skipped_tasks.csv").write_text(
        "\n".join([
            "task_id,library,backend,operation,N,d,m,batch_size,reason",
            "id,pysiglib,gpu,branchedsignature_planar,1000,8,3,256,CUDA branched sig: num_trees > 1024 not supported",
        ]),
        encoding="utf-8",
    )
    captured_labels = []

    def capture_labels(
        ax,
        matrix,
        norm,
        failure_labels=None,
        fontsize=plotting.HEATMAP_ANNOTATION_FONT_SIZE,
    ):
        captured_labels.append(failure_labels.copy())

    monkeypatch.setattr(plotting, "_annotate_heatmap", capture_labels)

    make_heatmap_plot(
        csv_path,
        tmp_path / "heatmap.pdf",
        operation="branchedsignature_planar",
        backend="gpu",
    )

    assert captured_labels[0].shape == (3, 1)
    assert captured_labels[0][2, 0] == "CUDA\nLIMIT"


def test_make_heatmap_plot_labels_worker_segfault_as_crash(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "10000,2,3,256,brownian,logsignature,cpu,python,other,method,path,10,0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "failed_tasks.csv").write_text(
        "\n".join([
            "task_id,library,backend,operation,N,d,m,batch_size,error_type,reason",
            "id,signatory,cpu,logsignature,10000,2,3,256,RuntimeError,signatory: worker exited with code 139",
        ]),
        encoding="utf-8",
    )
    captured_labels = []

    def capture_labels(
        ax,
        matrix,
        norm,
        failure_labels=None,
        fontsize=plotting.HEATMAP_ANNOTATION_FONT_SIZE,
    ):
        captured_labels.append(failure_labels.copy())

    monkeypatch.setattr(plotting, "_annotate_heatmap", capture_labels)

    make_heatmap_plot(
        csv_path,
        tmp_path / "heatmap.pdf",
        operation="logsignature",
        backend="cpu",
    )

    assert captured_labels[0].shape == (1, 2)
    assert np.count_nonzero(captured_labels[0] == "Crash") == 1


def test_make_heatmap_plot_classifies_only_known_cuda_limits(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "10000,8,2,256,brownian,sig_backprop,gpu,python,other,method,path,9,0",
            "10000,8,3,256,brownian,sig_backprop,gpu,python,other,method,path,10,0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "failed_tasks.csv").write_text(
        "\n".join([
            "task_id,library,backend,operation,N,d,m,batch_size,error_type,reason",
            "id1,log-signatures-pytorch,gpu,sig_backprop,10000,16,3,256,RuntimeError,CUDA error: device not ready",
            "id2,log-signatures-pytorch,gpu,sig_backprop,10000,16,2,256,RuntimeError,CUDA driver error: invalid argument",
        ]),
        encoding="utf-8",
    )
    captured_labels = []

    def capture_labels(
        ax,
        matrix,
        norm,
        failure_labels=None,
        fontsize=plotting.HEATMAP_ANNOTATION_FONT_SIZE,
    ):
        captured_labels.append(failure_labels.copy())

    monkeypatch.setattr(plotting, "_annotate_heatmap", capture_labels)

    make_heatmap_plot(
        csv_path,
        tmp_path / "heatmap.pdf",
        operation="sig_backprop",
        backend="gpu",
    )

    all_labels = np.concatenate([labels.ravel() for labels in captured_labels])
    assert np.count_nonzero(all_labels == "OOM") == 0
    assert (
        np.count_nonzero(
            all_labels == "CUDA\nLIMIT"
        )
        == 1
    )


def test_plot_series_keep_cpu_and_gpu_variants_distinct(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join([
            "N,d,m,batch_size,path_kind,operation,backend,language,library,method,path_type,t_ms,alloc_bytes",
            "32,2,2,256,fbm,signature,cpu,python,pysiglib,signature,numpy.ndarray,1.0,0",
            "32,2,2,256,fbm,signature,gpu,python,pysiglib,signature,torch.Tensor,0.5,0",
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


def test_branched_logsignature_operation_labels_are_publication_friendly():
    assert plotting._operation_label("branchedlogsignature_nonplanar") == (
        "Non-planar branched log signature"
    )
    assert plotting._operation_label(
        "branchedlogsignature_planar_backprop"
    ) == "Planar branched log signature backprop"
