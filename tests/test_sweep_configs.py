"""Checks for complete sweep and registry dependency configuration."""

import tomllib
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

SIGNATURE_OPERATIONS = {
    "signature",
    "logsignature",
    "sig_backprop",
    "branchedsignature_nonplanar",
    "branchedsignature_planar",
}
KERNEL_OPERATIONS = {"signaturekernel"}
BACKPROP_OPERATIONS = {
    "sig_backprop",
    "logsignature_backprop",
    "branchedsignature_nonplanar_backprop",
    "branchedsignature_planar_backprop",
    "signaturekernel_backprop",
}
PAPER_SIGNATURE_LIBRARIES = {
    "log-signatures-pytorch",
    "signatory",
    "pathsig",
    "pysiglib",
    "stochastax",
    "signax",
    "tensordev",
    "keras_sig",
}
PAPER_LOGSIGNATURE_LIBRARIES = PAPER_SIGNATURE_LIBRARIES - {"keras_sig"}
PAPER_BRANCHED_LIBRARIES = {"pysiglib", "stochastax"}
PAPER_KERNEL_LIBRARIES = {"pysiglib", "polysigkernel"}
PAPER_JAX_LIBRARIES = {
    "polysigkernel",
    "stochastax",
    "signax",
    "tensordev",
    "keras_sig",
}
PAPER_TORCH_CPU_LIBRARIES = {"log-signatures-pytorch", "signatory"}


def _load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_complete_sweep_operation_sets():
    signatures = _load_yaml("benchmark_sweep.yaml")
    kernels = _load_yaml("signature_kernel_sweep.yaml")
    combined = _load_yaml("combined_sweep.yaml")
    registry = _load_yaml("libraries_registry.yaml")

    assert set(signatures["operations"]) == SIGNATURE_OPERATIONS
    assert set(kernels["operations"]) == KERNEL_OPERATIONS
    assert set(combined["operations"]) == (
        SIGNATURE_OPERATIONS | KERNEL_OPERATIONS | BACKPROP_OPERATIONS
    )
    registry_operations = {
        operation
        for library in registry["libraries"].values()
        for operation in library["operations"]
    }
    assert registry_operations <= set(combined["operations"])


def test_every_registry_extra_is_defined():
    registry = _load_yaml("libraries_registry.yaml")
    with (REPO_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    defined_extras = set(project["project"]["optional-dependencies"])
    registry_extras = {
        extra
        for library in registry["libraries"].values()
        for extra in library.get("extras", [])
    }

    assert registry_extras <= defined_extras


def _assert_paper_settings(paper: dict, expected_N: int = 1000) -> None:
    assert paper["Ns"] == [expected_N]
    assert paper["Ds"] == [2, 4, 8, 16]
    assert paper["repeats"] == 3
    assert paper["warmup_iterations"] == 1
    assert paper["timing_statistic"] == "min"
    assert paper["worker_memory_limit_gb"] == 16
    assert "call_timeout_seconds" not in paper
    assert paper["clear_input_caches_after_task"] is True


def test_problem_specific_paper_sweeps():
    signatures = _load_yaml("paper_signatures_sweep.yaml")
    logsignatures = _load_yaml("paper_logsignatures_sweep.yaml")
    branched = _load_yaml("paper_branched_signatures_sweep.yaml")
    kernels = _load_yaml("paper_signature_kernel_sweep.yaml")

    _assert_paper_settings(signatures, expected_N=10000)
    for paper in (logsignatures, branched, kernels):
        _assert_paper_settings(paper)

    assert set(signatures["libraries"]) == PAPER_SIGNATURE_LIBRARIES
    assert set(signatures["operations"]) == {
        "signature",
        "sig_backprop",
    }
    assert set(logsignatures["libraries"]) == PAPER_LOGSIGNATURE_LIBRARIES
    assert set(logsignatures["operations"]) == {
        "logsignature",
        "logsignature_backprop",
    }
    assert set(branched["libraries"]) == PAPER_BRANCHED_LIBRARIES
    assert set(branched["operations"]) == {
        "branchedsignature_nonplanar",
        "branchedsignature_nonplanar_backprop",
        "branchedsignature_planar",
        "branchedsignature_planar_backprop",
    }
    assert set(kernels["libraries"]) == PAPER_KERNEL_LIBRARIES
    assert set(kernels["operations"]) == {
        "signaturekernel",
        "signaturekernel_backprop",
    }

    registry = _load_yaml("libraries_registry.yaml")["libraries"]
    for paper in (signatures, logsignatures, branched, kernels):
        operations = set(paper["operations"])
        for library_name in paper["libraries"]:
            assert operations <= set(registry[library_name]["operations"])

        pysiglib_config = paper["library_configs"]["pysiglib"]
        assert pysiglib_config["backend_configs"]["cpu"]["n_jobs"] == -1


def test_paper_cpu_frameworks_enable_all_available_threads():
    registry = _load_yaml("libraries_registry.yaml")["libraries"]

    for library_name in PAPER_JAX_LIBRARIES:
        cpu_backend = next(
            backend
            for backend in registry[library_name]["backends"]
            if backend["name"] == "cpu"
        )
        xla_flags = cpu_backend["env"]["XLA_FLAGS"]
        assert "--xla_cpu_multi_thread_eigen=true" in xla_flags
        assert "intra_op_parallelism_threads={cpu_count}" in xla_flags

    for library_name in PAPER_TORCH_CPU_LIBRARIES:
        cpu_backend = next(
            backend
            for backend in registry[library_name]["backends"]
            if backend["name"] == "cpu"
        )
        assert cpu_backend["env"]["OMP_NUM_THREADS"] == "{cpu_count}"
        assert cpu_backend["env"]["MKL_NUM_THREADS"] == "{cpu_count}"
