"""Checks for complete sweep and registry dependency configuration."""

import tomllib
from pathlib import Path

import yaml

import run_benchmarks


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

SIGNATURE_OPERATIONS = {
    "signature",
    "logsignature",
    "sig_backprop",
    "branchedsignature_nonplanar",
    "branchedsignature_planar",
    "branchedlogsignature_nonplanar",
    "branchedlogsignature_planar",
}
KERNEL_OPERATIONS = {"signaturekernel"}
BACKPROP_OPERATIONS = {
    "sig_backprop",
    "logsignature_backprop",
    "branchedsignature_nonplanar_backprop",
    "branchedsignature_planar_backprop",
    "branchedlogsignature_nonplanar_backprop",
    "branchedlogsignature_planar_backprop",
    "signaturekernel_backprop",
}
PAPER_SIGNATURE_LIBRARIES = {
    "iisignature",
    "log-signatures-pytorch",
    "signatory",
    "pathsig",
    "pysiglib",
    "signax",
    "tensordev",
    "keras_sig",
    "chen-signatures",
}
PAPER_LOGSIGNATURE_LIBRARIES = (
    PAPER_SIGNATURE_LIBRARIES
    | {"signature-rs"}
) - {"keras_sig", "tensordev"}
PAPER_BRANCHED_LIBRARIES = {"pysiglib", "stochastax"}
PAPER_FINITE_DIFFERENCE_KERNEL_LIBRARIES = {
    "pysiglib",
    "sigkerax",
    "sigkernel",
}
PAPER_POLYNOMIAL_KERNEL_LIBRARIES = {
    "pysiglib",
    "polysigkernel",
}
PAPER_JAX_LIBRARIES = {
    "polysigkernel",
    "sigkerax",
    "signax",
    "stochastax",
    "tensordev",
    "keras_sig",
}
PAPER_TORCH_CPU_LIBRARIES = {
    "log-signatures-pytorch",
    "signatory",
}


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


def test_combined_runner_runs_only_paper_heatmap_sweeps():
    assert run_benchmarks.PAPER_SWEEPS == (
        "paper_signatures_sweep.yaml",
        "paper_logsignatures_sweep.yaml",
        "paper_branched_signatures_sweep.yaml",
        "paper_branched_logsignatures_sweep.yaml",
        "paper_signature_kernel_sweep.yaml",
        "paper_polynomial_signature_kernel_sweep.yaml",
    )

    finite_difference = _load_yaml("paper_signature_kernel_sweep.yaml")
    polynomial = _load_yaml("paper_polynomial_signature_kernel_sweep.yaml")

    assert finite_difference["sig_kernel_method"] == "finite_difference"
    assert "polysigkernel" not in finite_difference["libraries"]
    assert polynomial["sig_kernel_method"] == "polynomial"
    assert set(polynomial["libraries"]) == {"pysiglib", "polysigkernel"}


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


def test_pysiglib_uses_the_local_source_checkout():
    with (REPO_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    sources = project["tool"]["uv"]["sources"]
    assert sources["pysiglib"] == {
        "path": "../pySigLib"
    }
    assert sources["pysiglib-cuda"] == {
        "path": "../pySigLib/plugins/cuda"
    }


def test_runtime_sweeps_use_at_least_ten_repetitions():
    for config_path in CONFIG_DIR.glob("*.yaml"):
        config = _load_yaml(config_path.name)
        if "repeats" in config:
            assert config["repeats"] >= 10


def _assert_paper_settings(
    paper: dict,
    *,
    expected_N: int = 1000,
    expected_Ms: tuple[int, ...] = (2, 3),
    expected_batch_size: int = 256,
) -> None:
    assert paper["path_kind"] == "brownian"
    assert paper["seed"] == 20260529
    assert paper["Ns"] == [expected_N]
    assert paper["Ds"] == [2, 4, 8, 16]
    assert paper["Ms"] == list(expected_Ms)
    assert paper["backends"] == ["cpu", "gpu"]
    assert paper["batch_size"] == expected_batch_size
    assert paper["repeats"] == 10
    assert paper["warmup_iterations"] == 1
    assert paper["timing_statistic"] == "min"
    assert paper["worker_memory_limit_gb"] == 16
    assert "call_timeout_seconds" not in paper
    assert paper["clear_input_caches_after_task"] is True


def test_problem_specific_paper_sweeps():
    signatures = _load_yaml("paper_signatures_sweep.yaml")
    logsignatures = _load_yaml("paper_logsignatures_sweep.yaml")
    branched = _load_yaml("paper_branched_signatures_sweep.yaml")
    branched_logsignatures = _load_yaml(
        "paper_branched_logsignatures_sweep.yaml"
    )
    kernels = _load_yaml("paper_signature_kernel_sweep.yaml")
    polynomial_kernels = _load_yaml(
        "paper_polynomial_signature_kernel_sweep.yaml"
    )

    _assert_paper_settings(signatures, expected_N=10000)
    for paper in (logsignatures, branched, branched_logsignatures):
        _assert_paper_settings(paper)
    _assert_paper_settings(
        kernels,
        expected_Ms=(3,),
        expected_batch_size=32,
    )
    _assert_paper_settings(
        polynomial_kernels,
        expected_Ms=(3,),
        expected_batch_size=32,
    )

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
    assert set(branched_logsignatures["libraries"]) == PAPER_BRANCHED_LIBRARIES
    assert set(branched_logsignatures["operations"]) == {
        "branchedlogsignature_nonplanar",
        "branchedlogsignature_nonplanar_backprop",
        "branchedlogsignature_planar",
        "branchedlogsignature_planar_backprop",
    }
    assert set(kernels["libraries"]) == (
        PAPER_FINITE_DIFFERENCE_KERNEL_LIBRARIES
    )
    assert set(kernels["operations"]) == {
        "signaturekernel",
        "signaturekernel_backprop",
    }
    assert set(polynomial_kernels["libraries"]) == (
        PAPER_POLYNOMIAL_KERNEL_LIBRARIES
    )
    assert set(polynomial_kernels["operations"]) == {
        "signaturekernel",
        "signaturekernel_backprop",
    }
    assert kernels["library_configs"]["sigkerax"][
        "sig_kernel_refinement_factor"
    ] == 1
    assert kernels["library_configs"]["sigkernel"][
        "sig_kernel_dyadic_order"
    ] == 0
    assert kernels["library_configs"]["sigkernel"][
        "sig_kernel_max_batch"
    ] == 4
    assert kernels["library_configs"]["pysiglib"][
        "sig_kernel_dyadic_order"
    ] == kernels["library_configs"]["sigkernel"][
        "sig_kernel_dyadic_order"
    ]
    assert kernels["library_configs"]["pysiglib"][
        "sig_kernel_max_batch"
    ] == kernels["library_configs"]["sigkernel"][
        "sig_kernel_max_batch"
    ]
    assert kernels["library_configs"]["pysiglib"][
        "sig_kernel_method"
    ] == "finite_difference"
    assert polynomial_kernels["library_configs"]["pysiglib"][
        "sig_kernel_method"
    ] == "polynomial"
    assert polynomial_kernels["library_configs"]["pysiglib"][
        "sig_kernel_order"
    ] == polynomial_kernels["library_configs"]["polysigkernel"][
        "sig_kernel_order"
    ]

    registry = _load_yaml("libraries_registry.yaml")["libraries"]
    for paper in (
        signatures,
        logsignatures,
        branched,
        branched_logsignatures,
        kernels,
        polynomial_kernels,
    ):
        operations = set(paper["operations"])
        eligible_libraries = {
            library_name
            for library_name, library_config in registry.items()
            if operations & set(library_config["operations"])
        }
        assert set(paper["libraries"]) <= eligible_libraries
        for library_name in paper["libraries"]:
            assert operations & set(registry[library_name]["operations"])

        pysiglib_config = paper["library_configs"]["pysiglib"]
        assert pysiglib_config["backend_configs"]["cpu"]["n_jobs"] == -1


def test_finite_difference_kernel_settings_match_across_sweeps():
    for config_name in (
        "combined_sweep.yaml",
        "signature_kernel_sweep.yaml",
        "paper_signature_kernel_sweep.yaml",
    ):
        config = _load_yaml(config_name)
        library_configs = config["library_configs"]
        pysiglib = library_configs["pysiglib"]
        sigkernel = library_configs["sigkernel"]

        assert pysiglib["sig_kernel_dyadic_order"] == sigkernel[
            "sig_kernel_dyadic_order"
        ]
        assert library_configs["sigkerax"][
            "sig_kernel_refinement_factor"
        ] == 2 ** pysiglib["sig_kernel_dyadic_order"]
        for backend, dtype in (("cpu", "float64"), ("gpu", "float32")):
            assert pysiglib["backend_configs"][backend]["dtype"] == dtype
            assert sigkernel["backend_configs"][backend]["dtype"] == dtype
            assert library_configs["sigkerax"]["backend_configs"][backend][
                "dtype"
            ] == dtype


def test_polynomial_kernel_settings_match_across_sweeps():
    for config_name in (
        "polynomial_signature_kernel_sweep.yaml",
        "paper_polynomial_signature_kernel_sweep.yaml",
    ):
        config = _load_yaml(config_name)
        library_configs = config["library_configs"]
        assert set(config["libraries"]) == {
            "pysiglib",
            "polysigkernel",
        }
        assert library_configs["pysiglib"][
            "sig_kernel_method"
        ] == "polynomial"
        assert library_configs["pysiglib"][
            "sig_kernel_order"
        ] == library_configs["polysigkernel"][
            "sig_kernel_order"
        ]
        assert library_configs["pysiglib"]["dtype"] == "float32"
        assert library_configs["polysigkernel"]["dtype"] == "float32"


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
