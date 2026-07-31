"""Checks for complete sweep and registry dependency configuration."""

import tomllib
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

SIGNATURE_OPERATIONS = {
    "signature",
    "logsignature",
    "sigdiff",
    "branchedsignature_nonplanar",
    "branchedsignature_planar",
}
KERNEL_OPERATIONS = {"signaturekernel"}


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
    assert set(combined["operations"]) == SIGNATURE_OPERATIONS | KERNEL_OPERATIONS
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
