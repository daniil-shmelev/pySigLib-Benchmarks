#!/usr/bin/env python3
"""Build CSV and LaTeX package-rank tables from the canonical benchmark runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("cpu", "gpu")
BENCHMARKS = (
    ("signature", "signature", "Signature"),
    ("signature", "sig_backprop", "Signature BP"),
    ("log_signature", "logsignature", "Log signature"),
    ("log_signature", "logsignature_backprop", "Log signature BP"),
    ("branched_signatures", "branchedsignature_nonplanar", "Branched sig. NP"),
    (
        "branched_signatures",
        "branchedsignature_nonplanar_backprop",
        "Branched sig. NP BP",
    ),
    ("branched_signatures", "branchedsignature_planar", "Branched sig. P"),
    (
        "branched_signatures",
        "branchedsignature_planar_backprop",
        "Branched sig. P BP",
    ),
    (
        "branched_log_signatures",
        "branchedlogsignature_nonplanar",
        "Branched log sig. NP",
    ),
    (
        "branched_log_signatures",
        "branchedlogsignature_nonplanar_backprop",
        "Branched log sig. NP BP",
    ),
    (
        "branched_log_signatures",
        "branchedlogsignature_planar",
        "Branched log sig. P",
    ),
    (
        "branched_log_signatures",
        "branchedlogsignature_planar_backprop",
        "Branched log sig. P BP",
    ),
    ("kernels", "signaturekernel", "Signature kernel"),
    ("kernels", "signaturekernel_backprop", "Signature kernel BP"),
)
FAMILIES = (
    ("Signatures", ("signature", "sig_backprop")),
    ("Log signatures", ("logsignature", "logsignature_backprop")),
    (
        "Branched",
        (
            "branchedsignature_nonplanar",
            "branchedsignature_nonplanar_backprop",
            "branchedsignature_planar",
            "branchedsignature_planar_backprop",
            "branchedlogsignature_nonplanar",
            "branchedlogsignature_nonplanar_backprop",
            "branchedlogsignature_planar",
            "branchedlogsignature_planar_backprop",
        ),
    ),
    ("Kernels", ("signaturekernel", "signaturekernel_backprop")),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _parameter_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        int(row["N"]),
        int(row["d"]),
        int(row["m"]),
        int(row.get("batch_size") or 1),
    )


def _geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _competition_ranks(
    scores: list[tuple[float, str]],
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    previous_score: float | None = None
    previous_rank = 0
    for position, (score, library) in enumerate(sorted(scores), start=1):
        if previous_score is None or not math.isclose(
            score,
            previous_score,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            previous_rank = position
            previous_score = score
        ranks[library] = previous_rank
    return ranks


def calculate_ranks(
    runs_root: Path,
) -> tuple[
    list[str],
    dict[tuple[str, str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
]:
    results_by_run: dict[str, list[dict[str, str]]] = {}
    attempts_by_run: dict[str, list[dict[str, str]]] = {}
    libraries: set[str] = set()

    for run_name, _, _ in BENCHMARKS:
        if run_name in results_by_run:
            continue
        run_dir = runs_root / run_name
        results = _read_csv(run_dir / "results.csv")
        attempts = list(results)
        attempts.extend(_read_csv(run_dir / "failed_tasks.csv"))
        attempts.extend(_read_csv(run_dir / "skipped_tasks.csv"))
        results_by_run[run_name] = results
        attempts_by_run[run_name] = attempts
        libraries.update(row["library"] for row in attempts)

    benchmark_ranks: dict[tuple[str, str, str], int] = {}
    for run_name, operation, _ in BENCHMARKS:
        results = results_by_run[run_name]
        attempts = attempts_by_run[run_name]
        for backend in BACKENDS:
            expected_parameters = {
                _parameter_key(row)
                for row in attempts
                if row["operation"] == operation and row["backend"] == backend
            }
            scores: list[tuple[float, str]] = []
            for library in libraries:
                runtimes = {
                    _parameter_key(row): float(row["t_ms"])
                    for row in results
                    if row["operation"] == operation
                    and row["backend"] == backend
                    and row["library"] == library
                }
                if expected_parameters and expected_parameters <= runtimes.keys():
                    score = _geometric_mean(
                        runtimes[parameters]
                        for parameters in expected_parameters
                    )
                    scores.append((score, library))
            for library, rank in _competition_ranks(scores).items():
                benchmark_ranks[(library, operation, backend)] = rank

    coverage: dict[tuple[str, str], int] = {}
    mean_ranks: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for backend in BACKENDS:
        for library in libraries:
            ranks = [
                benchmark_ranks[(library, operation, backend)]
                for _, operation, _ in BENCHMARKS
                if (library, operation, backend) in benchmark_ranks
            ]
            coverage[(library, backend)] = len(ranks)
            if ranks:
                mean_ranks[backend].append((sum(ranks) / len(ranks), library))

    overall_ranks: dict[tuple[str, str], int] = {}
    for backend in BACKENDS:
        for library, rank in _competition_ranks(mean_ranks[backend]).items():
            overall_ranks[(library, backend)] = rank

    return (
        sorted(libraries, key=str.casefold),
        benchmark_ranks,
        overall_ranks,
        coverage,
    )


def write_csv(
    output_path: Path,
    libraries: list[str],
    benchmark_ranks: dict[tuple[str, str, str], int],
    overall_ranks: dict[tuple[str, str], int],
    coverage: dict[tuple[str, str], int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Package"]
    for _, _, label in BENCHMARKS:
        fieldnames.extend([f"{label} CPU", f"{label} GPU"])
    fieldnames.extend(["Overall CPU", "Overall GPU"])

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for library in libraries:
            row: dict[str, Any] = {"Package": library}
            for _, operation, label in BENCHMARKS:
                for backend in BACKENDS:
                    row[f"{label} {backend.upper()}"] = benchmark_ranks.get(
                        (library, operation, backend),
                        "--",
                    )
            for backend in BACKENDS:
                rank = overall_ranks.get((library, backend))
                row[f"Overall {backend.upper()}"] = rank if rank is not None else "--"
            writer.writerow(row)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def write_latex(
    output_path: Path,
    libraries: list[str],
    benchmark_ranks: dict[tuple[str, str, str], int],
    overall_ranks: dict[tuple[str, str], int],
    coverage: dict[tuple[str, str], int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    family_ranks: dict[tuple[str, str, str], int] = {}
    for backend in BACKENDS:
        for family_label, operations in FAMILIES:
            scores: list[tuple[float, str]] = []
            for library in libraries:
                ranks = [
                    benchmark_ranks[(library, operation, backend)]
                    for operation in operations
                    if (library, operation, backend) in benchmark_ranks
                ]
                if ranks:
                    scores.append((sum(ranks) / len(ranks), library))
            for library, rank in _competition_ranks(scores).items():
                family_ranks[(library, family_label, backend)] = rank

    lines = [
        r"% Requires \usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Package ranks by benchmark family and backend, ordered from lowest to highest overall performance. Family and overall ranks average the completed constituent benchmarks;~-- denotes that no benchmark in the family was completed.}",
        r"\label{tab:benchmark-ranks}",
    ]
    for backend_index, backend in enumerate(BACKENDS):
        if backend_index:
            lines.append(r"\vspace{0.75em}")
        lines.extend([
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            rf"\multicolumn{{6}}{{c}}{{\textbf{{{backend.upper()}}}}} \\",
            r"\addlinespace",
            r"\textbf{Package} & \textbf{Signatures} & \textbf{Log signatures} & \textbf{Branched} & \textbf{Kernels} & \textbf{Overall} \\",
            r"\midrule",
        ])
        backend_libraries = sorted(
            (
                library
                for library in libraries
                if (library, backend) in overall_ranks
            ),
            key=lambda library: (
                -overall_ranks[(library, backend)],
                coverage[(library, backend)],
                library.casefold(),
            ),
        )
        best_rule_added = False
        for library in backend_libraries:
            cells = [_latex_escape(library)]
            for family_label, _ in FAMILIES:
                rank = family_ranks.get((library, family_label, backend))
                cells.append(str(rank) if rank is not None else "--")
            overall_rank = overall_ranks[(library, backend)]
            cells.append(str(overall_rank))
            if overall_rank == 1:
                if not best_rule_added:
                    lines.append(r"\hline")
                    best_rule_added = True
                cells = [rf"\textbf{{{cell}}}" for cell in cells]
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
        ])
    lines.append(r"\end{table}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=REPO_ROOT / "benchmark_rank_table.csv",
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmark_rank_table.tex",
    )
    args = parser.parse_args()

    ranks = calculate_ranks(args.runs_root)
    write_csv(args.csv_output, *ranks)
    write_latex(args.tex_output, *ranks)
    print(f"CSV rank table saved to: {args.csv_output}")
    print(f"LaTeX rank table saved to: {args.tex_output}")


if __name__ == "__main__":
    main()
