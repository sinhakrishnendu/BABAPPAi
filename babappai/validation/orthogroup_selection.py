"""Deterministic orthogroup QC and top-100 stratified selection workflow."""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from babappai.validation.orthogroup_qc import (
    apply_hard_filters,
    compute_orthogroup_metrics,
    list_orthogroup_alignments,
)


def _quantile_bin(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def _write_tsv(path: Path, rows: list[dict[str, Any]], fallback_fields: list[str]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) or fallback_fields
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stratified_pick(candidates: list[dict[str, Any]], target_n: int) -> list[dict[str, Any]]:
    if not candidates:
        return []

    div_values = np.array([float(c["mean_divergence"]) for c in candidates], dtype=float)
    len_values = np.array([float(c["median_length_nt"]) for c in candidates], dtype=float)
    occ_values = np.array([float(c["occupancy"]) for c in candidates], dtype=float)

    div_edges = np.quantile(div_values, [1 / 3, 2 / 3]).tolist()
    len_edges = np.quantile(len_values, [1 / 3, 2 / 3]).tolist()
    occ_edges = np.quantile(occ_values, [1 / 3, 2 / 3]).tolist()

    grouped: dict[str, deque[dict[str, Any]]] = {}
    for row in candidates:
        d_bin = _quantile_bin(float(row["mean_divergence"]), div_edges)
        l_bin = _quantile_bin(float(row["median_length_nt"]), len_edges)
        o_bin = _quantile_bin(float(row["occupancy"]), occ_edges)
        stratum = f"d{d_bin}_l{l_bin}_o{o_bin}"
        row["stratum"] = stratum

    for stratum in sorted({row["stratum"] for row in candidates}):
        items = [row for row in candidates if row["stratum"] == stratum]
        items.sort(key=lambda r: (-float(r["qc_score"]), r["orthogroup_id"]))
        grouped[stratum] = deque(items)

    selected: list[dict[str, Any]] = []
    active = sorted(grouped.keys())
    while active and len(selected) < target_n:
        next_active = []
        for stratum in active:
            queue = grouped[stratum]
            if queue:
                selected.append(queue.popleft())
            if queue:
                next_active.append(stratum)
            if len(selected) >= target_n:
                break
        active = next_active

    return selected


def select_orthogroups(
    *,
    input_dir: str,
    outdir: str,
    target_n: int = 100,
    min_taxa: int = 8,
    occupancy_threshold: float = 0.7,
    min_length_nt: int = 300,
    max_missingness: float = 0.2,
    enforce_one_to_one: bool = True,
    require_no_internal_stops: bool = True,
) -> Dict[str, Any]:
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    alignments = list_orthogroup_alignments(input_dir)
    if not alignments:
        raise FileNotFoundError(f"No alignment files found in: {input_dir}")

    max_taxa = 0
    raw_metrics = []
    for path in alignments:
        metrics = compute_orthogroup_metrics(path, dataset_max_taxa=1)
        raw_metrics.append(metrics)
        max_taxa = max(max_taxa, int(metrics["n_sequences"]))

    evaluated = []
    for path in alignments:
        metrics = compute_orthogroup_metrics(path, dataset_max_taxa=max_taxa)
        filtered = apply_hard_filters(
            metrics=metrics,
            min_taxa=min_taxa,
            occupancy_threshold=occupancy_threshold,
            min_length_nt=min_length_nt,
            max_missingness=max_missingness,
            enforce_one_to_one=enforce_one_to_one,
            require_no_internal_stops=require_no_internal_stops,
        )
        evaluated.append(filtered)

    accepted = [row for row in evaluated if row["hard_filter_pass"]]
    rejected = [row for row in evaluated if not row["hard_filter_pass"]]

    selected = _stratified_pick(accepted, target_n=target_n)
    selected_ids = {row["orthogroup_id"] for row in selected}

    not_selected = [
        {**row, "rejection_reasons": "not_in_stratified_top_set"}
        for row in accepted
        if row["orthogroup_id"] not in selected_ids
    ]
    rejected_all = rejected + not_selected
    rejected_all.sort(key=lambda row: row["orthogroup_id"])

    selected_rows = []
    for row in selected:
        selected_rows.append(
            {
                "orthogroup_id": row["orthogroup_id"],
                "alignment_path": row["alignment_path"],
                "tree_path": row.get("tree_path", ""),
                "qc_score": row["qc_score"],
                "occupancy": row["occupancy"],
                "median_length_nt": row["median_length_nt"],
                "mean_divergence": row["mean_divergence"],
                "stratum": row["stratum"],
            }
        )
    selected_rows.sort(key=lambda row: row["orthogroup_id"])

    _write_tsv(
        out / "selected_100_orthogroups.tsv",
        selected_rows,
        fallback_fields=[
            "orthogroup_id",
            "alignment_path",
            "tree_path",
            "qc_score",
            "occupancy",
            "median_length_nt",
            "mean_divergence",
            "stratum",
        ],
    )
    _write_tsv(
        out / "rejected_orthogroups.tsv",
        rejected_all,
        fallback_fields=["orthogroup_id", "alignment_path", "rejection_reasons"],
    )
    _write_tsv(
        out / "orthogroup_qc_metrics.tsv",
        evaluated,
        fallback_fields=[
            "orthogroup_id",
            "alignment_path",
            "hard_filter_pass",
            "rejection_reasons",
            "qc_score",
        ],
    )

    stratum_counts = defaultdict(int)
    for row in selected:
        stratum_counts[row["stratum"]] += 1

    report_lines = [
        "BABAPPAi Orthogroup Selection Report",
        "====================================",
        "",
        f"Total candidates: {len(evaluated)}",
        f"Hard-filter pass: {len(accepted)}",
        f"Selected (target={target_n}): {len(selected)}",
        "",
        "Selection policy:",
        "- Deterministic QC ranking with hard filters.",
        "- Anti-cherry-picking stratified selection across divergence, length, and occupancy bins.",
        "- Balanced round-robin draw across non-empty strata.",
        "",
        "Hard filters:",
        f"- min_taxa >= {min_taxa}",
        f"- occupancy >= {occupancy_threshold}",
        f"- median_length_nt >= {min_length_nt}",
        f"- missingness <= {max_missingness}",
        f"- enforce_one_to_one={enforce_one_to_one}",
        f"- require_no_internal_stops={require_no_internal_stops}",
        "",
        "Selected stratum counts:",
    ]
    for key in sorted(stratum_counts):
        report_lines.append(f"- {key}: {stratum_counts[key]}")
    (out / "orthogroup_selection_report.txt").write_text("\n".join(report_lines) + "\n")

    metadata = {
        "input_dir": str(Path(input_dir).resolve()),
        "target_n": target_n,
        "filters": {
            "min_taxa": min_taxa,
            "occupancy_threshold": occupancy_threshold,
            "min_length_nt": min_length_nt,
            "max_missingness": max_missingness,
            "enforce_one_to_one": enforce_one_to_one,
            "require_no_internal_stops": require_no_internal_stops,
        },
        "counts": {
            "total_candidates": len(evaluated),
            "hard_filter_pass": len(accepted),
            "selected": len(selected),
            "rejected": len(rejected_all),
        },
        "stratification": {
            "axes": ["mean_divergence", "median_length_nt", "occupancy"],
            "selected_counts": dict(sorted(stratum_counts.items())),
        },
        "provenance_note": (
            "BABAPPAi is the renamed continuation of the BABAPPAΩ codebase."
        ),
    }
    (out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return metadata
