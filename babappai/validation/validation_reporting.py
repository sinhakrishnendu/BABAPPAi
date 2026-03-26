"""Unified validation report generation for empirical + synthetic workflows."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from babappai.metadata import MODEL_COMPATIBILITY_NOTE
from babappai.validation.validation_plots import write_regime_bar_svg


def _find_optional(path_root: Path, filename: str) -> Path | None:
    matches = sorted(path_root.rglob(filename))
    return matches[0] if matches else None


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader if row]


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_validation_report(*, input_dir: str, outdir: str) -> Dict[str, Any]:
    inp = Path(input_dir).expanduser().resolve()
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    empirical_tsv = _find_optional(inp, "empirical_summary.tsv")
    synthetic_tsv = _find_optional(inp, "synthetic_replicates.tsv")
    selection_meta = _find_optional(inp, "selection_metadata.json")

    empirical_rows = _read_tsv(empirical_tsv) if empirical_tsv else []
    synthetic_rows = _read_tsv(synthetic_tsv) if synthetic_tsv else []

    master_rows: List[Dict[str, Any]] = []
    for row in empirical_rows:
        master_rows.append(
            {
                "dataset_type": "empirical",
                "id": row.get("orthogroup_id", ""),
                "EII_z": row.get("EII_z", ""),
                "EII_01": row.get("EII_01", ""),
                "p_emp": row.get("p_emp", ""),
                "q_emp": row.get("q_emp", ""),
                "significant_bool": row.get("significant_bool", ""),
                "significance_label": row.get("significance_label", ""),
                "identifiable_bool": row.get("identifiable_bool", ""),
                "identifiability_extent": row.get("identifiability_extent", ""),
                "source_table": str(empirical_tsv) if empirical_tsv else "",
            }
        )
    for row in synthetic_rows:
        master_rows.append(
            {
                "dataset_type": "synthetic",
                "id": row.get("replicate_id", ""),
                "EII_z": row.get("EII_z", ""),
                "EII_01": row.get("EII_01", ""),
                "p_emp": row.get("p_emp", ""),
                "q_emp": row.get("q_emp", ""),
                "significant_bool": row.get("significant_bool", ""),
                "significance_label": row.get("significance_label", ""),
                "identifiable_bool": row.get("identifiable_bool", ""),
                "identifiability_extent": row.get("identifiability_extent", ""),
                "source_table": str(synthetic_tsv) if synthetic_tsv else "",
            }
        )

    (out / "validation_master_summary.json").write_text(
        json.dumps(master_rows, indent=2) + "\n"
    )
    _write_tsv(out / "validation_master_summary.tsv", master_rows)

    empirical_counts = Counter(row.get("identifiability_extent", "") for row in empirical_rows)
    synthetic_counts = Counter(row.get("identifiability_extent", "") for row in synthetic_rows)
    empirical_sig_counts = Counter(row.get("significance_label", "") for row in empirical_rows)
    synthetic_sig_counts = Counter(row.get("significance_label", "") for row in synthetic_rows)

    fig_dir = out / "publication_ready_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_regime_bar_svg(
        dict(sorted(empirical_counts.items())),
        fig_dir / "empirical_regime_counts.svg",
        "Empirical Regime Counts",
    )
    write_regime_bar_svg(
        dict(sorted(synthetic_counts.items())),
        fig_dir / "synthetic_regime_counts.svg",
        "Synthetic Regime Counts",
    )

    supp_dir = out / "supplementary_tables"
    supp_dir.mkdir(parents=True, exist_ok=True)
    if empirical_tsv:
        shutil.copy2(empirical_tsv, supp_dir / "empirical_summary.tsv")
    if synthetic_tsv:
        shutil.copy2(synthetic_tsv, supp_dir / "synthetic_replicates.tsv")
    if selection_meta:
        shutil.copy2(selection_meta, supp_dir / "selection_metadata.json")

    selection_summary = {}
    if selection_meta and selection_meta.exists():
        selection_summary = json.loads(selection_meta.read_text())

    report_lines = [
        "# BABAPPAi Validation Report",
        "",
        "## 1) Overview",
        f"- Empirical rows: {len(empirical_rows)}",
        f"- Synthetic rows: {len(synthetic_rows)}",
        "",
        "## 2) Provenance and naming compatibility notes",
        f"- {MODEL_COMPATIBILITY_NOTE}",
        "",
        "## 3) Orthogroup selection criteria",
        "- Deterministic rule-based selection with hard filters.",
        "- Anti-cherry-picking stratified selection across divergence, length, and occupancy bins.",
        f"- Selection metadata path: {selection_meta if selection_meta else 'not found'}",
        "",
        "## 4) Empirical Anopheles validation results",
        f"- Regime counts: {dict(sorted(empirical_counts.items()))}",
        f"- Significance counts: {dict(sorted(empirical_sig_counts.items()))}",
        "",
        "## 5) Synthetic benchmark design",
        "- Simulator-driven parameter grid with simulate-and-bucket tracking.",
        f"- Synthetic significance counts: {dict(sorted(synthetic_sig_counts.items()))}",
        "",
        "## 6) Role of neutral calibration generator",
        "- External neutral generator can be run and logged via adapter metadata.",
        "",
        "## 7) Calibration behavior",
        "- Compare calibrated vs non-calibrated runs using empirical robustness metrics and synthetic summary.",
        "",
        "## 8) Stability/perturbation analyses",
        "- Repeatability, taxon subsampling sensitivity, and mild perturbation sensitivity are exported per orthogroup.",
        "",
        "## 9) Limitations",
        "- Legacy frozen model currently carries BABAPPAΩ provenance; BABAPPAi-specific weights are pending.",
        "",
        "## 10) Recommended manuscript wording",
        "- \"BABAPPAi is the renamed continuation of the BABAPPAΩ codebase.\"",
        "- \"EII is a recoverability diagnostic; inferential support is based on empirical p/q values under matched neutral calibration.\"",
        "- \"Significant q-values indicate excess dispersion relative to matched neutral simulation, not proof of adaptive substitution.\"",
        "",
        "## 11) Software citation and legacy asset citation notes",
        "- Cite BABAPPAi software version and explicitly reference legacy model DOI where applicable.",
        "",
    ]
    (out / "validation_report.md").write_text("\n".join(report_lines))

    metadata = {
        "input_dir": str(inp),
        "report_dir": str(out),
        "n_master_rows": len(master_rows),
        "empirical_regime_counts": dict(sorted(empirical_counts.items())),
        "synthetic_regime_counts": dict(sorted(synthetic_counts.items())),
    }
    return metadata
