#!/usr/bin/env python3
"""Compare old/new cEII behavior on empirical BABAPPAi result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.calibration.ceii import apply_ceii_calibration, load_calibration_asset  # noqa: E402


EXPECTED_CONTROL_CLASS: Dict[str, str] = {
    "ago2": "expected_positive_control",
    "dcr-2": "expected_positive_control",
    "r2d2": "expected_positive_control",
    "ago1": "conservative_rnai_control",
    "dcr-1": "conservative_rnai_control",
    "act5c": "housekeeping_control",
    "alphatub84d": "housekeeping_control",
    "eef1alpha": "housekeeping_control",
    "gapdh1": "housekeeping_control",
    "rpl32": "housekeeping_control",
    "tep1": "expected_positive_control",
    "spclip1": "expected_positive_control",
    "clipa8": "expected_positive_control",
    "clipb14": "expected_positive_control",
    "clipb15": "expected_positive_control",
}


def _n_branches_from_payload(payload: Mapping[str, Any]) -> Optional[int]:
    branch_results = payload.get("branch_results")
    if isinstance(branch_results, list) and branch_results:
        return int(len(branch_results))
    cal_group = str(payload.get("gene_summary", {}).get("calibration_group", ""))
    if "_K_" in cal_group:
        try:
            return int(cal_group.split("_K_")[-1])
        except Exception:
            return None
    return None


def _float(v: Any) -> Optional[float]:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _evidence_label(eii_01_raw: Optional[float], q_emp: Optional[float]) -> str:
    if eii_01_raw is None or q_emp is None:
        return "unknown"
    if eii_01_raw >= 0.80 and q_emp <= 0.10:
        return "strong"
    if eii_01_raw <= 0.40 and q_emp >= 0.80:
        return "weak"
    return "mixed"


def _coherence_note(
    *,
    evidence_label: str,
    old_ceii: Optional[float],
    new_ceii: Optional[float],
    new_status: str,
) -> str:
    if evidence_label == "weak" and old_ceii is not None and old_ceii >= 0.80:
        return "old_cEII_inflated_against_weak_raw_evidence"
    if evidence_label == "strong" and old_ceii is not None and old_ceii <= 0.20:
        return "old_cEII_suppressed_against_strong_raw_evidence"
    if new_status != "in_domain":
        return "new_cEII_abstained_due_applicability"
    if new_ceii is None:
        return "new_cEII_abstained"
    if evidence_label == "weak" and new_ceii <= 0.50:
        return "new_cEII_consistent_with_weak_evidence"
    if evidence_label == "strong" and new_ceii >= 0.50:
        return "new_cEII_consistent_with_strong_evidence"
    return "mixed_or_intermediate"


def _write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, help="Directory with *.results.json from empirical runs.")
    p.add_argument("--old-asset", required=True, help="Old calibration asset path (for example ceii_v2).")
    p.add_argument("--new-asset", required=True, help="New calibration asset path (for example ceii_v3.1).")
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--audit-genes",
        default="ago1,ago2,dcr-1,dcr-2,eef1alpha,act5c,gapdh1,r2d2",
        help="Comma-separated subset used for contradiction audit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    old_asset = load_calibration_asset(args.old_asset)
    new_asset = load_calibration_asset(args.new_asset)
    old_version = str(old_asset.get("calibration_version", "old"))
    new_version = str(new_asset.get("calibration_version", "new"))

    audit_genes = {x.strip() for x in str(args.audit_genes).split(",") if x.strip()}
    rows: List[Dict[str, Any]] = []

    for path in sorted(results_dir.glob("*.results.json")):
        gene = path.stem.replace(".results", "")
        payload = json.loads(path.read_text())
        gene_summary = dict(payload.get("gene_summary", {}))
        if not gene_summary:
            continue

        n_taxa = int(payload.get("input", {}).get("n_sequences", 0) or 0)
        gene_len_nt = int(payload.get("input", {}).get("alignment_length", 0) or 0)
        n_branches = _n_branches_from_payload(payload)

        eii_z = _float(gene_summary.get("eii_z_raw"))
        eii_01 = _float(gene_summary.get("eii_01_raw"))
        q_emp = _float(gene_summary.get("q_emp"))
        dispersion_ratio = _float(gene_summary.get("dispersion_ratio"))
        sigma0_final = _float(gene_summary.get("sigma0_final"))

        old_cal = apply_ceii_calibration(
            eii_z_raw=float(eii_z or 0.0),
            n_taxa=n_taxa,
            gene_length_nt=gene_len_nt,
            n_branches=n_branches,
            q_emp=q_emp,
            dispersion_ratio=dispersion_ratio,
            sigma0_final=sigma0_final,
            asset=old_asset,
        )
        new_cal = apply_ceii_calibration(
            eii_z_raw=float(eii_z or 0.0),
            n_taxa=n_taxa,
            gene_length_nt=gene_len_nt,
            n_branches=n_branches,
            q_emp=q_emp,
            dispersion_ratio=dispersion_ratio,
            sigma0_final=sigma0_final,
            asset=new_asset,
        )

        old_ceii = _float(old_cal.get("ceii_gene"))
        new_ceii = _float(new_cal.get("ceii_gene"))
        new_status = str(new_cal.get("applicability_status", "unknown"))
        evidence_label = _evidence_label(eii_01, q_emp)
        note = _coherence_note(
            evidence_label=evidence_label,
            old_ceii=old_ceii,
            new_ceii=new_ceii,
            new_status=new_status,
        )

        rows.append(
            {
                "gene": gene,
                "expected_control_class": EXPECTED_CONTROL_CLASS.get(gene, "unknown_control_class"),
                "applicability_status": str(new_cal.get("applicability_status", "")),
                "applicability_score": new_cal.get("applicability_score"),
                "calibration_unavailable_reason": str(new_cal.get("calibration_unavailable_reason", "")),
                "eii_01_raw": eii_01,
                "q_emp": q_emp,
                f"old_ceii_gene_{old_version}": old_ceii,
                f"new_ceii_gene_{new_version}": new_ceii,
                "old_ceii_gene_class": old_cal.get("ceii_gene_class"),
                "new_ceii_gene_class": new_cal.get("ceii_gene_class"),
                "evidence_label": evidence_label,
                "interpretation": note,
            }
        )

    rows.sort(key=lambda r: str(r["gene"]))
    _write_table(outdir / "ceii_v2_vs_v3_1_empirical_comparison.tsv", rows)

    audit_rows = [r for r in rows if str(r["gene"]) in audit_genes]
    _write_table(outdir / "ceii_contradiction_audit_genes.tsv", audit_rows)

    summary = {
        "old_calibration_version": old_version,
        "new_calibration_version": new_version,
        "n_genes_total": int(len(rows)),
        "n_audit_genes": int(len(audit_rows)),
        "comparison_tsv": str(outdir / "ceii_v2_vs_v3_1_empirical_comparison.tsv"),
        "audit_tsv": str(outdir / "ceii_contradiction_audit_genes.tsv"),
    }
    (outdir / "ceii_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

