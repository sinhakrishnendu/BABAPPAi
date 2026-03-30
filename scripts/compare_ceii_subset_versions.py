#!/usr/bin/env python3
"""Compare cEII outputs across two empirical panel summaries for a fixed gene subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _float(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _read_summary(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open() as fh:
        rows = [dict(r) for r in csv.DictReader(fh, delimiter="\t") if r]
    return {str(r.get("gene", "")).strip(): r for r in rows if str(r.get("gene", "")).strip()}


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _fmt(v: Any, nd: int = 4) -> str:
    fv = _float(v)
    return "n/a" if fv is None else f"{fv:.{nd}f}"


def _interpret(
    applicability: str,
    eii_01: float | None,
    q_emp: float | None,
    ceii_gene_new: float | None,
) -> str:
    if applicability != "in_domain":
        return "abstained_or_boundary"
    if ceii_gene_new is None:
        return "unexpected_null_in_domain"
    if eii_01 is not None and q_emp is not None:
        if eii_01 >= 0.8 and q_emp <= 0.1 and ceii_gene_new >= 0.5:
            return "coherent_strong_evidence"
        if eii_01 <= 0.4 and q_emp >= 0.8 and ceii_gene_new <= 0.5:
            return "coherent_weak_evidence"
        if eii_01 >= 0.8 and q_emp <= 0.1 and ceii_gene_new < 0.5:
            return "suppressed_against_strong_evidence"
        if eii_01 <= 0.4 and q_emp >= 0.8 and ceii_gene_new > 0.5:
            return "inflated_against_weak_evidence"
    return "mixed_or_intermediate"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-summary", required=True, help="panel_summary.tsv from ceii_v3.1 run")
    p.add_argument("--new-summary", required=True, help="panel_summary.tsv from ceii_v3.2 run")
    p.add_argument("--genes", required=True, help="comma-separated ordered subset")
    p.add_argument("--out-tsv", required=True)
    p.add_argument("--out-md", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    old_summary = Path(args.old_summary).expanduser().resolve()
    new_summary = Path(args.new_summary).expanduser().resolve()
    genes = [g.strip() for g in str(args.genes).split(",") if g.strip()]
    if not genes:
        raise ValueError("No genes provided")

    old = _read_summary(old_summary)
    new = _read_summary(new_summary)

    rows: List[Dict[str, Any]] = []
    for gene in genes:
        o = old.get(gene, {})
        n = new.get(gene, {})
        eii_01 = _float(n.get("eii_01_raw") or o.get("eii_01_raw"))
        q_emp = _float(n.get("q_emp") or o.get("q_emp"))
        old_gene = _float(o.get("ceii_gene"))
        old_site = _float(o.get("ceii_site"))
        new_gene = _float(n.get("ceii_gene"))
        new_site = _float(n.get("ceii_site"))
        applicability = str(n.get("applicability_status") or o.get("applicability_status") or "unknown")

        rows.append(
            {
                "gene": gene,
                "applicability_status": applicability,
                "eii_01_raw": eii_01,
                "q_emp": q_emp,
                "old_ceii_gene": old_gene,
                "new_ceii_gene": new_gene,
                "old_ceii_site": old_site,
                "new_ceii_site": new_site,
                "old_ceii_gene_class": o.get("ceii_gene_class", ""),
                "new_ceii_gene_class": n.get("ceii_gene_class", ""),
                "old_calibration_version": o.get("calibration_version", ""),
                "new_calibration_version": n.get("calibration_version", ""),
                "interpretation": _interpret(applicability, eii_01, q_emp, new_gene),
            }
        )

    out_tsv = Path(args.out_tsv).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    _write_tsv(out_tsv, rows)

    lines: List[str] = []
    lines.append("# cEII v3.2 Subset Comparison")
    lines.append("")
    lines.append(f"- old_summary: `{old_summary}`")
    lines.append(f"- new_summary: `{new_summary}`")
    lines.append("")
    for r in rows:
        lines.append(f"## {r['gene']}")
        lines.append(
            "- evidence: "
            f"eii_01_raw={_fmt(r.get('eii_01_raw'))}, "
            f"q_emp={_fmt(r.get('q_emp'))}, "
            f"applicability={r.get('applicability_status')}"
        )
        lines.append(
            "- gene cEII: "
            f"old={_fmt(r.get('old_ceii_gene'))}, "
            f"new={_fmt(r.get('new_ceii_gene'))}"
        )
        lines.append(
            "- site cEII: "
            f"old={_fmt(r.get('old_ceii_site'))}, "
            f"new={_fmt(r.get('new_ceii_site'))}"
        )
        lines.append(f"- interpretation: {r.get('interpretation')}")
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).rstrip() + "\n")

    print(json.dumps({"out_tsv": str(out_tsv), "out_md": str(out_md), "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
