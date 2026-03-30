#!/usr/bin/env python3
"""Generate per-gene cEII internal trace tables for empirical result JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.calibration import load_calibration_asset, trace_ceii_calibration  # noqa: E402


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _read_summary_rows(paths: Iterable[Path]) -> Dict[str, Path]:
    by_gene: Dict[str, Path] = {}
    for path in paths:
        with path.open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                gene = str(row.get("gene", "")).strip()
                if not gene:
                    continue
                rj = str(row.get("results_json", "")).strip()
                if not rj:
                    continue
                by_gene[gene] = Path(rj).expanduser().resolve()
    return by_gene


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for k in keys:
                v = row.get(k)
                if isinstance(v, (dict, list)):
                    out[k] = json.dumps(v, sort_keys=True)
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            writer.writerow(out)


def _fmt(v: Any, digits: int = 4) -> str:
    fv = _float(v)
    if fv is None:
        return "n/a"
    return f"{fv:.{digits}f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-tsv", action="append", required=True, help="panel_summary.tsv (repeatable)")
    p.add_argument("--ceii-asset", required=True)
    p.add_argument("--genes", required=True, help="comma-separated gene list in desired output order")
    p.add_argument("--out-tsv", required=True)
    p.add_argument("--out-md", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()

    summary_paths = [Path(x).expanduser().resolve() for x in args.summary_tsv]
    for p in summary_paths:
        if not p.exists():
            raise FileNotFoundError(f"Summary TSV not found: {p}")

    asset_path = Path(args.ceii_asset).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"cEII asset not found: {asset_path}")
    asset = load_calibration_asset(asset_path)

    genes = [g.strip() for g in str(args.genes).split(",") if g.strip()]
    if not genes:
        raise ValueError("No genes provided.")

    by_gene = _read_summary_rows(summary_paths)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for gene in genes:
        result_path = by_gene.get(gene)
        if result_path is None or not result_path.exists():
            missing.append(gene)
            rows.append({"gene": gene, "status": "missing_result_json"})
            continue

        payload = json.loads(result_path.read_text())
        gs = dict(payload.get("gene_summary", {}))

        n_taxa = int(payload.get("input", {}).get("n_sequences") or payload.get("n_sequences") or 0)
        gene_len = int(payload.get("input", {}).get("alignment_length") or payload.get("num_sites") or 0)
        n_branches = len(payload.get("branch_results", [])) if isinstance(payload.get("branch_results"), list) else None

        eii_z_raw = _float(gs.get("eii_z_raw", gs.get("EII_z"))) or 0.0
        q_emp = _float(gs.get("q_emp"))
        dispersion_ratio = _float(gs.get("dispersion_ratio"))
        sigma0_final = _float(gs.get("sigma0_final"))

        trace = trace_ceii_calibration(
            eii_z_raw=float(eii_z_raw),
            n_taxa=int(n_taxa),
            gene_length_nt=int(gene_len),
            n_branches=n_branches,
            q_emp=q_emp,
            dispersion_ratio=dispersion_ratio,
            sigma0_final=sigma0_final,
            asset=asset,
        )

        app = trace.get("applicability", {})
        gene_trace = trace.get("gene_trace", {})
        site_trace = trace.get("site_trace", {})
        final = trace.get("final", {})
        evidence_vals = (trace.get("evidence", {}) or {}).get("values", {})

        ceii_gene = _float(final.get("ceii_gene"))
        ceii_site = _float(final.get("ceii_site"))
        eii_01 = _float(gs.get("eii_01_raw", gs.get("EII_01")))
        significant = bool(gs.get("significant_bool", False))
        excess_gate_runtime_status = "pass" if significant else "fail"

        rows.append(
            {
                "gene": gene,
                "status": "ok",
                "results_json": str(result_path),
                "calibration_version": str(final.get("calibration_version", gs.get("calibration_version", ""))),
                "ceii_asset_path": str(asset_path),
                "n_taxa": n_taxa,
                "gene_length_nt": gene_len,
                "n_branches": n_branches,
                "eii_z_raw": eii_z_raw,
                "eii_01_raw": eii_01,
                "q_emp": q_emp,
                "neglog10_q_emp": evidence_vals.get("neglog10_q_emp"),
                "dispersion_ratio": dispersion_ratio,
                "dispersion_ratio_clipped": evidence_vals.get("dispersion_ratio_clipped"),
                "sigma0_final": sigma0_final,
                "applicability_support_features": app.get("support_features"),
                "applicability_score": app.get("applicability_score"),
                "applicability_status": app.get("applicability_status"),
                "within_applicability_envelope": app.get("within_applicability_envelope"),
                "distance_to_supported_domain": app.get("distance_to_supported_domain"),
                "nearest_supported_regime": app.get("nearest_supported_regime"),
                "calibration_unavailable_reason": app.get("calibration_unavailable_reason"),
                "should_calibrate": app.get("should_calibrate"),
                "target_definition_profile": (trace.get("evidence", {}) or {}).get("target_definition_profile"),
                "target_definitions": (trace.get("evidence", {}) or {}).get("target_definitions"),
                "evidence_features_used": evidence_vals,
                "gene_score_S": gene_trace.get("score"),
                "gene_isotonic_input": gene_trace.get("isotonic_input"),
                "gene_isotonic_output": gene_trace.get("isotonic_output"),
                "gene_logistic_output": gene_trace.get("logistic_output"),
                "gene_blended_output": gene_trace.get("blended_output"),
                "gene_feature_values": gene_trace.get("feature_values"),
                "site_score_S": site_trace.get("score"),
                "site_isotonic_input": site_trace.get("isotonic_input"),
                "site_isotonic_output": site_trace.get("isotonic_output"),
                "site_logistic_output": site_trace.get("logistic_output"),
                "site_blended_output": site_trace.get("blended_output"),
                "site_feature_values": site_trace.get("feature_values"),
                "excess_gate_runtime_status": excess_gate_runtime_status,
                "final_ceii_gene": ceii_gene,
                "final_ceii_site": ceii_site,
                "final_ceii_gene_class": final.get("ceii_gene_class"),
                "final_ceii_site_class": final.get("ceii_site_class"),
            }
        )

    out_tsv = Path(args.out_tsv).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    _write_tsv(out_tsv, rows)

    in_domain = [r for r in rows if r.get("status") == "ok" and str(r.get("applicability_status", "")) == "in_domain"]
    unique_scores = sorted({round(float(r["gene_score_S"]), 10) for r in in_domain if _float(r.get("gene_score_S")) is not None})
    score_collapse = len(unique_scores) <= 1 and len(in_domain) >= 2

    md_lines: List[str] = []
    md_lines.append("# cEII Internal Trace (Targeted Subset)")
    md_lines.append("")
    md_lines.append(f"- calibration_asset: `{asset_path}`")
    md_lines.append(f"- calibration_version: `{asset.get('calibration_version', 'unknown')}`")
    md_lines.append(f"- genes_requested: `{', '.join(genes)}`")
    if missing:
        md_lines.append(f"- missing_results: `{', '.join(missing)}`")
    md_lines.append(f"- in_domain_gene_count: `{len(in_domain)}`")
    md_lines.append(f"- in_domain_unique_gene_scores: `{len(unique_scores)}`")
    if score_collapse:
        md_lines.append("- diagnosis_flag: in-domain composite evidence score collapse (constant score -> isotonic floor)")
    md_lines.append("")

    for row in rows:
        gene = row.get("gene", "unknown")
        md_lines.append(f"## {gene}")
        if row.get("status") != "ok":
            md_lines.append("- status: missing_result_json")
            md_lines.append("")
            continue
        md_lines.append(
            "- raw evidence: "
            f"eii_01_raw={_fmt(row.get('eii_01_raw'))}, "
            f"q_emp={_fmt(row.get('q_emp'))}, "
            f"dispersion_ratio={_fmt(row.get('dispersion_ratio'))}"
        )
        md_lines.append(
            "- applicability: "
            f"status={row.get('applicability_status')}, "
            f"score={_fmt(row.get('applicability_score'))}, "
            f"reason={row.get('calibration_unavailable_reason') or 'none'}"
        )
        md_lines.append(
            "- stage-B internals: "
            f"S_gene={_fmt(row.get('gene_score_S'), 6)}, "
            f"iso_gene={_fmt(row.get('gene_isotonic_output'), 6)}, "
            f"blend_gene={_fmt(row.get('gene_blended_output'), 6)}"
        )
        md_lines.append(
            "- final output: "
            f"ceii_gene={_fmt(row.get('final_ceii_gene'))}, "
            f"ceii_site={_fmt(row.get('final_ceii_site'))}, "
            f"gene_class={row.get('final_ceii_gene_class')}"
        )
        if gene in {"ago1", "dcr-2"} and score_collapse:
            md_lines.append(
                "- diagnosis: strong raw evidence mapped to ceii=0 because in-domain score collapsed to a constant and isotonic output stayed at floor."
            )
        if gene == "eef1alpha":
            md_lines.append(
                "- diagnosis: low raw evidence and non-significant q_emp are coherent with low calibrated output."
            )
        md_lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines).rstrip() + "\n")

    print(json.dumps({
        "out_tsv": str(out_tsv),
        "out_md": str(out_md),
        "n_rows": len(rows),
        "n_missing": len(missing),
        "in_domain_unique_gene_scores": len(unique_scores),
        "score_collapse": bool(score_collapse),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
