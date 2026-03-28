"""Human-readable interpretation report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def _load_results(result_input) -> Dict[str, object]:
    if isinstance(result_input, (str, Path)):
        return json.loads(Path(result_input).read_text())
    if isinstance(result_input, Mapping):
        return dict(result_input)
    raise TypeError("result_input must be a mapping or a JSON file path")


def _regime_explanation(regime: str) -> str:
    mapping = {
        "not_identifiable": (
            "Current data do not support reliable episodic branch-site "
            "identifiability at the gene level within the calibrated simulation domain."
        ),
        "weak_or_ambiguous": (
            "Calibrated probability indicates ambiguous recoverability; interpret as "
            "exploratory and seek additional sequence depth or broader taxon sampling."
        ),
        "identifiable": (
            "The gene-level pattern is measurably identifiable under the empirical "
            "calibration regime."
        ),
        "strongly_identifiable": (
            "The gene-level pattern is strongly identifiable under the empirical "
            "calibration regime."
        ),
        "calibration_unavailable": (
            "Calibrated identifiability probability is abstained because the sample "
            "is outside or near the boundary of validated calibration support."
        ),
    }
    return mapping.get(regime, "Interpretation regime unavailable.")


def _sorted_branch_rows(branch_results: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    return sorted(
        branch_results,
        key=lambda row: float(row.get("background_score", 0.0)),
        reverse=True,
    )


def _sorted_site_rows(site_results: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    return sorted(
        site_results,
        key=lambda row: float(row.get("site_score", 0.0)),
        reverse=True,
    )


def render_interpretation(result_input, *, top_branches: int = 10, top_sites: int = 20) -> str:
    results = _load_results(result_input)

    gene = results.get("gene_summary") or results.get("gene_level_identifiability", {})
    eii_z_raw = gene.get("eii_z_raw", gene.get("EII_z"))
    eii_01_raw = gene.get("eii_01_raw", gene.get("EII_01"))
    ceii_gene = gene.get("ceii_gene")
    ceii_site = gene.get("ceii_site")
    ceii_gene_class = gene.get("ceii_gene_class", gene.get("identifiability_extent"))
    ceii_site_class = gene.get("ceii_site_class", "unavailable")
    ceii_ci = gene.get("ceii_ci", {})
    calibration_version = gene.get("calibration_version", "unknown")
    applicability = gene.get("domain_shift_or_applicability", "unknown")
    applicability_score = gene.get("applicability_score")
    applicability_status = gene.get("applicability_status", applicability)
    within_envelope = gene.get("within_applicability_envelope")
    calib_unavailable_reason = gene.get("calibration_unavailable_reason")
    nearest_supported_regime = gene.get("nearest_supported_regime")
    distance_to_supported_domain = gene.get("distance_to_supported_domain")
    p_emp = gene.get("p_emp")
    q_emp = gene.get("q_emp")
    alpha_used = gene.get("alpha_used")
    significant_bool = gene.get("significant_bool")
    significance_label = gene.get("significance_label")
    identifiable = gene.get("ceii_gene_identifiable_bool", gene.get("identifiable_bool"))
    extent = ceii_gene_class

    branch_results = _sorted_branch_rows(results.get("branch_results", []))
    site_results = _sorted_site_rows(results.get("site_results", []))

    lines: List[str] = []
    lines.append("BABAPPAi Interpretation Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Verdict")
    lines.append("-" * 60)
    if p_emp is not None:
        lines.append(f"Empirical p-value (p_emp): {float(p_emp):.4g}")
    if q_emp is not None:
        lines.append(f"BH-adjusted q-value (q_emp): {float(q_emp):.4g}")
    if alpha_used is not None:
        lines.append(
            f"Significant at q <= {float(alpha_used):.2f}: "
            f"{'YES' if bool(significant_bool) else 'NO'} ({significance_label})"
        )
    lines.append(f"Gene-level raw EII_z: {float(eii_z_raw):.3f}")
    lines.append(f"Gene-level raw EII_01: {float(eii_01_raw):.3f}")
    lines.append(f"cEII_gene (P[gene identifiable]): {float(ceii_gene):.3f}" if ceii_gene is not None else "cEII_gene: n/a")
    lines.append(f"cEII_site (P[site identifiable]): {float(ceii_site):.3f}" if ceii_site is not None else "cEII_site: n/a")
    if isinstance(ceii_ci, Mapping):
        gene_ci = ceii_ci.get("gene", {})
        if isinstance(gene_ci, Mapping):
            lo = gene_ci.get("lower")
            hi = gene_ci.get("upper")
            if lo is not None and hi is not None:
                lines.append(f"cEII_gene CI: [{float(lo):.3f}, {float(hi):.3f}]")
    lines.append(f"cEII gene class: {extent}")
    lines.append(f"cEII site class: {ceii_site_class}")
    lines.append(f"cEII gene identifiable at calibrated threshold: {'YES' if bool(identifiable) else 'NO'}")
    lines.append(f"Calibration version: {calibration_version}")
    lines.append(f"Applicability/domain flag: {applicability}")
    lines.append(f"Applicability status: {applicability_status}")
    if applicability_score is not None:
        lines.append(f"Applicability score: {float(applicability_score):.3f}")
    if within_envelope is not None:
        lines.append(f"Within applicability envelope: {'YES' if bool(within_envelope) else 'NO'}")
    if nearest_supported_regime:
        lines.append(f"Nearest supported regime: {nearest_supported_regime}")
    if distance_to_supported_domain is not None:
        lines.append(f"Distance to supported domain: {float(distance_to_supported_domain):.3f}")
    if calib_unavailable_reason:
        lines.append(f"Calibration unavailable reason: {calib_unavailable_reason}")
    lines.append(_regime_explanation(str(extent)))
    lines.append("")

    lines.append("Strongest Branches")
    lines.append("-" * 60)
    for idx, row in enumerate(branch_results[:top_branches], start=1):
        lines.append(
            f"{idx:2d}. {row['branch']:<30} "
            f"score={float(row['background_score']):.4f} "
            f"selected={bool(row.get('selected_foreground', False))}"
        )
    if not branch_results:
        lines.append("No branch-level rows available.")
    lines.append("")

    lines.append("Top Elevated Sites")
    lines.append("-" * 60)
    for row in site_results[:top_sites]:
        lines.append(
            f"Site {int(row['site']):4d} "
            f"score={float(row['site_score']):.4f} "
            f"logit_mean={float(row['site_logit_mean']):+.4f}"
        )
    if not site_results:
        lines.append("No site-level rows available.")
    lines.append("")

    lines.append("Interpretation Policy")
    lines.append("-" * 60)
    lines.append("raw EII_z / EII_01 are dispersion magnitude diagnostics.")
    lines.append("cEII_gene / cEII_site are empirically calibrated probabilities of recoverability.")
    lines.append("cEII class labels are derived from the loaded calibration asset, not fixed constants.")
    lines.append("")
    lines.append(
        "Inferential significance is based on empirical p_emp/q_emp from matched neutral calibration; "
        "this is distinct from cEII probability calibration."
    )
    lines.append(
        "Warning: significance indicates excess dispersion relative to matched neutral expectation, "
        "not proof of adaptive substitution."
    )

    return "\n".join(lines) + "\n"


def interpret_results(result_json, out_path=None, top_branches=10, top_sites=20):
    report = render_interpretation(
        result_json,
        top_branches=top_branches,
        top_sites=top_sites,
    )

    if out_path is None:
        if isinstance(result_json, (str, Path)):
            out_path = Path(result_json).with_name("interpretation.txt")
        else:
            raise ValueError("out_path is required when result_json is a mapping")

    out_path = Path(out_path)
    out_path.write_text(report)
    return out_path
