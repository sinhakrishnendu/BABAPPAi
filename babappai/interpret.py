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
            "identifiability at the gene level."
        ),
        "weak_or_ambiguous": (
            "Signal is present but ambiguous; interpret as exploratory and seek "
            "additional sequence depth or broader taxon sampling."
        ),
        "identifiable": (
            "The gene-level pattern is measurably identifiable under the chosen "
            "calibration regime."
        ),
        "strongly_identifiable": (
            "The gene-level pattern is strongly identifiable under the chosen "
            "calibration regime."
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
    eii_z = gene.get("EII_z")
    eii_01 = gene.get("EII_01")
    p_emp = gene.get("p_emp")
    q_emp = gene.get("q_emp")
    alpha_used = gene.get("alpha_used")
    significant_bool = gene.get("significant_bool")
    significance_label = gene.get("significance_label")
    identifiable = gene.get("identifiable_bool")
    extent = gene.get("identifiability_extent")

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
    lines.append(f"Gene-level EII_z: {float(eii_z):.3f}")
    lines.append(f"Gene-level EII_01: {float(eii_01):.3f}")
    lines.append(f"Descriptive EII band: {extent}")
    lines.append(f"Legacy identifiable_bool (descriptive): {'YES' if identifiable else 'NO'}")
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

    lines.append("Interpretation Policy (Descriptive Bands)")
    lines.append("-" * 60)
    lines.append("0.00 <= EII_01 < 0.30  -> not_identifiable")
    lines.append("0.30 <= EII_01 < 0.70  -> weak_or_ambiguous")
    lines.append("0.70 <= EII_01 < 0.90  -> identifiable")
    lines.append("0.90 <= EII_01 <= 1.00 -> strongly_identifiable")
    lines.append("")
    lines.append(
        "Inferential significance is based on empirical p_emp/q_emp from matched neutral calibration; "
        "EII bands are reporting-only."
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
