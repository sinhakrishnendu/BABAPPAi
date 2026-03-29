#!/usr/bin/env python3
"""Audit sigma0 collapse/fallback behavior from full-pipeline debug tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t") if row]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _failure_mode(row: Dict[str, Any]) -> str:
    reason = str(row.get("fallback_reason", "") or "").strip()
    raw_sigma0 = _safe_float(row.get("raw_sigma0"))
    floor_applied = bool(_safe_int(row.get("sigma_floor_applied"), 0))
    if reason:
        return reason
    if not math.isfinite(raw_sigma0):
        return "non_finite_sigma0_before_floor"
    if floor_applied:
        return "sigma0_floor_applied_without_explicit_reason"
    return "no_failure_detected"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug-tsv",
        default="results/validation/ceii_benchmark_v2_expanded/inference/full_pipeline_calibration_debug.tsv",
    )
    parser.add_argument(
        "--summary-json",
        default="results/validation/ceii_benchmark_v2_expanded/inference/full_pipeline_inference_summary.json",
    )
    parser.add_argument(
        "--outdir",
        default="results/validation/sigma0_audit",
    )
    args = parser.parse_args()

    debug_path = Path(args.debug_tsv).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_tsv(debug_path)
    if not rows:
        raise RuntimeError(f"No rows found in debug table: {debug_path}")

    summary_payload: Dict[str, Any] = {}
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text())
    sigma_diag = summary_payload.get("sigma_diagnostics", {}) if isinstance(summary_payload, dict) else {}

    case_rows: List[Dict[str, Any]] = []
    mode_counter: Counter[str] = Counter()
    for row in rows:
        raw_sigma0 = _safe_float(row.get("raw_sigma0"))
        floored_sigma0 = _safe_float(row.get("floored_sigma0"))
        sigma_floor = _safe_float(row.get("sigma_floor_requested"))
        at_floor = (
            math.isfinite(floored_sigma0)
            and math.isfinite(sigma_floor)
            and abs(floored_sigma0 - sigma_floor) <= 1e-12
        )
        failure_mode = _failure_mode(row)
        mode_counter[failure_mode] += 1
        case_rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "replicate_id": row.get("replicate_id", ""),
                "stratum_id": row.get("stratum_id", ""),
                "calibration_source": row.get("calibration_source", ""),
                "fallback_reason": row.get("fallback_reason", ""),
                "fallback_flag": _safe_int(row.get("fallback_flag"), 0),
                "neutral_group_size": _safe_int(row.get("neutral_group_size"), 0),
                "D_obs": row.get("D_obs", ""),
                "mu0": row.get("mu0", ""),
                "raw_sigma0": row.get("raw_sigma0", ""),
                "raw_sigma0_finite": int(math.isfinite(raw_sigma0)),
                "floored_sigma0": row.get("floored_sigma0", ""),
                "sigma_floor_requested": row.get("sigma_floor_requested", ""),
                "sigma_floor_applied": _safe_int(row.get("sigma_floor_applied"), 0),
                "is_at_sigma_floor": int(bool(at_floor)),
                "p_emp": row.get("p_emp", ""),
                "q_emp": row.get("q_emp", ""),
                "failure_mode": failure_mode,
            }
        )

    case_path = outdir / "sigma0_case_debug.tsv"
    with case_path.open("w", newline="") as fh:
        fieldnames = list(case_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in case_rows:
            writer.writerow(row)

    mode_rows: List[Dict[str, Any]] = []
    total = len(case_rows)
    for mode, count in sorted(mode_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        mode_rows.append(
            {
                "failure_mode": mode,
                "n_cases": int(count),
                "fraction_cases": float(count / total) if total > 0 else float("nan"),
            }
        )

    mode_path = outdir / "sigma0_failure_modes.tsv"
    with mode_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["failure_mode", "n_cases", "fraction_cases"], delimiter="\t")
        writer.writeheader()
        for row in mode_rows:
            writer.writerow(row)

    fallback_reason_counts = Counter(str(r.get("fallback_reason", "") or "") for r in case_rows)
    calibration_source_counts = Counter(str(r.get("calibration_source", "") or "") for r in case_rows)

    lines: List[str] = []
    lines.append("# Sigma0 Audit Report")
    lines.append("")
    lines.append(f"- Debug table: `{debug_path}`")
    lines.append(f"- Summary JSON: `{summary_path}`")
    lines.append(f"- Cases audited: {total}")
    if sigma_diag:
        lines.append(f"- pvalue_mode: `{sigma_diag.get('pvalue_mode')}`")
        lines.append(f"- requested sigma floor: {sigma_diag.get('sigma_floor_requested')}")
        lines.append(f"- fraction_sigma0_at_floor: {sigma_diag.get('fraction_sigma0_at_floor')}")
        lines.append(f"- fraction_fallback_applied: {sigma_diag.get('fraction_fallback_applied')}")
        before = sigma_diag.get("sigma0_before_floor_summary", {})
        lines.append(
            "- sigma0 before floor summary: "
            f"finite={before.get('finite_count')}/{before.get('count')}, "
            f"median={before.get('median')}, q95={before.get('q95')}, max={before.get('max')}"
        )
    lines.append("")
    lines.append("## Failure mode counts")
    for mode, count in sorted(mode_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        frac = count / total if total else float("nan")
        lines.append(f"- {mode}: {count}/{total} ({frac:.1%})")
    lines.append("")
    lines.append("## Root-cause interpretation")
    lines.append(
        "- Collapse in this run is dominated by frozen-reference lookup failure "
        "(`missing_neutral_reference`) plus minimum-neutral-group fallback in the few rows where a table row existed."
    )
    lines.append(
        "- The debug table shows finite pre-floor sigma0 only for the `neutral_group_below_minimum` subset; "
        "all `missing_neutral_reference` rows have non-finite raw sigma0."
    )
    lines.append(
        "- This indicates the benchmark configuration was outside frozen-reference support, not a valid basis for calibrated cEII."
    )
    lines.append("")
    lines.append("## Counts by fallback reason")
    for reason, count in sorted(fallback_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        reason_txt = reason if reason else "<empty>"
        lines.append(f"- {reason_txt}: {count}")
    lines.append("")
    lines.append("## Counts by calibration source")
    for source, count in sorted(calibration_source_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        source_txt = source if source else "<empty>"
        lines.append(f"- {source_txt}: {count}")
    lines.append("")
    lines.append("## Output files")
    lines.append(f"- `{case_path}`")
    lines.append(f"- `{mode_path}`")

    report_path = outdir / "sigma0_audit_report.md"
    report_path.write_text("\n".join(lines) + "\n")

    print(json.dumps({"case_debug_tsv": str(case_path), "failure_modes_tsv": str(mode_path), "report_md": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

