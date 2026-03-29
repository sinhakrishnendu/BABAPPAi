#!/usr/bin/env python3
"""Run sigma0 robustness ablations across dispersion statistics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.dispersion import PRIMARY_DISPERSION_METHOD, SUPPORTED_DISPERSION_METHODS  # noqa: E402
from babappai.validation.full_pipeline_validation import run_full_pipeline_inference_on_dataset  # noqa: E402


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t") if row]


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _subset_dataset(dataset_tsv: Path, out_path: Path, max_cases: int) -> Path:
    rows = _read_tsv(dataset_tsv)
    if max_cases <= 0 or max_cases >= len(rows):
        return dataset_tsv

    # Deterministic balanced pick by stratum_id.
    by_stratum: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_stratum.setdefault(str(row.get("stratum_id", "unknown")), []).append(row)
    for key in by_stratum:
        by_stratum[key] = sorted(
            by_stratum[key],
            key=lambda r: (str(r.get("scenario_id", "")), str(r.get("replicate_id", ""))),
        )

    selected: List[Dict[str, str]] = []
    strata = sorted(by_stratum)
    cursor = 0
    while len(selected) < max_cases and strata:
        key = strata[cursor % len(strata)]
        bucket = by_stratum[key]
        if bucket:
            selected.append(bucket.pop(0))
        if not any(by_stratum[s] for s in strata):
            break
        cursor += 1

    # Fallback fill if needed.
    if len(selected) < max_cases:
        remaining: List[Dict[str, str]] = []
        for key in strata:
            remaining.extend(by_stratum[key])
        selected.extend(remaining[: max_cases - len(selected)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in selected[:max_cases]:
            writer.writerow(row)
    return out_path


def _separation_metric(metrics_tsv: Path) -> float:
    rows = _read_tsv(metrics_tsv)
    if not rows:
        return float("nan")
    neutral = [_safe_float(r.get("eii_01_raw")) for r in rows if str(r.get("regime")) == "neutral"]
    high = [_safe_float(r.get("eii_01_raw")) for r in rows if str(r.get("regime")) == "high"]
    neutral = [x for x in neutral if np.isfinite(x)]
    high = [x for x in high if np.isfinite(x)]
    if not neutral or not high:
        return float("nan")
    return float(np.median(high) - np.median(neutral))


def _load_calibration_metrics(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in _read_tsv(path):
        split = str(row.get("split"))
        target = str(row.get("target"))
        out[(split, target)] = {
            "brier": _safe_float(row.get("brier")),
            "ece": _safe_float(row.get("ece")),
            "ppv": _safe_float(row.get("ppv_at_threshold")),
            "fdr": _safe_float(row.get("fdr_at_threshold")),
            "balanced_accuracy": _safe_float(row.get("balanced_accuracy_at_threshold")),
            "n": _safe_float(row.get("n")),
            "positive_rate": _safe_float(row.get("positive_rate")),
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-tsv",
        default="results/validation/ceii_benchmark_v2_expanded/dataset/synthetic_dataset.tsv",
    )
    p.add_argument("--outdir", default="results/validation/sigma0_ablation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sigma-floor", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--neutral-reps", type=int, default=10)
    p.add_argument("--min-neutral-group-size", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=40)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_tsv = Path(args.dataset_tsv).expanduser().resolve()
    sampled_dataset = _subset_dataset(
        dataset_tsv,
        outdir / "sampled_dataset.tsv",
        int(args.max_cases),
    )

    method_rows: List[Dict[str, Any]] = []
    reliability_rows: List[Dict[str, Any]] = []
    decision_details: List[str] = []

    methods = [
        "site_logit_variance",
        "site_logit_winsorized_variance",
        "site_logit_clipped_variance",
        "site_logit_mad_scaled",
    ]
    for m in methods:
        if m not in SUPPORTED_DISPERSION_METHODS:
            raise RuntimeError(f"Unsupported dispersion method in ablation list: {m}")

    for method in methods:
        method_dir = outdir / method
        inference_dir = method_dir / "inference"
        calibration_dir = method_dir / "calibration"
        inference_meta = run_full_pipeline_inference_on_dataset(
            dataset_tsv=str(sampled_dataset),
            outdir=str(inference_dir),
            tree_calibration=False,
            n_calibration=0,
            device=str(args.device),
            batch_size=int(args.batch_size),
            sigma_floor=float(args.sigma_floor),
            alpha=float(args.alpha),
            pvalue_mode="empirical_monte_carlo",
            dispersion_method=method,
            min_neutral_group_size=int(args.min_neutral_group_size),
            neutral_reps=int(args.neutral_reps),
            offline=bool(args.offline),
            overwrite=bool(args.overwrite),
        )

        fit_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "fit_ceii_calibration.py"),
            "--metrics-tsv",
            str(inference_meta["metrics_tsv"]),
            "--outdir",
            str(calibration_dir),
            "--calibration-version",
            f"ceii_ablation_{method}",
            "--bootstrap-reps",
            "100",
            "--seed",
            str(int(args.seed)),
            "--app-min-n-taxa",
            "8",
            "--app-max-n-taxa",
            "64",
            "--app-min-gene-length-nt",
            "120",
            "--app-max-gene-length-nt",
            "4500",
        ]
        sep_metric = _separation_metric(Path(inference_meta["metrics_tsv"]))
        floor_frac = _safe_float(inference_meta.get("fraction_sigma0_at_floor"))
        fallback_frac = _safe_float(inference_meta.get("fraction_fallback_applied"))

        calibration_error: str = ""
        split_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
        try:
            subprocess.run(fit_cmd, check=True)
            split_metrics = _load_calibration_metrics(calibration_dir / "ceii_split_performance.tsv")
        except subprocess.CalledProcessError as exc:
            calibration_error = str(exc)

        test_gene = split_metrics.get(("test", "I_gene"), {})
        test_site = split_metrics.get(("test", "I_site"), {})
        ood_gene = split_metrics.get(("ood", "I_gene"), {})
        ood_site = split_metrics.get(("ood", "I_site"), {})
        brier_test_mean = float(np.nanmean([test_gene.get("brier", float("nan")), test_site.get("brier", float("nan"))]))
        brier_ood_mean = float(np.nanmean([ood_gene.get("brier", float("nan")), ood_site.get("brier", float("nan"))]))
        ece_test_mean = float(np.nanmean([test_gene.get("ece", float("nan")), test_site.get("ece", float("nan"))]))

        # Higher is better: separation, lower collapse/fallback/reliability errors.
        calibration_success = bool(split_metrics)
        objective = float(
            (sep_metric if np.isfinite(sep_metric) else -1.0)
            - 2.0 * (floor_frac if np.isfinite(floor_frac) else 1.0)
            - 1.0 * (fallback_frac if np.isfinite(fallback_frac) else 1.0)
            - 0.5 * (brier_test_mean if np.isfinite(brier_test_mean) else 1.0)
            - 0.3 * (ece_test_mean if np.isfinite(ece_test_mean) else 1.0)
        )
        if not calibration_success:
            objective -= 5.0

        row = {
            "dispersion_method": method,
            "calibration_success": int(calibration_success),
            "calibration_error": calibration_error,
            "n_cases": int(args.max_cases),
            "neutral_reps": int(args.neutral_reps),
            "sigma_floor": float(args.sigma_floor),
            "fraction_sigma0_at_floor": floor_frac,
            "fraction_fallback_applied": fallback_frac,
            "separation_high_minus_neutral_eii01_median": sep_metric,
            "test_brier_gene": test_gene.get("brier"),
            "test_brier_site": test_site.get("brier"),
            "test_ece_gene": test_gene.get("ece"),
            "test_ece_site": test_site.get("ece"),
            "ood_brier_gene": ood_gene.get("brier"),
            "ood_brier_site": ood_site.get("brier"),
            "brier_test_mean": brier_test_mean,
            "brier_ood_mean": brier_ood_mean,
            "ece_test_mean": ece_test_mean,
            "objective_score": objective,
            "metrics_tsv": str(inference_meta["metrics_tsv"]),
            "split_performance_tsv": str(calibration_dir / "ceii_split_performance.tsv"),
            "reliability_tsv": str(calibration_dir / "ceii_reliability.tsv"),
        }
        method_rows.append(row)

        reliability_path = calibration_dir / "ceii_reliability.tsv"
        if reliability_path.exists():
            for metric_row in _read_tsv(reliability_path):
                reliability_rows.append(
                    {
                        "dispersion_method": method,
                        **metric_row,
                    }
                )
        decision_details.append(
            f"- {method}: success={calibration_success}, floor={floor_frac:.3f}, fallback={fallback_frac:.3f}, "
            f"sep={sep_metric:.3f}, test_brier_mean={brier_test_mean:.3f}, objective={objective:.3f}"
        )

    method_rows_sorted = sorted(method_rows, key=lambda r: float(r["objective_score"]), reverse=True)
    best = method_rows_sorted[0]

    _write_tsv(outdir / "sigma0_ablation_summary.tsv", method_rows_sorted)
    _write_tsv(outdir / "sigma0_ablation_reliability.tsv", reliability_rows)

    report_lines = [
        "# Sigma0 Ablation Decision",
        "",
        f"- Dataset: `{sampled_dataset}`",
        f"- Cases evaluated: {int(args.max_cases)}",
        f"- neutral_reps per run: {int(args.neutral_reps)}",
        f"- sigma_floor: {float(args.sigma_floor)}",
        "",
        "## Per-method summary",
        *decision_details,
        "",
        "## Chosen method",
        f"- Selected dispersion method: `{best['dispersion_method']}`",
        "- Selection criterion: maximize objective score combining lower sigma-floor collapse, lower fallback, better reliability, and better neutral-vs-high separation.",
    ]
    (outdir / "sigma0_ablation_decision.md").write_text("\n".join(report_lines) + "\n")

    payload = {
        "sampled_dataset": str(sampled_dataset),
        "best_method": best["dispersion_method"],
        "summary_tsv": str(outdir / "sigma0_ablation_summary.tsv"),
        "reliability_tsv": str(outdir / "sigma0_ablation_reliability.tsv"),
        "decision_md": str(outdir / "sigma0_ablation_decision.md"),
    }
    (outdir / "sigma0_ablation_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
