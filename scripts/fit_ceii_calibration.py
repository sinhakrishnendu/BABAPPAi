#!/usr/bin/env python3
"""Fit empirical cEII calibration assets from held-out synthetic truth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.calibration.ceii import (  # noqa: E402
    D_OBS_DEFINITION,
    binary_metrics,
    brier_score,
    derive_threshold,
    expected_calibration_error,
    fit_isotonic_binary,
    predict_isotonic,
    save_calibration_asset,
)
from babappai.calibration.recoverability import (  # noqa: E402
    assign_scenario_splits,
    attach_recoverability_targets,
    attach_scenario_stability,
    compute_truth_aware_metrics,
)


def _read_tsv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open() as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t") if row]


def _write_tsv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _subset(rows: List[Mapping[str, Any]], split: str) -> List[Mapping[str, Any]]:
    return [r for r in rows if str(r.get("split", "")) == split]


def _bootstrap_ci_calibrator(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_grid: np.ndarray,
    bootstrap_reps: int,
    seed: int,
) -> Dict[str, List[float]]:
    if x.size == 0:
        return {"x": x_grid.tolist(), "lower": [0.0] * x_grid.size, "upper": [1.0] * x_grid.size}
    rng = np.random.default_rng(seed)
    preds = []
    n = x.size
    for _ in range(int(bootstrap_reps)):
        idx = rng.integers(0, n, size=n)
        cal = fit_isotonic_binary(x[idx], y[idx])
        preds.append(predict_isotonic(cal, x_grid))
    arr = np.asarray(preds, dtype=float)
    return {
        "x": x_grid.tolist(),
        "lower": np.quantile(arr, 0.025, axis=0).tolist(),
        "upper": np.quantile(arr, 0.975, axis=0).tolist(),
    }


def _reliability_rows(y_true: np.ndarray, p_pred: np.ndarray, *, split: str, target: str, n_bins: int = 12) -> List[Dict[str, Any]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, Any]] = []
    for i in range(n_bins):
        left = float(bins[i])
        right = float(bins[i + 1])
        if i == n_bins - 1:
            mask = (p_pred >= left) & (p_pred <= right)
        else:
            mask = (p_pred >= left) & (p_pred < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "split": split,
                "target": target,
                "bin_left": left,
                "bin_right": right,
                "n": int(np.sum(mask)),
                "mean_pred": float(np.mean(p_pred[mask])),
                "empirical_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def _stabilize_threshold(
    *,
    threshold: float,
    p_cal: np.ndarray,
    lower_q: float,
    upper_q: float,
) -> float:
    if np.isfinite(threshold) and 0.0 < float(threshold) < 1.0:
        return float(threshold)
    if p_cal.size == 0:
        return 0.5
    lo = float(np.quantile(p_cal, lower_q))
    hi = float(np.quantile(p_cal, upper_q))
    if not np.isfinite(lo):
        lo = 0.25
    if not np.isfinite(hi):
        hi = 0.75
    return float(np.clip(0.5 * (lo + hi), 0.01, 0.99))


def _evaluate_target(
    rows: List[Mapping[str, Any]],
    calibrator: Mapping[str, Any],
    *,
    label_key: str,
    target_name: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for split in ("calibration", "test", "ood"):
        part = _subset(rows, split)
        if not part:
            continue
        x = np.asarray([float(r["EII_z"]) for r in part], dtype=float)
        y = np.asarray([int(r[label_key]) for r in part], dtype=int)
        p = predict_isotonic(calibrator, x)
        fixed = {
            "split": split,
            "target": target_name,
            "n": int(y.size),
            "positive_rate": float(np.mean(y)),
            "brier": brier_score(y, p),
            "ece": expected_calibration_error(y, p),
            "threshold": float(threshold),
        }
        fixed.update({"ppv_at_threshold": float("nan"), "fdr_at_threshold": float("nan"), "balanced_accuracy_at_threshold": float("nan")})

        thr = binary_metrics(y, p, float(threshold))
        fixed["ppv_at_threshold"] = float(thr["ppv"])
        fixed["fdr_at_threshold"] = float(thr["fdr"])
        fixed["balanced_accuracy_at_threshold"] = float(thr["balanced_accuracy"])
        out.append(fixed)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-tsv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--calibration-version", default="ceii_v1")
    p.add_argument("--tau-gene", type=float, default=0.42)
    p.add_argument("--tau-site", type=float, default=0.45)
    p.add_argument("--target-fdr-gene", type=float, default=0.10)
    p.add_argument("--target-fdr-site", type=float, default=0.10)
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--write-package-asset", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_tsv(args.metrics_tsv)
    if not rows:
        raise ValueError("No rows in metrics TSV.")

    augmented: List[Dict[str, Any]] = []
    for row in rows:
        extra = compute_truth_aware_metrics(row)
        augmented.append({**row, **extra})

    attach_scenario_stability(augmented)
    attach_recoverability_targets(augmented, tau_gene=float(args.tau_gene), tau_site=float(args.tau_site))
    assign_scenario_splits(augmented, seed=int(args.seed))

    _write_tsv(outdir / "recoverability_augmented.tsv", augmented)

    cal_rows = _subset(augmented, "calibration")
    if not cal_rows:
        raise RuntimeError("Calibration split is empty.")

    x_cal = np.asarray([float(r["EII_z"]) for r in cal_rows], dtype=float)
    y_gene_cal = np.asarray([int(r["I_gene"]) for r in cal_rows], dtype=int)
    y_site_cal = np.asarray([int(r["I_site"]) for r in cal_rows], dtype=int)

    gene_cal = fit_isotonic_binary(x_cal, y_gene_cal)
    site_cal = fit_isotonic_binary(x_cal, y_site_cal)

    p_gene_cal = predict_isotonic(gene_cal, x_cal)
    p_site_cal = predict_isotonic(site_cal, x_cal)

    thr_gene_main = derive_threshold(y_gene_cal, p_gene_cal, target_fdr=float(args.target_fdr_gene))
    thr_site_main = derive_threshold(y_site_cal, p_site_cal, target_fdr=float(args.target_fdr_site))
    thr_gene_weak = derive_threshold(y_gene_cal, p_gene_cal, target_fdr=0.25)
    thr_gene_strong = derive_threshold(y_gene_cal, p_gene_cal, target_fdr=0.05)
    thr_site_weak = derive_threshold(y_site_cal, p_site_cal, target_fdr=0.30)
    thr_site_strong = derive_threshold(y_site_cal, p_site_cal, target_fdr=0.05)

    # enforce non-degenerate monotone class edges with calibration-data fallback.
    g_main = _stabilize_threshold(
        threshold=float(thr_gene_main["threshold"]),
        p_cal=p_gene_cal,
        lower_q=0.45,
        upper_q=0.75,
    )
    s_main = _stabilize_threshold(
        threshold=float(thr_site_main["threshold"]),
        p_cal=p_site_cal,
        lower_q=0.55,
        upper_q=0.85,
    )
    g_weak = float(
        np.clip(
            min(_stabilize_threshold(threshold=float(thr_gene_weak["threshold"]), p_cal=p_gene_cal, lower_q=0.30, upper_q=0.50), g_main),
            0.0,
            g_main,
        )
    )
    g_strong = float(
        np.clip(
            max(_stabilize_threshold(threshold=float(thr_gene_strong["threshold"]), p_cal=p_gene_cal, lower_q=0.70, upper_q=0.90), g_main),
            g_main,
            1.0,
        )
    )
    s_weak = float(
        np.clip(
            min(_stabilize_threshold(threshold=float(thr_site_weak["threshold"]), p_cal=p_site_cal, lower_q=0.35, upper_q=0.55), s_main),
            0.0,
            s_main,
        )
    )
    s_strong = float(
        np.clip(
            max(_stabilize_threshold(threshold=float(thr_site_strong["threshold"]), p_cal=p_site_cal, lower_q=0.80, upper_q=0.95), s_main),
            s_main,
            1.0,
        )
    )
    # Guarantee strict class ordering so weak/identifiable bands are non-empty.
    if not (g_weak < g_main):
        g_weak = float(np.clip(g_main - 0.10, 0.0, max(0.0, g_main - 1e-6)))
    if not (g_main < g_strong):
        g_strong = float(np.clip(g_main + 0.10, min(1.0, g_main + 1e-6), 1.0))
    if not (s_weak < s_main):
        s_weak = float(np.clip(s_main - 0.10, 0.0, max(0.0, s_main - 1e-6)))
    if not (s_main < s_strong):
        s_strong = float(np.clip(s_main + 0.10, min(1.0, s_main + 1e-6), 1.0))

    x_grid = np.unique(np.asarray(gene_cal["x"], dtype=float))
    gene_ci = _bootstrap_ci_calibrator(
        x_cal,
        y_gene_cal,
        x_grid=x_grid,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 101,
    )
    site_ci = _bootstrap_ci_calibrator(
        x_cal,
        y_site_cal,
        x_grid=x_grid,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 202,
    )

    cal_taxa = np.asarray([int(float(r["n_taxa"])) for r in cal_rows], dtype=int)
    cal_len = np.asarray([int(float(r["gene_length_nt"])) for r in cal_rows], dtype=int)
    asset = {
        "calibration_version": str(args.calibration_version),
        "d_obs_definition": D_OBS_DEFINITION,
        "raw_eii_definition": "eii_z_raw = (D_obs - mu0) / max(sigma0_raw, sigma_floor), eii_01_raw = sigmoid(eii_z_raw)",
        "target_definitions": {
            "R_gene": "0.45*branch_rank_norm + 0.35*burden_alignment + 0.20*scenario_branch_stability",
            "R_site": "0.45*site_enrichment_at_k + 0.35*site_rank_norm + 0.20*scenario_site_stability",
            "I_gene": f"1 if R_gene >= {float(args.tau_gene):.2f} else 0",
            "I_site": f"1 if R_site >= {float(args.tau_site):.2f} else 0",
        },
        "gene_calibrator": gene_cal,
        "site_calibrator": site_cal,
        "prediction_ci": {
            "gene_lower": {"x": gene_ci["x"], "y": gene_ci["lower"]},
            "gene_upper": {"x": gene_ci["x"], "y": gene_ci["upper"]},
            "site_lower": {"x": site_ci["x"], "y": site_ci["lower"]},
            "site_upper": {"x": site_ci["x"], "y": site_ci["upper"]},
        },
        "thresholds": {
            "gene": {
                "threshold": g_main,
                "target_fdr": float(args.target_fdr_gene),
                "weak_threshold": g_weak,
                "strong_threshold": g_strong,
            },
            "site": {
                "threshold": s_main,
                "target_fdr": float(args.target_fdr_site),
                "weak_threshold": s_weak,
                "strong_threshold": s_strong,
            },
        },
        "classes": {
            "gene": [
                {"label": "not_identifiable", "min": 0.0, "max": g_weak},
                {"label": "weak_or_ambiguous", "min": g_weak, "max": g_main},
                {"label": "identifiable", "min": g_main, "max": g_strong},
                {"label": "strongly_identifiable", "min": g_strong, "max": 1.0},
            ],
            "site": [
                {"label": "not_identifiable", "min": 0.0, "max": s_weak},
                {"label": "weak_or_ambiguous", "min": s_weak, "max": s_main},
                {"label": "identifiable", "min": s_main, "max": s_strong},
                {"label": "strongly_identifiable", "min": s_strong, "max": 1.0},
            ],
        },
        "applicability": {
            "min_n_taxa": int(np.min(cal_taxa)),
            "max_n_taxa": int(np.max(cal_taxa)),
            "min_gene_length_nt": int(np.min(cal_len)),
            "max_gene_length_nt": int(np.max(cal_len)),
        },
        "provenance": {
            "metrics_tsv_name": str(Path(args.metrics_tsv).name),
            "bootstrap_reps": int(args.bootstrap_reps),
            "seed": int(args.seed),
        },
    }

    asset_path = save_calibration_asset(asset, outdir / "ceii_calibration_asset.json")

    metrics_rows: List[Dict[str, Any]] = []
    metrics_rows.extend(
        _evaluate_target(
            augmented,
            gene_cal,
            label_key="I_gene",
            target_name="I_gene",
            threshold=float(asset["thresholds"]["gene"]["threshold"]),
        )
    )
    metrics_rows.extend(
        _evaluate_target(
            augmented,
            site_cal,
            label_key="I_site",
            target_name="I_site",
            threshold=float(asset["thresholds"]["site"]["threshold"]),
        )
    )
    _write_tsv(outdir / "ceii_split_performance.tsv", metrics_rows)

    reliability_rows: List[Dict[str, Any]] = []
    for split in ("calibration", "test", "ood"):
        part = _subset(augmented, split)
        if not part:
            continue
        x = np.asarray([float(r["EII_z"]) for r in part], dtype=float)
        y_gene = np.asarray([int(r["I_gene"]) for r in part], dtype=int)
        y_site = np.asarray([int(r["I_site"]) for r in part], dtype=int)
        p_gene = predict_isotonic(gene_cal, x)
        p_site = predict_isotonic(site_cal, x)
        reliability_rows.extend(_reliability_rows(y_gene, p_gene, split=split, target="I_gene"))
        reliability_rows.extend(_reliability_rows(y_site, p_site, split=split, target="I_site"))
    _write_tsv(outdir / "ceii_reliability.tsv", reliability_rows)

    if args.write_package_asset:
        pkg_path = REPO_ROOT / "babappai" / "data" / "ceii_calibration_v1.json"
        save_calibration_asset(asset, pkg_path)
        print(f"Wrote package calibration asset: {pkg_path}")

    summary = {
        "asset_path": str(asset_path),
        "recoverability_augmented_tsv": str(outdir / "recoverability_augmented.tsv"),
        "split_performance_tsv": str(outdir / "ceii_split_performance.tsv"),
        "reliability_tsv": str(outdir / "ceii_reliability.tsv"),
        "calibration_version": str(args.calibration_version),
    }
    (outdir / "ceii_calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
