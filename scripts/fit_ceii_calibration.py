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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    try:
        return bool(int(float(text)))
    except (TypeError, ValueError):
        return bool(default)


def _feature_value(row: Mapping[str, Any], feature_name: str) -> float:
    if feature_name == "eii_z_raw":
        v = _safe_float(row.get("eii_z_raw", row.get("EII_z")))
        return v
    if feature_name == "eii_01_raw":
        v = _safe_float(row.get("eii_01_raw", row.get("EII_01")))
        return v
    if feature_name == "n_taxa":
        return _safe_float(row.get("n_taxa"))
    if feature_name == "gene_length_nt":
        return _safe_float(row.get("gene_length_nt"))
    if feature_name == "n_branches":
        direct = _safe_float(row.get("n_branches"))
        if np.isfinite(direct):
            return direct
        n_taxa = _safe_float(row.get("n_taxa"))
        if np.isfinite(n_taxa):
            return max(1.0, 2.0 * n_taxa - 3.0)
        return float("nan")
    if feature_name == "eii_z_clipped":
        v = _safe_float(row.get("eii_z_raw", row.get("EII_z")))
        if np.isfinite(v):
            return float(np.clip(v, -12.0, 12.0))
        return float("nan")
    if feature_name == "q_emp":
        v = _safe_float(row.get("q_emp"))
        if np.isfinite(v):
            return float(np.clip(v, 0.0, 1.0))
        return float("nan")
    if feature_name == "neglog10_q_emp":
        qv = _feature_value(row, "q_emp")
        if np.isfinite(qv):
            return float(-np.log10(max(qv, 1e-12)))
        return 0.0
    if feature_name == "dispersion_ratio":
        direct = _safe_float(row.get("dispersion_ratio"))
        if np.isfinite(direct):
            return max(0.0, float(direct))
        d_obs = _safe_float(row.get("D_obs"))
        mu0 = _safe_float(row.get("mu0"))
        if np.isfinite(d_obs) and np.isfinite(mu0) and mu0 > 0:
            return max(0.0, float(d_obs / mu0))
        return float("nan")
    if feature_name == "dispersion_ratio_clipped":
        ratio = _feature_value(row, "dispersion_ratio")
        if np.isfinite(ratio):
            return float(np.clip(ratio, 0.0, 10.0))
        return 0.0
    if feature_name == "sigma0_final":
        for key in ("sigma0_final", "sigma0", "neutral_sd_floored", "neutral_sd"):
            v = _safe_float(row.get(key))
            if np.isfinite(v):
                return max(0.0, float(v))
        return float("nan")
    if feature_name == "sigma0_inverse":
        s0 = _feature_value(row, "sigma0_final")
        if np.isfinite(s0):
            return float(1.0 / max(s0, 1e-8))
        return float("nan")
    if feature_name.startswith("log1p_"):
        base_name = feature_name[len("log1p_") :]
        base = _feature_value(row, base_name)
        if np.isfinite(base):
            return float(np.log1p(max(base, 0.0)))
        return float("nan")
    return _safe_float(row.get(feature_name))


def _rows_to_matrix(rows: List[Mapping[str, Any]], feature_names: List[str]) -> np.ndarray:
    mat = np.asarray(
        [[_feature_value(row, name) for name in feature_names] for row in rows],
        dtype=float,
    )
    if mat.size == 0:
        return mat
    col_means = np.nanmean(mat, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = ~np.isfinite(mat)
    if np.any(nan_mask):
        mat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return mat


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_1d(score: np.ndarray, y: np.ndarray, *, ridge_lambda: float = 1e-3) -> Dict[str, float]:
    x = np.asarray(score, dtype=float)
    t = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    if x.size == 0:
        return {"a": 0.0, "b": 0.0}
    mean_t = float(np.clip(np.mean(t), 1e-4, 1.0 - 1e-4))
    if np.all(t == t[0]):
        return {"a": 0.0, "b": float(np.log(mean_t / (1.0 - mean_t)))}

    X = np.column_stack([x, np.ones_like(x)])
    theta = np.asarray([0.0, float(np.log(mean_t / (1.0 - mean_t)))], dtype=float)
    reg = np.diag([float(ridge_lambda), 0.0])
    for _ in range(200):
        p = _sigmoid(X @ theta)
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = X.T @ (p - t) + reg @ theta
        hess = X.T @ (X * w[:, None]) + reg
        step = np.linalg.solve(hess, grad)
        theta = theta - step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    a = float(theta[0])
    b = float(theta[1])
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = 0.0
    # Keep calibration monotone increasing in score.
    if a < 0.0:
        a = abs(a)
    return {"a": a, "b": b}


def _predict_logistic_1d(model: Mapping[str, Any], score: np.ndarray) -> np.ndarray:
    a = float(model.get("a", 0.0))
    b = float(model.get("b", 0.0))
    return _sigmoid(a * np.asarray(score, dtype=float) + b)


def _predict_blended_probability(model: Mapping[str, Any], score: np.ndarray) -> np.ndarray:
    p_iso = predict_isotonic(model["isotonic_calibrator"], score)
    logi = model.get("logistic_calibrator")
    if isinstance(logi, Mapping):
        p_log = _predict_logistic_1d(logi, score)
        w_iso = float(np.clip(float(model.get("blend_weight_isotonic", 0.5)), 0.0, 1.0))
        return np.clip(w_iso * p_iso + (1.0 - w_iso) * p_log, 0.0, 1.0)
    return p_iso


def _fit_linear_score_isotonic(
    rows: List[Mapping[str, Any]],
    *,
    label_key: str,
    feature_names: List[str],
    ridge_lambda: float = 1e-3,
) -> Dict[str, Any]:
    x_raw = _rows_to_matrix(rows, feature_names)
    y = np.asarray([int(float(r[label_key])) for r in rows], dtype=float)
    if x_raw.size == 0:
        raise ValueError(f"No rows available for target {label_key}.")
    mu = np.mean(x_raw, axis=0)
    sigma = np.std(x_raw, axis=0)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    xz = (x_raw - mu) / sigma
    x_design = np.column_stack([np.ones(xz.shape[0], dtype=float), xz])
    reg = np.eye(x_design.shape[1], dtype=float) * float(ridge_lambda)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(x_design.T @ x_design + reg, x_design.T @ y)
    intercept = float(beta[0])
    coef = np.asarray(beta[1:], dtype=float)
    scores = intercept + (xz @ coef)
    iso = fit_isotonic_binary(scores, y)
    logi = _fit_logistic_1d(scores, y)
    p_iso = predict_isotonic(iso, scores)
    p_log = _predict_logistic_1d(logi, scores)
    # Prefer isotonic when it has resolution; otherwise rely more on smooth logistic mapping.
    iso_unique = len(np.unique(np.round(p_iso, 6)))
    blend_weight_iso = 0.35 if iso_unique <= 2 else 0.65
    return {
        "type": "linear_score_isotonic",
        "feature_names": list(feature_names),
        "feature_mean": mu.astype(float).tolist(),
        "feature_scale": sigma.astype(float).tolist(),
        "coef": coef.astype(float).tolist(),
        "intercept": intercept,
        "isotonic_calibrator": iso,
        "logistic_calibrator": logi,
        "blend_weight_isotonic": float(blend_weight_iso),
    }


def _fit_nonnegative_logistic_score(
    xz: np.ndarray,
    y: np.ndarray,
    *,
    ridge_lambda: float = 1e-3,
    max_iter: int = 2000,
    lr: float = 0.15,
) -> tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=float)
    xz = np.asarray(xz, dtype=float)
    if xz.ndim != 2:
        raise ValueError("xz must be 2D")
    n, p = xz.shape
    if n == 0:
        raise ValueError("No rows for nonnegative score fit.")
    mean_t = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    intercept = float(np.log(mean_t / (1.0 - mean_t)))
    coef = np.zeros(p, dtype=float)

    def _loss(b: float, w: np.ndarray) -> float:
        s = b + xz @ w
        p_hat = _sigmoid(s)
        eps = 1e-8
        nll = -np.mean(y * np.log(p_hat + eps) + (1.0 - y) * np.log(1.0 - p_hat + eps))
        reg = 0.5 * float(ridge_lambda) * float(np.sum(w * w))
        return float(nll + reg)

    prev = float("inf")
    for _ in range(int(max_iter)):
        s = intercept + xz @ coef
        p_hat = _sigmoid(s)
        grad_w = (xz.T @ (p_hat - y)) / float(n) + float(ridge_lambda) * coef
        grad_b = float(np.mean(p_hat - y))
        base_loss = _loss(intercept, coef)
        step = float(lr)
        updated = False
        for _ls in range(25):
            cand_w = np.maximum(0.0, coef - step * grad_w)
            cand_b = float(intercept - step * grad_b)
            cand_loss = _loss(cand_b, cand_w)
            if cand_loss <= base_loss + 1e-10:
                coef = cand_w
                intercept = cand_b
                updated = True
                break
            step *= 0.5
        if not updated:
            break
        current = _loss(intercept, coef)
        if abs(prev - current) < 1e-9:
            break
        prev = current
    return coef.astype(float), float(intercept)


def _fit_monotone_evidence_model(
    rows: List[Mapping[str, Any]],
    *,
    label_key: str,
    evidence_features: List[str],
    ridge_lambda: float = 1e-3,
) -> Dict[str, Any]:
    x_raw = _rows_to_matrix(rows, evidence_features)
    y = np.asarray([int(float(r[label_key])) for r in rows], dtype=float)
    if x_raw.size == 0:
        raise ValueError(f"No rows available for target {label_key}.")
    mu = np.mean(x_raw, axis=0)
    sigma = np.std(x_raw, axis=0)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    xz = (x_raw - mu) / sigma
    coef, intercept = _fit_nonnegative_logistic_score(
        xz,
        y,
        ridge_lambda=float(ridge_lambda),
    )
    score = intercept + (xz @ coef)
    iso = fit_isotonic_binary(score, y)
    return {
        "type": "linear_score_isotonic",
        "score_design": "monotone_evidence_nonnegative",
        "evidence_features_only": True,
        "monotonic_directions": {name: "nondecreasing" for name in evidence_features},
        "feature_names": list(evidence_features),
        "feature_mean": mu.astype(float).tolist(),
        "feature_scale": sigma.astype(float).tolist(),
        "coef": coef.astype(float).tolist(),
        "intercept": float(intercept),
        "isotonic_calibrator": iso,
    }


def _predict_linear_score(model: Mapping[str, Any], rows: List[Mapping[str, Any]]) -> np.ndarray:
    feature_names = [str(x) for x in model.get("feature_names", [])]
    if not feature_names:
        raise ValueError("linear_score_isotonic model missing feature_names")
    x_raw = _rows_to_matrix(rows, feature_names)
    mu = np.asarray(model.get("feature_mean", []), dtype=float)
    sigma = np.asarray(model.get("feature_scale", []), dtype=float)
    coef = np.asarray(model.get("coef", []), dtype=float)
    if mu.size != len(feature_names) or sigma.size != len(feature_names) or coef.size != len(feature_names):
        raise ValueError("linear_score_isotonic model payload has inconsistent dimensions")
    sigma = np.where(np.abs(sigma) > 1e-8, sigma, 1.0)
    xz = (x_raw - mu) / sigma
    intercept = float(model.get("intercept", 0.0))
    return intercept + (xz @ coef)


def _predict_target(model: Mapping[str, Any], rows: List[Mapping[str, Any]]) -> np.ndarray:
    if str(model.get("type", "")) == "linear_score_isotonic":
        score = _predict_linear_score(model, rows)
        return _predict_blended_probability(model, score)
    # backward compatibility: direct isotonic calibrator payload.
    x = np.asarray([float(r["EII_z"]) for r in rows], dtype=float)
    return predict_isotonic(model, x)


def _bootstrap_ci_calibrator(
    x_score: np.ndarray,
    y: np.ndarray,
    *,
    score_grid: np.ndarray,
    bootstrap_reps: int,
    seed: int,
    blend_with_logistic: bool = True,
) -> Dict[str, List[float]]:
    if x_score.size == 0:
        return {"x": score_grid.tolist(), "lower": [0.0] * score_grid.size, "upper": [1.0] * score_grid.size}
    rng = np.random.default_rng(seed)
    preds = []
    n = x_score.size
    for _ in range(int(bootstrap_reps)):
        idx = rng.integers(0, n, size=n)
        cal = fit_isotonic_binary(x_score[idx], y[idx])
        p = predict_isotonic(cal, score_grid)
        if blend_with_logistic:
            logi = _fit_logistic_1d(x_score[idx], y[idx])
            p_log = _predict_logistic_1d(logi, score_grid)
            p_fit = predict_isotonic(cal, x_score[idx])
            iso_unique = len(np.unique(np.round(p_fit, 6)))
            w_iso = 0.35 if iso_unique <= 2 else 0.65
            p = np.clip(w_iso * p + (1.0 - w_iso) * p_log, 0.0, 1.0)
        preds.append(p)
    arr = np.asarray(preds, dtype=float)
    return {
        "x": score_grid.tolist(),
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
    model: Mapping[str, Any],
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
        y = np.asarray([int(r[label_key]) for r in part], dtype=int)
        p = _predict_target(model, part)
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
    p.add_argument("--calibration-version", default="ceii_v2")
    p.add_argument("--tau-gene", type=float, default=None)
    p.add_argument("--tau-site", type=float, default=None)
    p.add_argument("--label-profile", choices=["auto", "v2", "v3"], default="auto")
    p.add_argument("--target-fdr-gene", type=float, default=0.10)
    p.add_argument("--target-fdr-site", type=float, default=0.10)
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--app-min-n-taxa", type=int, default=None)
    p.add_argument("--app-max-n-taxa", type=int, default=None)
    p.add_argument("--app-min-gene-length-nt", type=int, default=None)
    p.add_argument("--app-max-gene-length-nt", type=int, default=None)
    p.add_argument("--app-min-n-branches", type=int, default=None)
    p.add_argument("--app-max-n-branches", type=int, default=None)
    p.add_argument("--near-boundary-fraction", type=float, default=0.08)
    p.add_argument("--min-applicability-score", type=float, default=0.95)
    p.add_argument("--allow-near-boundary-calibration", action="store_true")
    p.add_argument("--write-package-asset", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    calibration_version = str(args.calibration_version)

    rows = _read_tsv(args.metrics_tsv)
    if not rows:
        raise ValueError("No rows in metrics TSV.")

    augmented: List[Dict[str, Any]] = []
    for row in rows:
        extra = compute_truth_aware_metrics(row)
        augmented.append({**row, **extra})

    label_profile = str(args.label_profile)
    if label_profile == "auto":
        label_profile = "v3" if calibration_version.startswith("ceii_v3") else "v2"
    tau_gene = float(args.tau_gene) if args.tau_gene is not None else (0.50 if label_profile == "v3" else 0.42)
    tau_site = float(args.tau_site) if args.tau_site is not None else (0.55 if label_profile == "v3" else 0.45)

    attach_scenario_stability(augmented)
    if label_profile == "v3":
        attach_recoverability_targets(
            augmented,
            tau_gene=tau_gene,
            tau_site=tau_site,
            rank_nan_fallback_gene=0.0,
            rank_nan_fallback_site=0.0,
            gene_weights=(0.55, 0.30, 0.15),
            site_weights=(0.50, 0.35, 0.15),
            use_stability_gating=True,
        )
    else:
        attach_recoverability_targets(
            augmented,
            tau_gene=tau_gene,
            tau_site=tau_site,
        )
    assign_scenario_splits(augmented, seed=int(args.seed))

    # cEII calibration is valid only when the null-dispersion layer was numerically valid.
    # Keep all rows for traceability, but fit/evaluate calibration on sigma-valid rows only.
    for row in augmented:
        sigma0_valid = _safe_bool(row.get("sigma0_valid"))
        sigma0_floored = _safe_bool(row.get("sigma0_floored", row.get("sigma_floor_applied")))
        fallback_applied = _safe_bool(row.get("fallback_applied", row.get("calibration_fallback_flag")))
        row["sigma0_valid_for_calibration"] = int(bool(sigma0_valid and not sigma0_floored and not fallback_applied))
        if row["sigma0_valid_for_calibration"]:
            row["sigma0_exclusion_reason"] = ""
        else:
            reasons: List[str] = []
            if not sigma0_valid:
                reasons.append("sigma0_invalid")
            if sigma0_floored:
                reasons.append("sigma0_floored")
            if fallback_applied:
                reasons.append("fallback_applied")
            row["sigma0_exclusion_reason"] = ",".join(reasons) if reasons else "sigma0_invalid_or_missing"

    _write_tsv(outdir / "recoverability_augmented.tsv", augmented)

    valid_rows = [r for r in augmented if int(r.get("sigma0_valid_for_calibration", 0)) == 1]
    cal_rows = _subset(valid_rows, "calibration")
    if not cal_rows:
        raise RuntimeError("Calibration split is empty after sigma-valid filtering.")

    if calibration_version.startswith("ceii_v3"):
        feature_names = [
            "eii_01_raw",
            "eii_z_clipped",
            "neglog10_q_emp",
            "dispersion_ratio_clipped",
        ]
        feature_design = "stage2_evidence_only_monotone"
    else:
        feature_names = [
            "eii_z_raw",
            "eii_01_raw",
            "log1p_n_taxa",
            "log1p_gene_length_nt",
            "log1p_n_branches",
        ]
        feature_design = "stage2_mixed_linear"
    y_gene_cal = np.asarray([int(r["I_gene"]) for r in cal_rows], dtype=int)
    y_site_cal = np.asarray([int(r["I_site"]) for r in cal_rows], dtype=int)

    if calibration_version.startswith("ceii_v3"):
        gene_model = _fit_monotone_evidence_model(
            cal_rows,
            label_key="I_gene",
            evidence_features=feature_names,
        )
        site_model = _fit_monotone_evidence_model(
            cal_rows,
            label_key="I_site",
            evidence_features=feature_names,
        )
    else:
        gene_model = _fit_linear_score_isotonic(
            cal_rows,
            label_key="I_gene",
            feature_names=feature_names,
        )
        site_model = _fit_linear_score_isotonic(
            cal_rows,
            label_key="I_site",
            feature_names=feature_names,
        )

    score_gene_cal = _predict_linear_score(gene_model, cal_rows)
    score_site_cal = _predict_linear_score(site_model, cal_rows)
    p_gene_cal = _predict_blended_probability(gene_model, score_gene_cal)
    p_site_cal = _predict_blended_probability(site_model, score_site_cal)

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

    score_gene_grid = np.unique(np.asarray(gene_model["isotonic_calibrator"]["x"], dtype=float))
    score_site_grid = np.unique(np.asarray(site_model["isotonic_calibrator"]["x"], dtype=float))
    gene_ci = _bootstrap_ci_calibrator(
        score_gene_cal,
        y_gene_cal,
        score_grid=score_gene_grid,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 101,
        blend_with_logistic=not calibration_version.startswith("ceii_v3"),
    )
    site_ci = _bootstrap_ci_calibrator(
        score_site_cal,
        y_site_cal,
        score_grid=score_site_grid,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 202,
        blend_with_logistic=not calibration_version.startswith("ceii_v3"),
    )

    gene_model["prediction_ci"] = {
        "lower": {"x": gene_ci["x"], "y": gene_ci["lower"]},
        "upper": {"x": gene_ci["x"], "y": gene_ci["upper"]},
    }
    site_model["prediction_ci"] = {
        "lower": {"x": site_ci["x"], "y": site_ci["lower"]},
        "upper": {"x": site_ci["x"], "y": site_ci["upper"]},
    }

    cal_taxa = np.asarray([int(float(r["n_taxa"])) for r in cal_rows], dtype=int)
    cal_len = np.asarray([int(float(r["gene_length_nt"])) for r in cal_rows], dtype=int)
    cal_br = np.asarray([_feature_value(r, "n_branches") for r in cal_rows], dtype=float)
    cal_br = cal_br[np.isfinite(cal_br)]

    app_min_n_taxa = int(np.min(cal_taxa)) if args.app_min_n_taxa is None else int(args.app_min_n_taxa)
    app_max_n_taxa = int(np.max(cal_taxa)) if args.app_max_n_taxa is None else int(args.app_max_n_taxa)
    app_min_gene_len = int(np.min(cal_len)) if args.app_min_gene_length_nt is None else int(args.app_min_gene_length_nt)
    app_max_gene_len = int(np.max(cal_len)) if args.app_max_gene_length_nt is None else int(args.app_max_gene_length_nt)
    app_min_n_branches = (
        int(np.min(cal_br))
        if args.app_min_n_branches is None and cal_br.size > 0
        else int(args.app_min_n_branches if args.app_min_n_branches is not None else 1)
    )
    app_max_n_branches = (
        int(np.max(cal_br))
        if args.app_max_n_branches is None and cal_br.size > 0
        else int(args.app_max_n_branches if args.app_max_n_branches is not None else max(app_min_n_branches, 1))
    )

    regime_groups: Dict[str, Dict[str, Any]] = {}
    for row in cal_rows:
        name = "|".join(
            [
                str(row.get("tree_bin", "unknown")),
                str(row.get("gene_length_bin", "unknown")),
                str(row.get("recombination_bin", "unknown")),
                str(row.get("alignment_noise_bin", "unknown")),
                str(row.get("mutation_rate_heterogeneity_bin", "unknown")),
            ]
        )
        bucket = regime_groups.setdefault(name, {"n_taxa": [], "gene_length_nt": [], "n_branches": []})
        bucket["n_taxa"].append(float(row["n_taxa"]))
        bucket["gene_length_nt"].append(float(row["gene_length_nt"]))
        br = _feature_value(row, "n_branches")
        if np.isfinite(br):
            bucket["n_branches"].append(float(br))

    supported_regimes = []
    for name, vals in sorted(regime_groups.items()):
        supported_regimes.append(
            {
                "name": name,
                "center": {
                    "n_taxa": float(np.median(np.asarray(vals["n_taxa"], dtype=float))),
                    "gene_length_nt": float(np.median(np.asarray(vals["gene_length_nt"], dtype=float))),
                    "n_branches": (
                        float(np.median(np.asarray(vals["n_branches"], dtype=float)))
                        if vals["n_branches"]
                        else float("nan")
                    ),
                },
            }
        )

    target_definitions: Dict[str, str]
    if label_profile == "v3":
        target_definitions = {
            "R_gene": (
                "0.55*branch_rank_norm + 0.30*burden_alignment + "
                "0.15*(scenario_branch_stability * max(branch_rank_norm, burden_alignment)); "
                "NaN branch rank fallback=0.0"
            ),
            "R_site": (
                "0.50*site_enrichment_at_k + 0.35*site_rank_norm + "
                "0.15*(scenario_site_stability * max(site_enrichment_at_k, site_rank_norm)); "
                "NaN site rank fallback=0.0"
            ),
            "I_gene": f"1 if R_gene >= {tau_gene:.2f} else 0",
            "I_site": f"1 if R_site >= {tau_site:.2f} else 0",
        }
    else:
        target_definitions = {
            "R_gene": "0.45*branch_rank_norm + 0.35*burden_alignment + 0.20*scenario_branch_stability",
            "R_site": "0.45*site_enrichment_at_k + 0.35*site_rank_norm + 0.20*scenario_site_stability",
            "I_gene": f"1 if R_gene >= {tau_gene:.2f} else 0",
            "I_site": f"1 if R_site >= {tau_site:.2f} else 0",
        }

    asset = {
        "calibration_version": calibration_version,
        "d_obs_definition": D_OBS_DEFINITION,
        "raw_eii_definition": "eii_z_raw = (D_obs - mu0) / max(sigma0_raw, sigma_floor), eii_01_raw = sigmoid(eii_z_raw)",
        "calibration_semantics": (
            "cEII is a conditional calibrated identifiability probability valid only when "
            "applicability criteria are satisfied; otherwise calibration abstains."
        ),
        "feature_set": {
            "stage1_support_features": ["n_taxa", "gene_length_nt", "n_branches"],
            "stage2_evidence_features": list(feature_names),
            "stage2_design": feature_design,
        },
        "target_definitions": target_definitions,
        "gene_model": gene_model,
        "site_model": site_model,
        # Compatibility fields retained for older consumers; ceii_v2 should use *_model.
        "gene_calibrator": gene_model["isotonic_calibrator"],
        "site_calibrator": site_model["isotonic_calibrator"],
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
            "features": {
                "n_taxa": {"min": app_min_n_taxa, "max": app_max_n_taxa},
                "gene_length_nt": {"min": app_min_gene_len, "max": app_max_gene_len},
                "n_branches": {"min": app_min_n_branches, "max": app_max_n_branches},
            },
            "near_boundary_fraction": float(args.near_boundary_fraction),
            "min_applicability_score_for_calibration": float(args.min_applicability_score),
            "allow_near_boundary_calibration": bool(args.allow_near_boundary_calibration),
            "supported_regimes": supported_regimes,
        },
        "provenance": {
            "metrics_tsv_name": str(Path(args.metrics_tsv).name),
            "bootstrap_reps": int(args.bootstrap_reps),
            "seed": int(args.seed),
            "label_profile": label_profile,
            "tau_gene": tau_gene,
            "tau_site": tau_site,
            "split_counts": {
                "train": int(sum(1 for r in augmented if str(r.get("split")) == "train")),
                "calibration": int(sum(1 for r in augmented if str(r.get("split")) == "calibration")),
                "test": int(sum(1 for r in augmented if str(r.get("split")) == "test")),
                "ood": int(sum(1 for r in augmented if str(r.get("split")) == "ood")),
            },
            "split_counts_sigma_valid": {
                "train": int(sum(1 for r in valid_rows if str(r.get("split")) == "train")),
                "calibration": int(sum(1 for r in valid_rows if str(r.get("split")) == "calibration")),
                "test": int(sum(1 for r in valid_rows if str(r.get("split")) == "test")),
                "ood": int(sum(1 for r in valid_rows if str(r.get("split")) == "ood")),
            },
            "n_rows_total": int(len(augmented)),
            "n_rows_sigma_valid": int(len(valid_rows)),
        },
    }

    asset_path = save_calibration_asset(asset, outdir / "ceii_calibration_asset.json")

    metrics_rows: List[Dict[str, Any]] = []
    metrics_rows.extend(
        _evaluate_target(
            valid_rows,
            gene_model,
            label_key="I_gene",
            target_name="I_gene",
            threshold=float(asset["thresholds"]["gene"]["threshold"]),
        )
    )
    metrics_rows.extend(
        _evaluate_target(
            valid_rows,
            site_model,
            label_key="I_site",
            target_name="I_site",
            threshold=float(asset["thresholds"]["site"]["threshold"]),
        )
    )
    _write_tsv(outdir / "ceii_split_performance.tsv", metrics_rows)

    reliability_rows: List[Dict[str, Any]] = []
    for split in ("calibration", "test", "ood"):
        part = _subset(valid_rows, split)
        if not part:
            continue
        y_gene = np.asarray([int(r["I_gene"]) for r in part], dtype=int)
        y_site = np.asarray([int(r["I_site"]) for r in part], dtype=int)
        p_gene = _predict_target(gene_model, part)
        p_site = _predict_target(site_model, part)
        reliability_rows.extend(_reliability_rows(y_gene, p_gene, split=split, target="I_gene"))
        reliability_rows.extend(_reliability_rows(y_site, p_site, split=split, target="I_site"))
    _write_tsv(outdir / "ceii_reliability.tsv", reliability_rows)

    exclusion_counts: Dict[str, int] = {}
    for row in augmented:
        reason = str(row.get("sigma0_exclusion_reason", "") or "")
        if reason:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    applicability_summary = {
        "n_total_rows": int(len(augmented)),
        "n_sigma_valid_rows": int(len(valid_rows)),
        "fraction_sigma_valid_rows": float(len(valid_rows) / len(augmented)) if augmented else float("nan"),
        "n_abstained_due_sigma_or_fallback": int(len(augmented) - len(valid_rows)),
        "sigma_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "split_counts_sigma_valid": {
            "train": int(sum(1 for r in valid_rows if str(r.get("split")) == "train")),
            "calibration": int(sum(1 for r in valid_rows if str(r.get("split")) == "calibration")),
            "test": int(sum(1 for r in valid_rows if str(r.get("split")) == "test")),
            "ood": int(sum(1 for r in valid_rows if str(r.get("split")) == "ood")),
        },
    }
    (outdir / "ceii_applicability_summary.json").write_text(
        json.dumps(applicability_summary, indent=2) + "\n"
    )

    if args.write_package_asset:
        if calibration_version.startswith("ceii_v3"):
            pkg_name = "ceii_calibration_v3.json"
        elif calibration_version.startswith("ceii_v2"):
            pkg_name = "ceii_calibration_v2.json"
        else:
            pkg_name = "ceii_calibration_v1.json"
        pkg_path = REPO_ROOT / "babappai" / "data" / pkg_name
        save_calibration_asset(asset, pkg_path)
        print(f"Wrote package calibration asset: {pkg_path}")

    summary = {
        "asset_path": str(asset_path),
        "recoverability_augmented_tsv": str(outdir / "recoverability_augmented.tsv"),
        "split_performance_tsv": str(outdir / "ceii_split_performance.tsv"),
        "reliability_tsv": str(outdir / "ceii_reliability.tsv"),
        "applicability_summary_json": str(outdir / "ceii_applicability_summary.json"),
        "calibration_version": calibration_version,
    }
    (outdir / "ceii_calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
