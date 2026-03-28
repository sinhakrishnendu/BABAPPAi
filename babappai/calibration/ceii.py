"""Empirical calibration of raw EII into calibrated identifiability probabilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


D_OBS_DEFINITION = (
    "Sample variance (ddof=1) of site-level mean branch logits (site_logit_mean) "
    "across codon sites for the observed alignment."
)


def default_calibration_asset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "ceii_calibration_v1.json"


def load_calibration_asset(path: Optional[str | Path] = None) -> Dict[str, Any]:
    asset_path = Path(path) if path is not None else default_calibration_asset_path()
    if not asset_path.exists():
        raise FileNotFoundError(f"cEII calibration asset not found: {asset_path}")
    payload = json.loads(asset_path.read_text())
    return payload


def save_calibration_asset(payload: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2) + "\n")
    return out


def _pav_isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic regression on sorted unique x."""
    blocks: List[List[float]] = []
    # block: [start_idx, end_idx, weight, mean]
    for idx, (yi, wi) in enumerate(zip(y.tolist(), w.tolist())):
        blocks.append([float(idx), float(idx), float(wi), float(yi)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            total_w = b1[2] + b2[2]
            mean = (b1[2] * b1[3] + b2[2] * b2[3]) / total_w
            blocks.append([b1[0], b2[1], total_w, mean])

    out = np.empty_like(y, dtype=float)
    for start, end, _weight, mean in blocks:
        out[int(start) : int(end) + 1] = float(mean)
    return out


def fit_isotonic_binary(
    x: Sequence[float],
    y_binary: Sequence[int | float],
    *,
    sample_weight: Optional[Sequence[float]] = None,
) -> Dict[str, List[float]]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y_binary, dtype=float)
    if sample_weight is None:
        w_arr = np.ones_like(y_arr, dtype=float)
    else:
        w_arr = np.asarray(sample_weight, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(w_arr) & (w_arr > 0)
    if not np.any(mask):
        raise ValueError("No valid rows for isotonic fit.")

    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    w_arr = w_arr[mask]

    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    w_sorted = w_arr[order]

    x_unique, inv = np.unique(x_sorted, return_inverse=True)
    sum_w = np.bincount(inv, weights=w_sorted)
    sum_yw = np.bincount(inv, weights=y_sorted * w_sorted)
    y_avg = np.divide(sum_yw, sum_w, out=np.zeros_like(sum_yw), where=sum_w > 0)

    y_iso = _pav_isotonic(y_avg.astype(float), sum_w.astype(float))
    y_iso = np.clip(y_iso, 0.0, 1.0)
    return {"x": x_unique.astype(float).tolist(), "y": y_iso.astype(float).tolist()}


def predict_isotonic(calibrator: Mapping[str, Sequence[float]], x_new: Sequence[float]) -> np.ndarray:
    x = np.asarray(calibrator["x"], dtype=float)
    y = np.asarray(calibrator["y"], dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        raise ValueError("Invalid isotonic calibrator payload.")
    query = np.asarray(x_new, dtype=float)
    return np.interp(query, x, y, left=float(y[0]), right=float(y[-1]))


def brier_score(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    pp = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(pp)
    if not np.any(mask):
        return float("nan")
    yt = yt[mask]
    pp = pp[mask]
    return float(np.mean((pp - yt) ** 2))


def expected_calibration_error(
    y_true: Sequence[int],
    p_pred: Sequence[float],
    *,
    n_bins: int = 15,
) -> float:
    yt = np.asarray(y_true, dtype=float)
    pp = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(pp)
    if not np.any(mask):
        return float("nan")
    yt = yt[mask]
    pp = pp[mask]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(pp.size)
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            bmask = (pp >= left) & (pp <= right)
        else:
            bmask = (pp >= left) & (pp < right)
        if not np.any(bmask):
            continue
        acc = float(np.mean(yt[bmask]))
        conf = float(np.mean(pp[bmask]))
        frac = float(np.sum(bmask)) / n
        ece += frac * abs(acc - conf)
    return float(ece)


def binary_metrics(y_true: Sequence[int], p_pred: Sequence[float], threshold: float) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    pp = np.asarray(p_pred, dtype=float)
    pred = (pp >= float(threshold)).astype(int)
    tp = int(np.sum((pred == 1) & (yt == 1)))
    fp = int(np.sum((pred == 1) & (yt == 0)))
    tn = int(np.sum((pred == 0) & (yt == 0)))
    fn = int(np.sum((pred == 0) & (yt == 1)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    bal = float(np.nanmean([tpr, tnr]))
    fdr = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": tpr,
        "specificity": tnr,
        "fpr": fpr,
        "ppv": precision,
        "npv": npv,
        "fdr": fdr,
        "balanced_accuracy": bal,
    }


def derive_threshold(
    y_true: Sequence[int],
    p_pred: Sequence[float],
    *,
    target_fdr: Optional[float] = None,
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    pp = np.asarray(p_pred, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], pp, [1.0])))
    metric_rows = [binary_metrics(yt, pp, float(t)) for t in candidates.tolist()]

    if target_fdr is not None:
        eligible = [
            row for row in metric_rows if np.isfinite(row["fdr"]) and float(row["fdr"]) <= float(target_fdr)
        ]
        if eligible:
            preferred = [
                row
                for row in eligible
                if (0.0 < float(row["threshold"]) < 1.0) and np.isfinite(row["balanced_accuracy"])
            ]
            pool = preferred if preferred else eligible
            best = max(
                pool,
                key=lambda r: (
                    float(r["balanced_accuracy"]) if np.isfinite(r["balanced_accuracy"]) else -1.0,
                    float(r["ppv"]) if np.isfinite(r["ppv"]) else -1.0,
                    float(r["sensitivity"]) if np.isfinite(r["sensitivity"]) else -1.0,
                    -abs(float(r["threshold"]) - 0.5),
                ),
            )
            return {k: float(v) if isinstance(v, (int, float)) else v for k, v in best.items()}

    # fallback: max balanced accuracy
    clean = [row for row in metric_rows if np.isfinite(row["balanced_accuracy"])]
    if not clean:
        clean = metric_rows
    best = max(clean, key=lambda r: (float(r["balanced_accuracy"]), float(r["ppv"]), -float(r["threshold"])))
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in best.items()}


def class_from_probability(
    probability: float,
    *,
    classes: Sequence[Mapping[str, Any]],
) -> str:
    if not classes:
        return "unclassified"
    p = float(np.clip(probability, 0.0, 1.0))
    for i, row in enumerate(classes):
        lo = float(row["min"])
        hi = float(row["max"])
        label = str(row["label"])
        if i < len(classes) - 1:
            if lo <= p < hi:
                return label
        else:
            if lo <= p <= hi:
                return label
    if p < float(classes[0]["min"]):
        return str(classes[0]["label"])
    return str(classes[-1]["label"])


def apply_ceii_calibration(
    *,
    eii_z_raw: float,
    n_taxa: int,
    gene_length_nt: int,
    asset: Mapping[str, Any],
) -> Dict[str, Any]:
    p_gene = float(predict_isotonic(asset["gene_calibrator"], [eii_z_raw])[0])
    p_site = float(predict_isotonic(asset["site_calibrator"], [eii_z_raw])[0])

    thresholds = asset.get("thresholds", {})
    gene_thr = float(thresholds.get("gene", {}).get("threshold", 0.5))
    site_thr = float(thresholds.get("site", {}).get("threshold", 0.5))
    classes = asset.get("classes", {})

    applicability = asset.get("applicability", {})
    min_taxa = int(applicability.get("min_n_taxa", 0))
    max_taxa = int(applicability.get("max_n_taxa", 10**9))
    min_len = int(applicability.get("min_gene_length_nt", 0))
    max_len = int(applicability.get("max_gene_length_nt", 10**9))
    in_domain = (min_taxa <= int(n_taxa) <= max_taxa) and (min_len <= int(gene_length_nt) <= max_len)

    ci = asset.get("prediction_ci", {})
    if ci:
        gene_low = float(predict_isotonic(ci["gene_lower"], [eii_z_raw])[0])
        gene_high = float(predict_isotonic(ci["gene_upper"], [eii_z_raw])[0])
        site_low = float(predict_isotonic(ci["site_lower"], [eii_z_raw])[0])
        site_high = float(predict_isotonic(ci["site_upper"], [eii_z_raw])[0])
    else:
        gene_low = gene_high = p_gene
        site_low = site_high = p_site

    return {
        "ceii_gene": p_gene,
        "ceii_site": p_site,
        "ceii_gene_class": class_from_probability(
            p_gene,
            classes=classes.get("gene", []),
        ),
        "ceii_site_class": class_from_probability(
            p_site,
            classes=classes.get("site", []),
        ),
        "ceii_gene_identifiable_bool": bool(p_gene >= gene_thr),
        "ceii_site_identifiable_bool": bool(p_site >= site_thr),
        "ceii_ci": {
            "gene": {"lower": gene_low, "upper": gene_high},
            "site": {"lower": site_low, "upper": site_high},
        },
        "domain_shift_or_applicability": "in_domain" if in_domain else "out_of_domain",
        "calibration_version": str(asset.get("calibration_version", "unknown")),
    }


__all__ = [
    "D_OBS_DEFINITION",
    "apply_ceii_calibration",
    "brier_score",
    "binary_metrics",
    "class_from_probability",
    "default_calibration_asset_path",
    "derive_threshold",
    "expected_calibration_error",
    "fit_isotonic_binary",
    "load_calibration_asset",
    "predict_isotonic",
    "save_calibration_asset",
]
