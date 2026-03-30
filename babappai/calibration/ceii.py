"""Empirical calibration of raw EII into calibrated identifiability probabilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


D_OBS_DEFINITION = (
    "Two-sided 5% winsorized sample variance (ddof=1) of site-level mean "
    "branch logits (site_logit_mean) across codon sites for the observed alignment."
)


def default_calibration_asset_path() -> Path:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    v3_2 = data_dir / "ceii_calibration_v3_2.json"
    if v3_2.exists():
        return v3_2
    v3_1 = data_dir / "ceii_calibration_v3_1.json"
    if v3_1.exists():
        return v3_1
    v3 = data_dir / "ceii_calibration_v3.json"
    if v3.exists():
        return v3
    v2 = data_dir / "ceii_calibration_v2.json"
    if v2.exists():
        return v2
    return data_dir / "ceii_calibration_v1.json"


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


def _safe_float(value: Any) -> Optional[float]:
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fv):
        return None
    return float(fv)


def _build_feature_context(
    *,
    eii_z_raw: float,
    n_taxa: int,
    gene_length_nt: int,
    n_branches: Optional[int],
    q_emp: Optional[float],
    dispersion_ratio: Optional[float],
    sigma0_final: Optional[float],
    extra_covariates: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    eii_z = float(eii_z_raw)
    eii_01 = float(1.0 / (1.0 + math.exp(-eii_z)))
    eii_z_clip = float(np.clip(eii_z, -12.0, 12.0))
    qv = _safe_float(q_emp)
    if qv is None:
        qv = 1.0
    qv = float(np.clip(qv, 0.0, 1.0))
    neglog10_q = float(-math.log10(max(qv, 1e-12)))
    dr = _safe_float(dispersion_ratio)
    if dr is None:
        dr = 0.0
    dr = max(float(dr), 0.0)
    dr_clip = float(np.clip(dr, 0.0, 10.0))
    s0 = _safe_float(sigma0_final)
    if s0 is None:
        s0 = 0.0
    s0 = max(float(s0), 0.0)

    ctx: Dict[str, float] = {
        "eii_z_raw": eii_z,
        "eii_01_raw": eii_01,
        "eii_z_clipped": eii_z_clip,
        "q_emp": qv,
        "neglog10_q_emp": neglog10_q,
        "dispersion_ratio": dr,
        "dispersion_ratio_clipped": dr_clip,
        "sigma0_final": s0,
        "sigma0_inverse": float(1.0 / max(s0, 1e-8)),
        "n_taxa": float(n_taxa),
        "gene_length_nt": float(gene_length_nt),
        "log1p_n_taxa": float(math.log1p(max(float(n_taxa), 0.0))),
        "log1p_gene_length_nt": float(math.log1p(max(float(gene_length_nt), 0.0))),
    }
    if n_branches is not None:
        ctx["n_branches"] = float(n_branches)
        ctx["log1p_n_branches"] = float(math.log1p(max(float(n_branches), 0.0)))
    if extra_covariates:
        for key, value in extra_covariates.items():
            fv = _safe_float(value)
            if fv is None:
                continue
            k = str(key)
            ctx[k] = float(fv)
            if k.startswith("log1p_"):
                continue
            if fv >= -0.999999:
                ctx[f"log1p_{k}"] = float(math.log1p(max(fv, 0.0)))
    return ctx


def _predict_linear_score(model: Mapping[str, Any], ctx: Mapping[str, float]) -> float:
    feature_names = [str(x) for x in model.get("feature_names", [])]
    coef = np.asarray(model.get("coef", []), dtype=float)
    means = np.asarray(model.get("feature_mean", []), dtype=float)
    scales = np.asarray(model.get("feature_scale", []), dtype=float)
    intercept = float(model.get("intercept", 0.0))

    if not feature_names or coef.size != len(feature_names):
        raise ValueError("Invalid linear-score model payload.")
    if means.size != len(feature_names) or scales.size != len(feature_names):
        means = np.zeros(len(feature_names), dtype=float)
        scales = np.ones(len(feature_names), dtype=float)

    vals = []
    for i, key in enumerate(feature_names):
        default = float(means[i]) if i < means.size and np.isfinite(means[i]) else 0.0
        v = _safe_float(ctx.get(key, default))
        if v is None:
            v = default
        vals.append(v)
    x = np.asarray(vals, dtype=float)
    scale = np.where(np.abs(scales) > 1e-9, scales, 1.0)
    z = (x - means) / scale
    return float(intercept + np.dot(coef, z))


def _logistic_prob(a: float, b: float, score: float) -> float:
    z = float(np.clip(a * score + b, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(-z)))


def _predict_target_probability(
    *,
    target: str,
    asset: Mapping[str, Any],
    eii_z_raw: float,
    ctx: Mapping[str, float],
) -> Tuple[float, float, float]:
    model_key = f"{target}_model"
    model = asset.get(model_key)
    v1_cal_key = f"{target}_calibrator"

    # ceii_v2 style: linear-score model followed by isotonic calibration.
    if isinstance(model, Mapping) and str(model.get("type", "")) == "linear_score_isotonic":
        score = _predict_linear_score(model, ctx)
        iso = model.get("isotonic_calibrator", {})
        p_iso = float(predict_isotonic(iso, [score])[0])
        logi = model.get("logistic_calibrator")
        if isinstance(logi, Mapping):
            p_log = _logistic_prob(
                float(logi.get("a", 0.0)),
                float(logi.get("b", 0.0)),
                score,
            )
            w_iso = float(np.clip(float(model.get("blend_weight_isotonic", 0.5)), 0.0, 1.0))
            p = float(np.clip(w_iso * p_iso + (1.0 - w_iso) * p_log, 0.0, 1.0))
        else:
            p = p_iso
        ci_payload = model.get("prediction_ci", {})
        if isinstance(ci_payload, Mapping) and ci_payload.get("lower") and ci_payload.get("upper"):
            lo_iso = float(predict_isotonic(ci_payload["lower"], [score])[0])
            hi_iso = float(predict_isotonic(ci_payload["upper"], [score])[0])
            if isinstance(logi, Mapping):
                p_log = _logistic_prob(
                    float(logi.get("a", 0.0)),
                    float(logi.get("b", 0.0)),
                    score,
                )
                w_iso = float(np.clip(float(model.get("blend_weight_isotonic", 0.5)), 0.0, 1.0))
                lo = float(np.clip(w_iso * lo_iso + (1.0 - w_iso) * p_log, 0.0, 1.0))
                hi = float(np.clip(w_iso * hi_iso + (1.0 - w_iso) * p_log, 0.0, 1.0))
            else:
                lo = lo_iso
                hi = hi_iso
        else:
            lo = p
            hi = p
        return p, lo, hi

    # Compatibility path for earlier assets: isotonic directly on eii_z_raw.
    calibrator = asset.get(v1_cal_key)
    if not isinstance(calibrator, Mapping):
        raise ValueError(f"Missing {v1_cal_key} in calibration asset.")
    p = float(predict_isotonic(calibrator, [eii_z_raw])[0])
    ci_root = asset.get("prediction_ci", {})
    if isinstance(ci_root, Mapping) and f"{target}_lower" in ci_root and f"{target}_upper" in ci_root:
        lo = float(predict_isotonic(ci_root[f"{target}_lower"], [eii_z_raw])[0])
        hi = float(predict_isotonic(ci_root[f"{target}_upper"], [eii_z_raw])[0])
    else:
        lo = p
        hi = p
    return p, lo, hi


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


def _coerce_feature_envelope(applicability: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Return required feature envelope with compatibility for earlier assets."""
    if isinstance(applicability.get("features"), Mapping):
        out: Dict[str, Dict[str, float]] = {}
        for key, spec in applicability["features"].items():
            if not isinstance(spec, Mapping):
                continue
            if "min" not in spec or "max" not in spec:
                continue
            out[str(key)] = {"min": float(spec["min"]), "max": float(spec["max"])}
        if out:
            return out

    # Compatibility path for ceii_v1 assets.
    return {
        "n_taxa": {
            "min": float(applicability.get("min_n_taxa", 0)),
            "max": float(applicability.get("max_n_taxa", 10**9)),
        },
        "gene_length_nt": {
            "min": float(applicability.get("min_gene_length_nt", 0)),
            "max": float(applicability.get("max_gene_length_nt", 10**9)),
        },
    }


def _normalized_outside_distance(value: float, lo: float, hi: float) -> float:
    span = max(float(hi) - float(lo), 1.0)
    if value < lo:
        return float((lo - value) / span)
    if value > hi:
        return float((value - hi) / span)
    return 0.0


def _in_domain_boundary_proximity(value: float, lo: float, hi: float) -> float:
    if value < lo or value > hi:
        return 0.0
    span = max(float(hi) - float(lo), 1.0)
    left = float((value - lo) / span)
    right = float((hi - value) / span)
    return float(max(0.0, min(left, right)))


def _nearest_supported_regime(
    *,
    features: Mapping[str, float],
    applicability: Mapping[str, Any],
    envelope: Mapping[str, Mapping[str, float]],
) -> str:
    regimes = applicability.get("supported_regimes", [])
    if isinstance(regimes, Sequence) and regimes:
        best_name = "unknown_regime"
        best_distance = float("inf")
        for row in regimes:
            if not isinstance(row, Mapping):
                continue
            center = row.get("center", {})
            if not isinstance(center, Mapping):
                continue
            dist_terms: List[float] = []
            for key, env in envelope.items():
                if key not in features or key not in center:
                    continue
                span = max(float(env["max"]) - float(env["min"]), 1.0)
                dist_terms.append(((float(features[key]) - float(center[key])) / span) ** 2)
            if not dist_terms:
                continue
            dist = float(math.sqrt(sum(dist_terms)))
            if dist < best_distance:
                best_distance = dist
                best_name = str(row.get("name", best_name))
        return best_name

    ranges = ", ".join(
        f"{k}=[{float(v['min']):g},{float(v['max']):g}]"
        for k, v in envelope.items()
    )
    return f"envelope:{ranges}" if ranges else "envelope:unspecified"


def evaluate_applicability(
    *,
    n_taxa: int,
    gene_length_nt: int,
    asset: Mapping[str, Any],
    extra_covariates: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    applicability = asset.get("applicability", {})
    if not isinstance(applicability, Mapping):
        applicability = {}

    envelope = _coerce_feature_envelope(applicability)
    features: Dict[str, float] = {
        "n_taxa": float(n_taxa),
        "gene_length_nt": float(gene_length_nt),
    }
    if extra_covariates:
        for key, value in extra_covariates.items():
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                features[str(key)] = float(fv)

    missing_required: List[str] = []
    outside: Dict[str, float] = {}
    boundary_proximity: Dict[str, float] = {}

    for key, spec in envelope.items():
        value = features.get(key)
        if value is None or not np.isfinite(value):
            missing_required.append(key)
            continue
        lo = float(spec["min"])
        hi = float(spec["max"])
        d_out = _normalized_outside_distance(float(value), lo, hi)
        outside[key] = float(d_out)
        boundary_proximity[key] = _in_domain_boundary_proximity(float(value), lo, hi)

    near_boundary_fraction = float(applicability.get("near_boundary_fraction", 0.10))
    min_score_for_calibration = float(applicability.get("min_applicability_score_for_calibration", 0.95))
    allow_near_boundary = bool(applicability.get("allow_near_boundary_calibration", False))

    if missing_required:
        status = "out_of_domain"
        within_envelope = False
        distance = float("inf")
        score = 0.0
        reason = "missing_required_applicability_features:" + ",".join(sorted(missing_required))
    else:
        outside_terms = np.asarray(list(outside.values()), dtype=float)
        outside_terms = outside_terms[np.isfinite(outside_terms)]
        distance = float(np.linalg.norm(outside_terms)) if outside_terms.size > 0 else 0.0
        within_envelope = bool(np.all(outside_terms <= 0.0))

        if not within_envelope:
            status = "out_of_domain"
            score = float(np.exp(-2.0 * max(distance, 0.0)))
            violated = [k for k, d in outside.items() if d > 0.0]
            reason = "outside_supported_domain:" + ",".join(sorted(violated))
        else:
            min_boundary_prox = min(boundary_proximity.values()) if boundary_proximity else 1.0
            if min_boundary_prox < near_boundary_fraction:
                status = "near_boundary"
                score = float(np.clip(min_boundary_prox / max(near_boundary_fraction, 1e-6), 0.0, 1.0))
                reason = "near_boundary_of_supported_domain"
            else:
                status = "in_domain"
                score = 1.0
                reason = ""

    should_calibrate = (
        status == "in_domain"
        and score >= min_score_for_calibration
    )
    if status == "near_boundary" and allow_near_boundary and score >= min_score_for_calibration:
        should_calibrate = True

    nearest_regime = _nearest_supported_regime(
        features=features,
        applicability=applicability,
        envelope=envelope,
    )
    return {
        "applicability_score": float(np.clip(score, 0.0, 1.0)),
        "applicability_status": str(status),
        "within_applicability_envelope": bool(within_envelope),
        "distance_to_supported_domain": float(distance) if np.isfinite(distance) else None,
        "nearest_supported_regime": nearest_regime,
        "calibration_unavailable_reason": "" if should_calibrate else str(reason or "outside_or_low_applicability"),
        "should_calibrate": bool(should_calibrate),
        "domain_shift_or_applicability": str(status),
    }


def apply_ceii_calibration(
    *,
    eii_z_raw: float,
    n_taxa: int,
    gene_length_nt: int,
    n_branches: Optional[int] = None,
    q_emp: Optional[float] = None,
    dispersion_ratio: Optional[float] = None,
    sigma0_final: Optional[float] = None,
    extra_covariates: Optional[Mapping[str, float]] = None,
    asset: Mapping[str, Any],
) -> Dict[str, Any]:
    support_covars: Dict[str, float] = {}
    evidence_covars: Dict[str, float] = {}
    reserved_evidence_keys = {
        "q_emp",
        "neglog10_q_emp",
        "dispersion_ratio",
        "dispersion_ratio_clipped",
        "sigma0_final",
        "sigma0_inverse",
        "eii_z_raw",
        "eii_01_raw",
        "eii_z_clipped",
    }
    if extra_covariates:
        for k, v in extra_covariates.items():
            fv = _safe_float(v)
            if fv is not None:
                key = str(k)
                if key in reserved_evidence_keys:
                    evidence_covars[key] = float(fv)
                else:
                    support_covars[key] = float(fv)
    if n_branches is not None and np.isfinite(float(n_branches)):
        support_covars["n_branches"] = float(n_branches)
    qv = _safe_float(q_emp)
    if qv is not None:
        evidence_covars["q_emp"] = float(qv)
        evidence_covars["neglog10_q_emp"] = float(-math.log10(max(min(qv, 1.0), 1e-12)))
    drv = _safe_float(dispersion_ratio)
    if drv is not None:
        evidence_covars["dispersion_ratio"] = float(max(drv, 0.0))
        evidence_covars["dispersion_ratio_clipped"] = float(np.clip(max(drv, 0.0), 0.0, 10.0))
    s0 = _safe_float(sigma0_final)
    if s0 is not None:
        evidence_covars["sigma0_final"] = float(max(s0, 0.0))
        evidence_covars["sigma0_inverse"] = float(1.0 / max(float(s0), 1e-8))

    applicability_meta = evaluate_applicability(
        n_taxa=int(n_taxa),
        gene_length_nt=int(gene_length_nt),
        asset=asset,
        extra_covariates=support_covars,
    )

    thresholds = asset.get("thresholds", {})
    gene_thr = float(thresholds.get("gene", {}).get("threshold", 0.5))
    site_thr = float(thresholds.get("site", {}).get("threshold", 0.5))
    classes = asset.get("classes", {})

    if not bool(applicability_meta["should_calibrate"]):
        return {
            "ceii_gene": None,
            "ceii_site": None,
            "ceii_gene_class": "calibration_unavailable",
            "ceii_site_class": "calibration_unavailable",
            "ceii_gene_identifiable_bool": False,
            "ceii_site_identifiable_bool": False,
            "ceii_ci": {
                "gene": {"lower": None, "upper": None},
                "site": {"lower": None, "upper": None},
            },
            "calibration_version": str(asset.get("calibration_version", "unknown")),
            **{k: v for k, v in applicability_meta.items() if k != "should_calibrate"},
        }

    ctx = _build_feature_context(
        eii_z_raw=float(eii_z_raw),
        n_taxa=int(n_taxa),
        gene_length_nt=int(gene_length_nt),
        n_branches=n_branches,
        q_emp=q_emp,
        dispersion_ratio=dispersion_ratio,
        sigma0_final=sigma0_final,
        extra_covariates={**support_covars, **evidence_covars},
    )
    p_gene, gene_low, gene_high = _predict_target_probability(
        target="gene",
        asset=asset,
        eii_z_raw=float(eii_z_raw),
        ctx=ctx,
    )
    p_site, site_low, site_high = _predict_target_probability(
        target="site",
        asset=asset,
        eii_z_raw=float(eii_z_raw),
        ctx=ctx,
    )

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
        "calibration_version": str(asset.get("calibration_version", "unknown")),
        **{k: v for k, v in applicability_meta.items() if k != "should_calibrate"},
    }


def trace_ceii_calibration(
    *,
    eii_z_raw: float,
    n_taxa: int,
    gene_length_nt: int,
    n_branches: Optional[int] = None,
    q_emp: Optional[float] = None,
    dispersion_ratio: Optional[float] = None,
    sigma0_final: Optional[float] = None,
    extra_covariates: Optional[Mapping[str, float]] = None,
    asset: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return detailed stage-wise cEII internals for debugging/calibration audits."""

    support_covars: Dict[str, float] = {}
    evidence_covars: Dict[str, float] = {}
    reserved_evidence_keys = {
        "q_emp",
        "neglog10_q_emp",
        "dispersion_ratio",
        "dispersion_ratio_clipped",
        "sigma0_final",
        "sigma0_inverse",
        "eii_z_raw",
        "eii_01_raw",
        "eii_z_clipped",
    }
    if extra_covariates:
        for k, v in extra_covariates.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            key = str(k)
            if key in reserved_evidence_keys:
                evidence_covars[key] = float(fv)
            else:
                support_covars[key] = float(fv)
    if n_branches is not None and np.isfinite(float(n_branches)):
        support_covars["n_branches"] = float(n_branches)
    qv = _safe_float(q_emp)
    if qv is not None:
        evidence_covars["q_emp"] = float(qv)
        evidence_covars["neglog10_q_emp"] = float(-math.log10(max(min(qv, 1.0), 1e-12)))
    drv = _safe_float(dispersion_ratio)
    if drv is not None:
        evidence_covars["dispersion_ratio"] = float(max(drv, 0.0))
        evidence_covars["dispersion_ratio_clipped"] = float(np.clip(max(drv, 0.0), 0.0, 10.0))
    s0 = _safe_float(sigma0_final)
    if s0 is not None:
        evidence_covars["sigma0_final"] = float(max(s0, 0.0))
        evidence_covars["sigma0_inverse"] = float(1.0 / max(float(s0), 1e-8))

    applicability_meta = evaluate_applicability(
        n_taxa=int(n_taxa),
        gene_length_nt=int(gene_length_nt),
        asset=asset,
        extra_covariates=support_covars,
    )
    ctx = _build_feature_context(
        eii_z_raw=float(eii_z_raw),
        n_taxa=int(n_taxa),
        gene_length_nt=int(gene_length_nt),
        n_branches=n_branches,
        q_emp=q_emp,
        dispersion_ratio=dispersion_ratio,
        sigma0_final=sigma0_final,
        extra_covariates={**support_covars, **evidence_covars},
    )

    def _target_trace(target: str) -> Dict[str, Any]:
        model_key = f"{target}_model"
        model = asset.get(model_key)
        out: Dict[str, Any] = {
            "target": target,
            "model_type": str(model.get("type", "")) if isinstance(model, Mapping) else "compat_isotonic",
            "feature_names": [],
            "feature_values": {},
            "score": None,
            "isotonic_input": None,
            "isotonic_output": None,
            "logistic_output": None,
            "blended_output": None,
        }

        if isinstance(model, Mapping) and str(model.get("type", "")) == "linear_score_isotonic":
            feature_names = [str(x) for x in model.get("feature_names", [])]
            means = np.asarray(model.get("feature_mean", []), dtype=float)
            scales = np.asarray(model.get("feature_scale", []), dtype=float)
            if means.size != len(feature_names):
                means = np.zeros(len(feature_names), dtype=float)
            if scales.size != len(feature_names):
                scales = np.ones(len(feature_names), dtype=float)
            scales = np.where(np.abs(scales) > 1e-9, scales, 1.0)
            feature_values: Dict[str, float] = {}
            standardized_values: Dict[str, float] = {}
            for i, key in enumerate(feature_names):
                default = float(means[i]) if i < means.size and np.isfinite(means[i]) else 0.0
                raw_v = _safe_float(ctx.get(key, default))
                if raw_v is None:
                    raw_v = default
                feature_values[key] = float(raw_v)
                standardized_values[key] = float((float(raw_v) - float(means[i])) / float(scales[i]))

            score = _predict_linear_score(model, ctx)
            iso_payload = model.get("isotonic_calibrator", {})
            iso_out = (
                float(predict_isotonic(iso_payload, [score])[0])
                if isinstance(iso_payload, Mapping) and iso_payload.get("x") and iso_payload.get("y")
                else None
            )
            logistic_out = None
            blended = iso_out
            logi = model.get("logistic_calibrator")
            if isinstance(logi, Mapping):
                logistic_out = _logistic_prob(
                    float(logi.get("a", 0.0)),
                    float(logi.get("b", 0.0)),
                    score,
                )
                if iso_out is not None:
                    w_iso = float(np.clip(float(model.get("blend_weight_isotonic", 0.5)), 0.0, 1.0))
                    blended = float(np.clip(w_iso * iso_out + (1.0 - w_iso) * logistic_out, 0.0, 1.0))
                else:
                    blended = float(logistic_out)

            out.update(
                {
                    "feature_names": feature_names,
                    "feature_values": feature_values,
                    "feature_values_standardized": standardized_values,
                    "score": float(score),
                    "isotonic_input": float(score),
                    "isotonic_output": iso_out,
                    "logistic_output": float(logistic_out) if logistic_out is not None else None,
                    "blended_output": float(blended) if blended is not None else None,
                    "isotonic_fit_source": model.get("isotonic_fit_source"),
                    "score_design": model.get("score_design"),
                }
            )
            return out

        # Compatibility path: direct isotonic on EII_z.
        p, lo, hi = _predict_target_probability(
            target=target,
            asset=asset,
            eii_z_raw=float(eii_z_raw),
            ctx=ctx,
        )
        out.update(
            {
                "feature_names": ["eii_z_raw"],
                "feature_values": {"eii_z_raw": float(eii_z_raw)},
                "score": float(eii_z_raw),
                "isotonic_input": float(eii_z_raw),
                "isotonic_output": float(p),
                "logistic_output": None,
                "blended_output": float(p),
                "compatibility_prediction_ci": {"lower": float(lo), "upper": float(hi)},
            }
        )
        return out

    gene_trace = _target_trace("gene")
    site_trace = _target_trace("site")
    final = apply_ceii_calibration(
        eii_z_raw=float(eii_z_raw),
        n_taxa=int(n_taxa),
        gene_length_nt=int(gene_length_nt),
        n_branches=n_branches,
        q_emp=q_emp,
        dispersion_ratio=dispersion_ratio,
        sigma0_final=sigma0_final,
        extra_covariates=extra_covariates,
        asset=asset,
    )

    evidence_feature_names = list(
        dict.fromkeys(
            [
                *[str(x) for x in (asset.get("feature_set", {}) or {}).get("stage2_evidence_features", [])],
                *[str(x) for x in gene_trace.get("feature_names", [])],
                *[str(x) for x in site_trace.get("feature_names", [])],
            ]
        )
    )
    evidence_values = {}
    for key in evidence_feature_names:
        fv = _safe_float(ctx.get(key))
        if fv is not None:
            evidence_values[key] = float(fv)

    return {
        "applicability": {
            **{k: v for k, v in applicability_meta.items() if k != "should_calibrate"},
            "should_calibrate": bool(applicability_meta.get("should_calibrate", False)),
            "support_features": {
                "n_taxa": float(n_taxa),
                "gene_length_nt": float(gene_length_nt),
                **{k: float(v) for k, v in support_covars.items()},
            },
        },
        "evidence": {
            "values": evidence_values,
            "raw_context": ctx,
            "target_definition_profile": (
                "v3_2"
                if str(asset.get("calibration_version", "")).startswith("ceii_v3.2")
                else ("v3_1" if str(asset.get("calibration_version", "")).startswith("ceii_v3.1") else "unknown")
            ),
            "target_definitions": asset.get("target_definitions", {}),
        },
        "gene_trace": gene_trace,
        "site_trace": site_trace,
        "final": final,
    }


__all__ = [
    "D_OBS_DEFINITION",
    "apply_ceii_calibration",
    "evaluate_applicability",
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
    "trace_ceii_calibration",
]
