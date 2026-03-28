from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from babappai.calibration.ceii import (
    apply_ceii_calibration,
    binary_metrics,
    derive_threshold,
    expected_calibration_error,
    fit_isotonic_binary,
    predict_isotonic,
    save_calibration_asset,
)


def _toy_asset() -> dict:
    return {
        "calibration_version": "ceii_unit_v2",
        "gene_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.1, 0.5, 0.9]},
        "site_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.05, 0.4, 0.8]},
        "thresholds": {
            "gene": {"threshold": 0.6},
            "site": {"threshold": 0.65},
        },
        "classes": {
            "gene": [
                {"label": "not_identifiable", "min": 0.0, "max": 0.4},
                {"label": "weak_or_ambiguous", "min": 0.4, "max": 0.6},
                {"label": "identifiable", "min": 0.6, "max": 0.8},
                {"label": "strongly_identifiable", "min": 0.8, "max": 1.0},
            ],
            "site": [
                {"label": "not_identifiable", "min": 0.0, "max": 0.4},
                {"label": "weak_or_ambiguous", "min": 0.4, "max": 0.65},
                {"label": "identifiable", "min": 0.65, "max": 0.8},
                {"label": "strongly_identifiable", "min": 0.8, "max": 1.0},
            ],
        },
        "prediction_ci": {
            "gene_lower": {"x": [-2.0, 0.0, 2.0], "y": [0.05, 0.45, 0.85]},
            "gene_upper": {"x": [-2.0, 0.0, 2.0], "y": [0.15, 0.55, 0.95]},
            "site_lower": {"x": [-2.0, 0.0, 2.0], "y": [0.02, 0.35, 0.75]},
            "site_upper": {"x": [-2.0, 0.0, 2.0], "y": [0.08, 0.45, 0.85]},
        },
        "applicability": {
            "features": {
                "n_taxa": {"min": 4, "max": 64},
                "gene_length_nt": {"min": 150, "max": 3000},
            },
            "near_boundary_fraction": 0.08,
            "min_applicability_score_for_calibration": 0.95,
            "allow_near_boundary_calibration": False,
        },
    }


def _toy_linear_asset() -> dict:
    return {
        "calibration_version": "ceii_unit_v2_linear",
        "gene_model": {
            "type": "linear_score_isotonic",
            "feature_names": ["eii_z_raw", "log1p_n_taxa", "log1p_gene_length_nt"],
            "feature_mean": [0.0, 2.5, 7.0],
            "feature_scale": [1.0, 0.5, 0.8],
            "coef": [0.8, 0.2, 0.1],
            "intercept": 0.0,
            "isotonic_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.1, 0.5, 0.9]},
            "logistic_calibrator": {"a": 1.0, "b": 0.0},
            "blend_weight_isotonic": 0.5,
            "prediction_ci": {
                "lower": {"x": [-2.0, 0.0, 2.0], "y": [0.05, 0.40, 0.80]},
                "upper": {"x": [-2.0, 0.0, 2.0], "y": [0.15, 0.60, 0.95]},
            },
        },
        "site_model": {
            "type": "linear_score_isotonic",
            "feature_names": ["eii_z_raw", "log1p_n_taxa", "log1p_gene_length_nt"],
            "feature_mean": [0.0, 2.5, 7.0],
            "feature_scale": [1.0, 0.5, 0.8],
            "coef": [0.5, 0.1, 0.05],
            "intercept": -0.2,
            "isotonic_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.05, 0.35, 0.7]},
            "logistic_calibrator": {"a": 0.8, "b": -0.1},
            "blend_weight_isotonic": 0.4,
            "prediction_ci": {
                "lower": {"x": [-2.0, 0.0, 2.0], "y": [0.02, 0.25, 0.60]},
                "upper": {"x": [-2.0, 0.0, 2.0], "y": [0.10, 0.45, 0.80]},
            },
        },
        "gene_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.1, 0.5, 0.9]},
        "site_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.05, 0.35, 0.7]},
        "thresholds": {
            "gene": {"threshold": 0.6},
            "site": {"threshold": 0.5},
        },
        "classes": {
            "gene": [
                {"label": "not_identifiable", "min": 0.0, "max": 0.4},
                {"label": "weak_or_ambiguous", "min": 0.4, "max": 0.6},
                {"label": "identifiable", "min": 0.6, "max": 0.8},
                {"label": "strongly_identifiable", "min": 0.8, "max": 1.0},
            ],
            "site": [
                {"label": "not_identifiable", "min": 0.0, "max": 0.35},
                {"label": "weak_or_ambiguous", "min": 0.35, "max": 0.5},
                {"label": "identifiable", "min": 0.5, "max": 0.75},
                {"label": "strongly_identifiable", "min": 0.75, "max": 1.0},
            ],
        },
        "applicability": {
            "features": {
                "n_taxa": {"min": 8, "max": 64},
                "gene_length_nt": {"min": 120, "max": 4500},
            },
            "near_boundary_fraction": 0.08,
            "min_applicability_score_for_calibration": 0.95,
            "allow_near_boundary_calibration": False,
        },
    }


def test_isotonic_fit_monotone_predictions() -> None:
    x = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
    y = np.asarray([0, 0, 1, 0, 1], dtype=int)
    cal = fit_isotonic_binary(x, y)
    pred = predict_isotonic(cal, x)
    assert np.all(np.diff(pred) >= -1e-12)
    assert np.all((pred >= 0.0) & (pred <= 1.0))


def test_threshold_metrics_and_ece() -> None:
    y = np.asarray([0, 0, 1, 1, 1, 0], dtype=int)
    p = np.asarray([0.1, 0.2, 0.6, 0.7, 0.8, 0.4], dtype=float)
    thr = derive_threshold(y, p, target_fdr=0.2)
    assert 0.0 <= float(thr["threshold"]) <= 1.0
    bm = binary_metrics(y, p, float(thr["threshold"]))
    assert bm["tp"] + bm["fp"] + bm["tn"] + bm["fn"] == int(y.size)
    ece = expected_calibration_error(y, p, n_bins=5)
    assert 0.0 <= float(ece) <= 1.0


def test_apply_ceii_calibration_fields(tmp_path: Path) -> None:
    asset = _toy_asset()
    out = apply_ceii_calibration(
        eii_z_raw=1.0,
        n_taxa=12,
        gene_length_nt=450,
        asset=asset,
    )
    assert 0.0 <= float(out["ceii_gene"]) <= 1.0
    assert 0.0 <= float(out["ceii_site"]) <= 1.0
    assert out["ceii_gene_class"] in {
        "not_identifiable",
        "weak_or_ambiguous",
        "identifiable",
        "strongly_identifiable",
    }
    assert out["domain_shift_or_applicability"] == "in_domain"
    assert out["applicability_status"] == "in_domain"
    assert bool(out["within_applicability_envelope"]) is True
    assert out["calibration_unavailable_reason"] == ""


def test_apply_ceii_calibration_abstains_out_of_domain(tmp_path: Path) -> None:
    asset = _toy_asset()
    out = apply_ceii_calibration(
        eii_z_raw=1.0,
        n_taxa=128,
        gene_length_nt=7000,
        asset=asset,
    )
    assert out["ceii_gene"] is None
    assert out["ceii_site"] is None
    assert out["ceii_gene_class"] == "calibration_unavailable"
    assert out["ceii_site_class"] == "calibration_unavailable"
    assert out["applicability_status"] == "out_of_domain"
    assert bool(out["within_applicability_envelope"]) is False
    assert isinstance(out["calibration_unavailable_reason"], str) and out["calibration_unavailable_reason"]

    # Ensure asset serialization is stable for packaged calibration data.
    path = save_calibration_asset(asset, tmp_path / "asset.json")
    payload = json.loads(path.read_text())
    assert payload["calibration_version"] == "ceii_unit_v2"


def test_apply_ceii_calibration_linear_score_model() -> None:
    asset = _toy_linear_asset()
    out = apply_ceii_calibration(
        eii_z_raw=0.75,
        n_taxa=20,
        gene_length_nt=1200,
        n_branches=37,
        asset=asset,
    )
    assert out["applicability_status"] == "in_domain"
    assert out["ceii_gene"] is not None
    assert out["ceii_site"] is not None
    assert 0.0 <= float(out["ceii_gene"]) <= 1.0
    assert 0.0 <= float(out["ceii_site"]) <= 1.0
