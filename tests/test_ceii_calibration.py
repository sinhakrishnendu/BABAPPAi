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
    load_calibration_asset,
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


def _toy_v3_evidence_asset() -> dict:
    return {
        "calibration_version": "ceii_v3_unit",
        "gene_model": {
            "type": "linear_score_isotonic",
            "score_design": "monotone_evidence_nonnegative",
            "evidence_features_only": True,
            "feature_names": ["eii_01_raw", "eii_z_clipped", "neglog10_q_emp", "dispersion_ratio_clipped"],
            "feature_mean": [0.5, 0.0, 0.5, 1.0],
            "feature_scale": [0.25, 2.0, 1.0, 1.0],
            "coef": [0.6, 0.4, 0.8, 0.2],
            "intercept": -0.1,
            "isotonic_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.05, 0.5, 0.95]},
            "prediction_ci": {
                "lower": {"x": [-2.0, 0.0, 2.0], "y": [0.02, 0.40, 0.85]},
                "upper": {"x": [-2.0, 0.0, 2.0], "y": [0.10, 0.60, 0.98]},
            },
        },
        "site_model": {
            "type": "linear_score_isotonic",
            "score_design": "monotone_evidence_nonnegative",
            "evidence_features_only": True,
            "feature_names": ["eii_01_raw", "eii_z_clipped", "neglog10_q_emp", "dispersion_ratio_clipped"],
            "feature_mean": [0.5, 0.0, 0.5, 1.0],
            "feature_scale": [0.25, 2.0, 1.0, 1.0],
            "coef": [0.5, 0.3, 0.7, 0.2],
            "intercept": -0.2,
            "isotonic_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.03, 0.4, 0.9]},
            "prediction_ci": {
                "lower": {"x": [-2.0, 0.0, 2.0], "y": [0.01, 0.30, 0.80]},
                "upper": {"x": [-2.0, 0.0, 2.0], "y": [0.08, 0.50, 0.95]},
            },
        },
        "gene_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.05, 0.5, 0.95]},
        "site_calibrator": {"x": [-2.0, 0.0, 2.0], "y": [0.03, 0.4, 0.9]},
        "thresholds": {"gene": {"threshold": 0.7}, "site": {"threshold": 0.6}},
        "classes": {
            "gene": [
                {"label": "not_identifiable", "min": 0.0, "max": 0.4},
                {"label": "weak_or_ambiguous", "min": 0.4, "max": 0.7},
                {"label": "identifiable", "min": 0.7, "max": 0.9},
                {"label": "strongly_identifiable", "min": 0.9, "max": 1.0},
            ],
            "site": [
                {"label": "not_identifiable", "min": 0.0, "max": 0.35},
                {"label": "weak_or_ambiguous", "min": 0.35, "max": 0.6},
                {"label": "identifiable", "min": 0.6, "max": 0.85},
                {"label": "strongly_identifiable", "min": 0.85, "max": 1.0},
            ],
        },
        "applicability": {
            "features": {
                "n_taxa": {"min": 8, "max": 64},
                "gene_length_nt": {"min": 120, "max": 4500},
                "n_branches": {"min": 13, "max": 125},
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


def test_packaged_ceii_v2_asset_abstains_out_of_domain() -> None:
    asset = load_calibration_asset()
    assert str(asset.get("calibration_version", "")).startswith("ceii_v")
    out = apply_ceii_calibration(
        eii_z_raw=0.5,
        n_taxa=512,
        gene_length_nt=9000,
        asset=asset,
    )
    assert out["ceii_gene"] is None
    assert out["ceii_site"] is None
    assert out["ceii_gene_class"] == "calibration_unavailable"
    assert out["ceii_site_class"] == "calibration_unavailable"
    assert str(out["applicability_status"]) in {"out_of_domain", "near_boundary"}


def test_ceii_v3_evidence_only_does_not_use_support_covariates_in_stage2() -> None:
    asset = _toy_v3_evidence_asset()
    base = apply_ceii_calibration(
        eii_z_raw=1.2,
        n_taxa=16,
        gene_length_nt=900,
        n_branches=29,
        q_emp=0.01,
        dispersion_ratio=2.0,
        sigma0_final=0.2,
        asset=asset,
    )
    shifted_support = apply_ceii_calibration(
        eii_z_raw=1.2,
        n_taxa=48,
        gene_length_nt=3600,
        n_branches=95,
        q_emp=0.01,
        dispersion_ratio=2.0,
        sigma0_final=0.2,
        asset=asset,
    )
    assert base["applicability_status"] == "in_domain"
    assert shifted_support["applicability_status"] == "in_domain"
    assert abs(float(base["ceii_gene"]) - float(shifted_support["ceii_gene"])) < 1e-10
    assert abs(float(base["ceii_site"]) - float(shifted_support["ceii_site"])) < 1e-10


def test_ceii_v3_probability_is_monotone_with_evidence() -> None:
    asset = _toy_v3_evidence_asset()
    weak = apply_ceii_calibration(
        eii_z_raw=-0.5,
        n_taxa=24,
        gene_length_nt=1200,
        n_branches=45,
        q_emp=1.0,
        dispersion_ratio=0.7,
        sigma0_final=0.2,
        asset=asset,
    )
    strong = apply_ceii_calibration(
        eii_z_raw=2.0,
        n_taxa=24,
        gene_length_nt=1200,
        n_branches=45,
        q_emp=1e-4,
        dispersion_ratio=3.5,
        sigma0_final=0.2,
        asset=asset,
    )
    assert float(strong["ceii_gene"]) >= float(weak["ceii_gene"])
    assert float(strong["ceii_site"]) >= float(weak["ceii_site"])
