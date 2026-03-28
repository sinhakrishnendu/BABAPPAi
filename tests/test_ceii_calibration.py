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
        "calibration_version": "ceii_unit_v1",
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
            "min_n_taxa": 4,
            "max_n_taxa": 64,
            "min_gene_length_nt": 150,
            "max_gene_length_nt": 3000,
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

    # Ensure asset serialization is stable for packaged calibration data.
    path = save_calibration_asset(asset, tmp_path / "asset.json")
    payload = json.loads(path.read_text())
    assert payload["calibration_version"] == "ceii_unit_v1"
