"""Raw-EII descriptive helper utilities.

Primary release-facing identifiability decisions should use calibrated cEII
probabilities. This module remains for compatibility-oriented descriptive output.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


REGIME_NOT_IDENTIFIABLE = "not_identifiable"
REGIME_WEAK_OR_AMBIGUOUS = "weak_or_ambiguous"
REGIME_IDENTIFIABLE = "identifiable"
REGIME_STRONGLY_IDENTIFIABLE = "strongly_identifiable"
EII_BANDS_DESCRIPTIVE_ONLY = True
COMPATIBILITY_FALLBACK_THRESHOLDS = (0.30, 0.70, 0.90)


def eii01_from_eiiz(eii_z: float) -> float:
    """Map raw EII_z into bounded [0, 1] space via a stable sigmoid transform."""
    clipped = max(min(eii_z, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _resolve_thresholds(calibration_asset_path: Optional[str] = None) -> Tuple[Tuple[float, float, float], str]:
    try:
        from babappai.calibration import load_calibration_asset

        asset = load_calibration_asset(calibration_asset_path)
        gene_thr = asset.get("thresholds", {}).get("gene", {})
        weak = float(gene_thr.get("weak_threshold", COMPATIBILITY_FALLBACK_THRESHOLDS[0]))
        main = float(gene_thr.get("threshold", COMPATIBILITY_FALLBACK_THRESHOLDS[1]))
        strong = float(gene_thr.get("strong_threshold", COMPATIBILITY_FALLBACK_THRESHOLDS[2]))
        if not (0.0 <= weak <= main <= strong <= 1.0):
            raise ValueError("invalid threshold ordering in calibration asset")
        return (weak, main, strong), "empirical_calibration_asset"
    except Exception:
        return COMPATIBILITY_FALLBACK_THRESHOLDS, "compatibility_fallback_fixed_thresholds"


def identifiability_extent(
    eii_01: float,
    *,
    thresholds: Tuple[float, float, float] = COMPATIBILITY_FALLBACK_THRESHOLDS,
) -> str:
    weak, main, strong = thresholds
    if 0.0 <= eii_01 < weak:
        return REGIME_NOT_IDENTIFIABLE
    if weak <= eii_01 < main:
        return REGIME_WEAK_OR_AMBIGUOUS
    if main <= eii_01 < strong:
        return REGIME_IDENTIFIABLE
    return REGIME_STRONGLY_IDENTIFIABLE


def identifiability_bool(
    eii_01: float,
    *,
    threshold: float = COMPATIBILITY_FALLBACK_THRESHOLDS[1],
) -> bool:
    """Compatibility descriptive flag retained for downstream continuity."""
    return eii_01 >= float(threshold)


def interpret_identifiability(eii_z: float, *, calibration_asset_path: Optional[str] = None) -> Dict[str, object]:
    eii_01 = eii01_from_eiiz(eii_z)
    thresholds, source = _resolve_thresholds(calibration_asset_path)
    extent = identifiability_extent(eii_01, thresholds=thresholds)
    return {
        "EII_01": eii_01,
        "identifiable_bool": identifiability_bool(eii_01, threshold=thresholds[1]),
        "identifiability_extent": extent,
        "eii_band_descriptive_only": EII_BANDS_DESCRIPTIVE_ONLY,
        "threshold_source": source,
        "threshold_values": {
            "weak": thresholds[0],
            "main": thresholds[1],
            "strong": thresholds[2],
        },
    }
