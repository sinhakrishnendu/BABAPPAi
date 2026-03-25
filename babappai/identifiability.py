"""Identifiability score transforms and interpretation policy."""

from __future__ import annotations

import math
from typing import Dict


REGIME_NOT_IDENTIFIABLE = "not_identifiable"
REGIME_WEAK_OR_AMBIGUOUS = "weak_or_ambiguous"
REGIME_IDENTIFIABLE = "identifiable"
REGIME_STRONGLY_IDENTIFIABLE = "strongly_identifiable"


def eii01_from_eiiz(eii_z: float) -> float:
    """Map raw EII_z into bounded [0, 1] space via a stable sigmoid transform."""
    clipped = max(min(eii_z, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def identifiability_extent(eii_01: float) -> str:
    if 0.0 <= eii_01 < 0.30:
        return REGIME_NOT_IDENTIFIABLE
    if 0.30 <= eii_01 < 0.70:
        return REGIME_WEAK_OR_AMBIGUOUS
    if 0.70 <= eii_01 < 0.90:
        return REGIME_IDENTIFIABLE
    return REGIME_STRONGLY_IDENTIFIABLE


def identifiability_bool(eii_01: float) -> bool:
    return eii_01 >= 0.70


def interpret_identifiability(eii_z: float) -> Dict[str, object]:
    eii_01 = eii01_from_eiiz(eii_z)
    extent = identifiability_extent(eii_01)
    return {
        "EII_01": eii_01,
        "identifiable_bool": identifiability_bool(eii_01),
        "identifiability_extent": extent,
    }
