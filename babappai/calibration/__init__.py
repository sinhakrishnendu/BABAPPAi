"""BABAPPAi calibration utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from babappai.metadata import MODEL_TAG


def _get_data_dir() -> Path:
    # Keep calibration references in the shared package data directory.
    return Path(__file__).resolve().parent.parent / "data"


def _get_reference_path(model_tag: str) -> Path:
    if model_tag in {"legacy_frozen", "frozen"}:
        suffix = "frozen"
    else:
        suffix = model_tag
    return _get_data_dir() / f"neutral_reference_{suffix}.json"


def load_neutral_reference(model_tag: str = MODEL_TAG) -> Dict[str, Dict[str, float]]:
    path = _get_reference_path(model_tag)
    if not path.exists():
        raise FileNotFoundError(f"Neutral calibration file not found: {path}")
    return json.loads(path.read_text())


def _extract_grid(table: Dict[str, Dict[str, float]]) -> Tuple[list[int], list[int]]:
    l_values = set()
    k_values = set()
    for key in table.keys():
        parts = key.split("_")
        l_values.add(int(parts[1]))
        k_values.add(int(parts[3]))
    return sorted(l_values), sorted(k_values)


def lookup_interpolated(
    L: int,
    K: int,
    table: Dict[str, Dict[str, float]],
) -> Optional[Dict[str, float]]:
    l_grid, k_grid = _extract_grid(table)
    if not l_grid or not k_grid:
        return None

    l_low = max((x for x in l_grid if x <= L), default=None)
    l_high = min((x for x in l_grid if x >= L), default=None)
    k_low = max((x for x in k_grid if x <= K), default=None)
    k_high = min((x for x in k_grid if x >= K), default=None)
    if None in (l_low, l_high, k_low, k_high):
        return None

    def get(lv: int, kv: int) -> Optional[Dict[str, float]]:
        return table.get(f"L_{lv}_K_{kv}")

    q11 = get(l_low, k_low)
    q12 = get(l_low, k_high)
    q21 = get(l_high, k_low)
    q22 = get(l_high, k_high)
    if None in (q11, q12, q21, q22):
        return None

    denom = (l_high - l_low) * (k_high - k_low)
    if denom == 0:
        return None

    def bilinear(field: str) -> float:
        return float(
            (
                q11[field] * (l_high - L) * (k_high - K)
                + q21[field] * (L - l_low) * (k_high - K)
                + q12[field] * (l_high - L) * (K - k_low)
                + q22[field] * (L - l_low) * (K - k_low)
            )
            / denom
        )

    return {"sigma2_mean": bilinear("sigma2_mean"), "sigma2_sd": bilinear("sigma2_sd")}


def get_neutral_reference(
    L: int,
    K: int,
    model_tag: str = MODEL_TAG,
) -> Optional[Dict[str, float]]:
    table = load_neutral_reference(model_tag)
    return lookup_interpolated(L, K, table)


__all__ = [
    "_get_reference_path",
    "get_neutral_reference",
    "load_neutral_reference",
    "lookup_interpolated",
]

# cEII calibration helpers are imported lazily for backward compatibility.
try:  # pragma: no cover - import availability is validated in dedicated tests.
    from babappai.calibration.ceii import (
        D_OBS_DEFINITION,
        apply_ceii_calibration,
        default_calibration_asset_path,
        load_calibration_asset,
    )

    __all__.extend(
        [
            "D_OBS_DEFINITION",
            "apply_ceii_calibration",
            "default_calibration_asset_path",
            "load_calibration_asset",
        ]
    )
except Exception:
    # Keep neutral-reference calibration available even if cEII module import fails.
    pass
