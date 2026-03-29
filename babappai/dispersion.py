"""Dispersion statistics for raw EII.

The release-facing primary statistic is ``site_logit_winsorized_variance``:
sample variance (ddof=1) of site-level mean branch logits after two-sided
5% winsorization.

Alternative statistics are exposed for controlled validation/ablation only.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

DISPERSION_METHOD_SITE_LOGIT_VARIANCE = "site_logit_variance"
DISPERSION_METHOD_SITE_LOGIT_WINSORIZED_VARIANCE = "site_logit_winsorized_variance"
DISPERSION_METHOD_SITE_LOGIT_CLIPPED_VARIANCE = "site_logit_clipped_variance"
DISPERSION_METHOD_SITE_LOGIT_MAD_SCALED = "site_logit_mad_scaled"

PRIMARY_DISPERSION_METHOD = DISPERSION_METHOD_SITE_LOGIT_WINSORIZED_VARIANCE

SUPPORTED_DISPERSION_METHODS = (
    DISPERSION_METHOD_SITE_LOGIT_VARIANCE,
    DISPERSION_METHOD_SITE_LOGIT_WINSORIZED_VARIANCE,
    DISPERSION_METHOD_SITE_LOGIT_CLIPPED_VARIANCE,
    DISPERSION_METHOD_SITE_LOGIT_MAD_SCALED,
)


def _as_finite_array(values: Sequence[float] | Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def compute_dispersion(
    site_logit_mean: Sequence[float] | Iterable[float],
    *,
    method: str = PRIMARY_DISPERSION_METHOD,
    winsor_quantile: float = 0.05,
    clip_abs_logit: float = 6.0,
) -> float:
    arr = _as_finite_array(site_logit_mean)
    if arr.size <= 1:
        return 0.0

    mode = str(method)
    if mode not in SUPPORTED_DISPERSION_METHODS:
        raise ValueError(
            "Unsupported dispersion method "
            f"{mode!r}. Supported: {list(SUPPORTED_DISPERSION_METHODS)}"
        )

    if mode == DISPERSION_METHOD_SITE_LOGIT_VARIANCE:
        return float(np.var(arr, ddof=1))

    if mode == DISPERSION_METHOD_SITE_LOGIT_WINSORIZED_VARIANCE:
        q = float(np.clip(winsor_quantile, 0.0, 0.49))
        lo = float(np.quantile(arr, q))
        hi = float(np.quantile(arr, 1.0 - q))
        clipped = np.clip(arr, lo, hi)
        return float(np.var(clipped, ddof=1))

    if mode == DISPERSION_METHOD_SITE_LOGIT_CLIPPED_VARIANCE:
        bound = float(max(clip_abs_logit, 0.1))
        clipped = np.clip(arr, -bound, bound)
        return float(np.var(clipped, ddof=1))

    # DISPERSION_METHOD_SITE_LOGIT_MAD_SCALED
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    # Convert MAD to Gaussian-equivalent sigma, then square to keep variance units.
    sigma_equiv = 1.4826 * mad
    return float(sigma_equiv * sigma_equiv)


__all__ = [
    "DISPERSION_METHOD_SITE_LOGIT_CLIPPED_VARIANCE",
    "DISPERSION_METHOD_SITE_LOGIT_MAD_SCALED",
    "DISPERSION_METHOD_SITE_LOGIT_VARIANCE",
    "DISPERSION_METHOD_SITE_LOGIT_WINSORIZED_VARIANCE",
    "PRIMARY_DISPERSION_METHOD",
    "SUPPORTED_DISPERSION_METHODS",
    "compute_dispersion",
]
