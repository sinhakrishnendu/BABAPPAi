"""Statistical helper utilities for empirical significance workflows."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np


def empirical_monte_carlo_pvalue(observed: float, neutral_samples: Sequence[float]) -> float:
    """One-sided empirical Monte Carlo p-value for excess dispersion."""
    arr = np.asarray(list(neutral_samples), dtype=float)
    arr = arr[np.isfinite(arr)]
    m = int(arr.size)
    if m <= 0:
        return 1.0
    exceed = int(np.sum(arr >= float(observed)))
    return float((1 + exceed) / (m + 1))


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment with NaN preservation."""
    arr = np.asarray(p_values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return out

    p = arr[finite_mask]
    order = np.argsort(p)
    ranked = p[order]
    m = int(ranked.size)

    ranked_q = np.empty(m, dtype=float)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * m / rank
        running_min = min(running_min, candidate)
        ranked_q[i] = min(running_min, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = ranked_q
    out[finite_mask] = q
    return out


def annotate_bh_qvalues(
    rows: List[MutableMapping[str, Any]],
    *,
    p_key: str = "p_emp",
    q_key: str = "q_emp",
) -> List[MutableMapping[str, Any]]:
    """In-place BH annotation for dict-like rows with p-values."""
    pvals = []
    for row in rows:
        try:
            pvals.append(float(row.get(p_key, float("nan"))))
        except (TypeError, ValueError):
            pvals.append(float("nan"))
    qvals = bh_adjust(pvals)
    for row, q in zip(rows, qvals.tolist()):
        row[q_key] = q
    return rows


__all__ = [
    "annotate_bh_qvalues",
    "bh_adjust",
    "empirical_monte_carlo_pvalue",
]

