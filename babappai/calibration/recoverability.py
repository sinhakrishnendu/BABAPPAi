"""Truth-aware recoverability metrics used to calibrate cEII."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 3:
        return float("nan")
    if np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return float("nan")
    rx = _rankdata(xa)
    ry = _rankdata(ya)
    sx = np.std(rx, ddof=1)
    sy = np.std(ry, ddof=1)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _normalize_rank_metric(rho: float, *, neutral_fallback: float = 0.5) -> float:
    if not np.isfinite(rho):
        return float(neutral_fallback)
    return float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0))


def _read_site_scores(path: str | Path) -> np.ndarray:
    vals: List[float] = []
    with Path(path).open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            vals.append(float(row["site_score"]))
    return np.asarray(vals, dtype=float)


def _read_branch_scores(path: str | Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with Path(path).open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[str(row["branch"])] = float(row["background_score"])
    return out


def _load_truth_payload(truth_metadata_path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(truth_metadata_path).read_text())
    return payload


def _load_latent_truth_arrays(truth_metadata_path: str | Path) -> Dict[str, Any]:
    truth_meta = _load_truth_payload(truth_metadata_path)
    latent_npz = truth_meta.get("latent_truth_npz_path")
    if not latent_npz:
        return {}
    npz_path = Path(latent_npz)
    if not npz_path.exists():
        return {}
    data = np.load(npz_path, allow_pickle=True)
    return {
        "site_burden_true": np.asarray(data["site_burden_true"], dtype=float),
        "branch_burden_true": np.asarray(data["branch_burden_true"], dtype=float),
        "branch_names": [str(x) for x in data["branch_names"].tolist()],
        "active_site_indicator": np.asarray(data["active_site_indicator"], dtype=int),
        "active_branch_indicator": np.asarray(data["active_branch_indicator"], dtype=int),
        "latent_cell_fraction_true": float(np.asarray(data["branch_site_matrix_true"], dtype=float).mean()),
    }


def _site_enrichment_at_k(site_score: np.ndarray, active_site_indicator: np.ndarray) -> float:
    if site_score.size == 0 or active_site_indicator.size == 0:
        return float("nan")
    n = min(site_score.size, active_site_indicator.size)
    score = site_score[:n]
    active = active_site_indicator[:n].astype(int)
    n_pos = int(np.sum(active))
    if n_pos <= 0:
        # Neutral case: reward low top-k signal.
        k = max(1, int(math.ceil(0.10 * n)))
        topk = np.sort(score)[-k:]
        return float(np.clip(1.0 - float(np.mean(topk)), 0.0, 1.0))

    k = max(1, n_pos)
    idx = np.argsort(score)[-k:]
    hit = int(np.sum(active[idx] == 1))
    return float(hit / k)


def _burden_alignment_score(pred_gene_burden: float, true_gene_burden: float) -> float:
    err = abs(float(pred_gene_burden) - float(true_gene_burden))
    scale = 0.05 + 0.5 * max(float(true_gene_burden), 0.0)
    return float(np.exp(-err / scale))


def compute_truth_aware_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    site_scores = _read_site_scores(row["site_summary_tsv"])
    branch_scores = _read_branch_scores(row["branch_summary_tsv"])
    truth = _load_latent_truth_arrays(row["truth_metadata_path"])

    if not truth:
        # fallback for older datasets lacking latent arrays
        pred_gene_burden = float(np.mean(site_scores)) if site_scores.size > 0 else float("nan")
        true_gene_burden = float(row.get("latent_cell_fraction_realized", 0.0))
        return {
            "site_enrichment_at_k": float("nan"),
            "site_spearman": float("nan"),
            "branch_spearman": float("nan"),
            "pred_gene_burden": pred_gene_burden,
            "true_gene_burden": true_gene_burden,
            "burden_alignment_score": _burden_alignment_score(pred_gene_burden, true_gene_burden),
        }

    site_true = np.asarray(truth["site_burden_true"], dtype=float)
    branch_true = np.asarray(truth["branch_burden_true"], dtype=float)
    active_site = np.asarray(truth["active_site_indicator"], dtype=int)
    branch_names = truth["branch_names"]

    n_site = min(site_scores.size, site_true.size)
    site_scores = site_scores[:n_site]
    site_true = site_true[:n_site]
    active_site = active_site[:n_site]

    pred_branch = np.asarray([float(branch_scores.get(name, 0.0)) for name in branch_names], dtype=float)
    n_branch = min(pred_branch.size, branch_true.size)
    pred_branch = pred_branch[:n_branch]
    branch_true = branch_true[:n_branch]

    site_srho = spearman_corr(site_scores, site_true)
    branch_srho = spearman_corr(pred_branch, branch_true)

    pred_gene_burden = float(np.mean(site_scores)) if site_scores.size > 0 else float("nan")
    true_gene_burden = float(np.mean(site_true)) if site_true.size > 0 else float("nan")

    return {
        "site_enrichment_at_k": _site_enrichment_at_k(site_scores, active_site),
        "site_spearman": site_srho,
        "branch_spearman": branch_srho,
        "pred_gene_burden": pred_gene_burden,
        "true_gene_burden": true_gene_burden,
        "burden_alignment_score": _burden_alignment_score(pred_gene_burden, true_gene_burden),
    }


def attach_scenario_stability(rows: List[MutableMapping[str, Any]]) -> List[MutableMapping[str, Any]]:
    by_scenario: Dict[str, List[MutableMapping[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)

    for _sid, scenario_rows in by_scenario.items():
        site_vectors = []
        branch_vectors = []
        for row in scenario_rows:
            site_vectors.append(_read_site_scores(row["site_summary_tsv"]))
            branch_scores = _read_branch_scores(row["branch_summary_tsv"])
            branch_names = sorted(branch_scores.keys())
            branch_vectors.append(np.asarray([branch_scores[b] for b in branch_names], dtype=float))

        def _pairwise_mean_corr(vectors: List[np.ndarray]) -> float:
            vals = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    n = min(vectors[i].size, vectors[j].size)
                    if n < 3:
                        continue
                    corr = np.corrcoef(vectors[i][:n], vectors[j][:n])[0, 1]
                    if np.isfinite(corr):
                        vals.append((corr + 1.0) / 2.0)
            if not vals:
                return 1.0
            return float(np.mean(vals))

        site_stability = _pairwise_mean_corr(site_vectors)
        branch_stability = _pairwise_mean_corr(branch_vectors)

        for row in scenario_rows:
            row["scenario_site_stability"] = site_stability
            row["scenario_branch_stability"] = branch_stability
    return rows


def attach_recoverability_targets(
    rows: List[MutableMapping[str, Any]],
    *,
    tau_gene: float = 0.42,
    tau_site: float = 0.45,
) -> List[MutableMapping[str, Any]]:
    for row in rows:
        site_enrich = float(row.get("site_enrichment_at_k", float("nan")))
        site_rank = _normalize_rank_metric(float(row.get("site_spearman", float("nan"))), neutral_fallback=0.5)
        branch_rank = _normalize_rank_metric(float(row.get("branch_spearman", float("nan"))), neutral_fallback=0.5)
        burden_align = float(row.get("burden_alignment_score", float("nan")))
        site_stab = float(row.get("scenario_site_stability", 1.0))
        branch_stab = float(row.get("scenario_branch_stability", 1.0))

        if not np.isfinite(site_enrich):
            site_enrich = 0.5
        if not np.isfinite(burden_align):
            burden_align = 0.5
        if not np.isfinite(site_stab):
            site_stab = 0.5
        if not np.isfinite(branch_stab):
            branch_stab = 0.5

        r_gene = float(np.clip(0.45 * branch_rank + 0.35 * burden_align + 0.20 * branch_stab, 0.0, 1.0))
        r_site = float(np.clip(0.45 * site_enrich + 0.35 * site_rank + 0.20 * site_stab, 0.0, 1.0))

        row["R_gene"] = r_gene
        row["R_site"] = r_site
        row["I_gene"] = int(r_gene >= float(tau_gene))
        row["I_site"] = int(r_site >= float(tau_site))
    return rows


def assign_scenario_splits(
    rows: List[MutableMapping[str, Any]],
    *,
    seed: int = 123,
) -> List[MutableMapping[str, Any]]:
    # OOD: intentionally difficult strata not used for calibration threshold fitting.
    scenario_keys: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row["scenario_id"])
        scenario_keys.setdefault(
            sid,
            {
                "scenario_id": sid,
                "regime": str(row.get("regime", "")),
                "tree_bin": str(row.get("tree_bin", "")),
                "recombination_bin": str(row.get("recombination_bin", "")),
                "alignment_noise_bin": str(row.get("alignment_noise_bin", "")),
                "n_taxa": int(float(row.get("n_taxa", 0) or 0)),
            },
        )

    ood_ids = {
        sid
        for sid, meta in scenario_keys.items()
        if (
            int(meta.get("n_taxa", 0)) >= 24
            and (meta["recombination_bin"] == "high" or meta["alignment_noise_bin"] == "high")
        )
    }

    rng = np.random.default_rng(seed)
    by_regime: Dict[str, List[str]] = {}
    for sid, meta in scenario_keys.items():
        if sid in ood_ids:
            continue
        by_regime.setdefault(meta["regime"], []).append(sid)

    split_map: Dict[str, str] = {}
    for sid in ood_ids:
        split_map[sid] = "ood"

    for _regime, ids in by_regime.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(0.60 * n))
        n_cal = int(round(0.20 * n))
        # ensure non-empty calibration/test when possible
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_cal = max(1, min(n - n_train - 1, n_cal))
        n_test = max(0, n - n_train - n_cal)
        train_ids = ids[:n_train]
        cal_ids = ids[n_train : n_train + n_cal]
        test_ids = ids[n_train + n_cal : n_train + n_cal + n_test]
        for sid in train_ids:
            split_map[sid] = "train"
        for sid in cal_ids:
            split_map[sid] = "calibration"
        for sid in test_ids:
            split_map[sid] = "test"

    for row in rows:
        row["split"] = split_map.get(str(row["scenario_id"]), "test")
    return rows


__all__ = [
    "assign_scenario_splits",
    "attach_recoverability_targets",
    "attach_scenario_stability",
    "compute_truth_aware_metrics",
    "spearman_corr",
]
