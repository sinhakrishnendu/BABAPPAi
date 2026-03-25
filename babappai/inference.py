"""Core inference and calibrated identifiability outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from babappai.calibration import get_neutral_reference
from babappai.encoding import encode_alignment
from babappai.identifiability import interpret_identifiability
from babappai.metadata import MODEL_TAG
from babappai.model_manager import ensure_model, model_status
from babappai.tree import enumerate_branches, load_tree
from babappai.utils import resolve_device


def _is_terminal_branch(name: str) -> bool:
    return not name.startswith("internal_")


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_foreground_list(path: Path) -> List[str]:
    entries: List[str] = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(value)
    return entries


def _select_foregrounds(
    branches: Sequence[str],
    foreground_mode: str,
    foreground_list_path: Optional[str],
) -> Set[str]:
    branch_set = set(branches)

    if foreground_mode == "all-leaves":
        return {b for b in branches if _is_terminal_branch(b)}

    if foreground_mode == "all-branches":
        return set(branches)

    if foreground_mode != "user-list":
        raise ValueError(f"Unsupported foreground mode: {foreground_mode}")

    if not foreground_list_path:
        raise ValueError(
            "foreground_mode=user-list requires --foreground-list to be set."
        )

    selected = _load_foreground_list(Path(foreground_list_path))
    if not selected:
        raise ValueError("Foreground list is empty.")

    unknown = sorted({item for item in selected if item not in branch_set})
    if unknown:
        raise ValueError(
            "Foreground list contains unknown branch IDs: " + ", ".join(unknown)
        )

    return set(selected)


def _extract_site_statistics(site_logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    squeezed = site_logits.squeeze(0)

    if squeezed.ndim == 1:
        site_logit_mean = squeezed
        site_logit_var = torch.zeros_like(squeezed)
        site_scores = torch.sigmoid(squeezed)
    elif squeezed.ndim == 2:
        site_logit_mean = squeezed.mean(dim=0)
        site_logit_var = squeezed.var(dim=0, unbiased=False)
        site_scores = torch.sigmoid(squeezed).mean(dim=0)
    else:
        raise ValueError(
            f"Unexpected site_logits dimensionality: {tuple(squeezed.shape)}"
        )

    return (
        site_scores.detach().cpu().numpy(),
        site_logit_mean.detach().cpu().numpy(),
        site_logit_var.detach().cpu().numpy(),
    )


def _extract_branch_logits(branch_logits: torch.Tensor, n_branches: int) -> np.ndarray:
    squeezed = branch_logits.squeeze(0)

    if squeezed.ndim == 1:
        branch_values = squeezed
    elif squeezed.ndim == 2:
        if squeezed.shape[0] == n_branches:
            branch_values = squeezed.mean(dim=1)
        elif squeezed.shape[1] == n_branches:
            branch_values = squeezed.mean(dim=0)
        else:
            branch_values = squeezed.reshape(-1)
    else:
        branch_values = squeezed.reshape(-1)

    if branch_values.numel() != n_branches:
        raise ValueError(
            "Model branch output size mismatch: "
            f"expected {n_branches}, got {branch_values.numel()}"
        )

    return branch_values.detach().cpu().numpy()


def run_inference(
    alignment_path: str,
    tree_path: Optional[str] = None,
    tree_obj=None,
    out_path: Optional[str] = None,
    device: str = "auto",
    model_tag: str = MODEL_TAG,
    tree_calibration: bool = False,
    N_calibration: int = 200,
    offline: bool = False,
    seed: Optional[int] = None,
    foreground_mode: str = "all-leaves",
    foreground_list_path: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, object]:
    """Run branch-conditioned inference and return structured outputs."""

    warnings: List[str] = [
        "Diagnostic identifiability score, not evidence of adaptive substitution.",
    ]

    _set_seed(seed)

    resolved_device = resolve_device(device)
    model_path = ensure_model(model_tag, offline=offline)
    model = torch.jit.load(str(model_path), map_location=resolved_device)
    model.eval()

    X, ntaxa, L = encode_alignment(alignment_path)

    tree = tree_obj if tree_obj is not None else load_tree(tree_path)
    branches = enumerate_branches(tree)
    K = len(branches)

    selected_foregrounds = _select_foregrounds(
        branches=branches,
        foreground_mode=foreground_mode,
        foreground_list_path=foreground_list_path,
    )

    taxon_names = [leaf.name for leaf in tree.get_leaves()]
    taxon_to_idx = {name: idx for idx, name in enumerate(taxon_names)}

    branch_to_taxa: List[List[int]] = []
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue
        descendants = [
            taxon_to_idx[leaf.name]
            for leaf in node.get_leaves()
            if leaf.name in taxon_to_idx
        ]
        branch_to_taxa.append(descendants)

    parent = torch.zeros((1, K, L), dtype=torch.long, device=resolved_device)
    child = torch.zeros((1, K, L), dtype=torch.long, device=resolved_device)

    for i in range(L):
        col = X[:, i].tolist()
        consensus = Counter(col).most_common(1)[0][0]

        for b, taxa in enumerate(branch_to_taxa):
            if not taxa:
                continue
            mismatches = sum(X[t, i] != consensus for t in taxa)
            parent[0, b, i] = len(taxa) - mismatches
            child[0, b, i] = mismatches

    branch_length = torch.ones((1, K), dtype=torch.float32, device=resolved_device)

    with torch.no_grad():
        site_logits, branch_logits = model(parent, child, branch_length)

    site_scores, site_logit_mean, site_logit_var = _extract_site_statistics(site_logits)
    branch_logit_mean = _extract_branch_logits(branch_logits, K)
    branch_background = 1.0 / (1.0 + np.exp(-np.clip(branch_logit_mean, -60.0, 60.0)))

    sigma2_obs = float(np.var(site_logit_mean, ddof=1)) if len(site_logit_mean) > 1 else 0.0

    if batch_size != 1:
        warnings.append(
            "batch_size is currently accepted for CLI compatibility but not used by "
            "the frozen model runner."
        )

    calibration_source = "tree_conditional_monte_carlo" if tree_calibration else "frozen_reference_table"
    neutral_expected_variance: Optional[float]
    neutral_sd: Optional[float]
    empirical_p: Optional[float] = None

    if tree_calibration:
        from babappai.tree_calibration import monte_carlo_neutral

        mu0, sd0, sims = monte_carlo_neutral(
            tree=tree,
            L=L,
            inference_function=run_inference,
            model_tag=model_tag,
            N=N_calibration,
            offline=offline,
            foreground_mode=foreground_mode,
            foreground_list_path=foreground_list_path,
            batch_size=batch_size,
        )
        neutral_expected_variance = float(mu0)
        neutral_sd = float(sd0)
        empirical_p = float((1 + np.sum(sims >= sigma2_obs)) / (len(sims) + 1))
    else:
        reference = get_neutral_reference(L=L, K=K, model_tag=model_tag)
        if reference is None:
            neutral_expected_variance = sigma2_obs
            neutral_sd = 1.0
            calibration_source = "fallback_uncalibrated"
            warnings.append(
                "No neutral reference available for this alignment/tree size; "
                "using deterministic fallback calibration (EII_z=0 baseline)."
            )
        else:
            neutral_expected_variance = float(reference["sigma2_mean"])
            neutral_sd = float(reference["sigma2_sd"])

    dispersion_ratio = None
    if neutral_expected_variance and neutral_expected_variance > 0:
        dispersion_ratio = float(sigma2_obs / neutral_expected_variance)

    if neutral_sd is None or neutral_sd <= 0:
        eii_z = 0.0
        warnings.append("Neutral variance spread was non-positive; EII_z set to 0.")
    else:
        eii_z = float((sigma2_obs - neutral_expected_variance) / neutral_sd)

    ident = interpret_identifiability(eii_z)
    gene_summary = {
        "observed_variance": sigma2_obs,
        "neutral_expected_variance": neutral_expected_variance,
        "dispersion_ratio": dispersion_ratio,
        "EII_z": eii_z,
        "EII_01": float(ident["EII_01"]),
        "identifiable_bool": bool(ident["identifiable_bool"]),
        "identifiability_extent": ident["identifiability_extent"],
        "empirical_p": empirical_p,
        "calibration_source": calibration_source,
    }

    branch_results = []
    for idx, branch in enumerate(branches):
        branch_results.append(
            {
                "branch": branch,
                "background_score": float(branch_background[idx]),
                "logit_mean": float(branch_logit_mean[idx]),
                "is_terminal": _is_terminal_branch(branch),
                "selected_foreground": branch in selected_foregrounds,
            }
        )

    site_results = []
    for idx in range(L):
        site_results.append(
            {
                "site": idx + 1,
                "site_score": float(site_scores[idx]),
                "site_logit_mean": float(site_logit_mean[idx]),
                "site_logit_var": float(site_logit_var[idx]),
            }
        )

    result = {
        "model": "BABAPPAiLegacyFrozenModel",
        "model_tag": model_tag,
        "num_branches": K,
        "num_sites": L,
        "n_sequences": ntaxa,
        "device": str(resolved_device),
        "model_status": {
            **model_status(),
            "cached_path": str(model_path),
            "verified": True,
        },
        "site_scores": site_scores.tolist(),
        "site_logit_mean": site_logit_mean.tolist(),
        "site_logit_var": site_logit_var.tolist(),
        "branch_background": {
            branch: float(score)
            for branch, score in zip(branches, branch_background.tolist())
        },
        "branch_logit_mean": {
            branch: float(score)
            for branch, score in zip(branches, branch_logit_mean.tolist())
        },
        "gene_level_identifiability": gene_summary,
        "branch_results": branch_results,
        "site_results": site_results,
        "foreground": {
            "mode": foreground_mode,
            "selected_branches": sorted(selected_foregrounds),
        },
        "warnings": warnings,
    }

    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2) + "\n")

    return result
