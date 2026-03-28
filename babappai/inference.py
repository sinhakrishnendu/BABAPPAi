"""Core inference and calibrated identifiability outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from babappai.calibration import (
    D_OBS_DEFINITION,
    apply_ceii_calibration,
    get_neutral_reference,
    load_calibration_asset,
)
from babappai.encoding import encode_alignment
from babappai.identifiability import eii01_from_eiiz
from babappai.metadata import MODEL_DOI, MODEL_FILE_NAME, MODEL_SHA256, MODEL_TAG
from babappai.model_manager import ensure_model, model_status
from babappai.stats import empirical_monte_carlo_pvalue
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
    sigma_floor: float = 0.0,
    alpha: float = 0.05,
    pvalue_mode: str = "empirical_monte_carlo",
    neutral_reps: int = 200,
    min_neutral_group_size: int = 20,
    retain_eii_bands: bool = True,
    report_threshold_bands: bool = True,
    ceii_enabled: bool = True,
    ceii_asset_path: Optional[str] = None,
    _model_override=None,
    _resolved_device_override=None,
) -> Dict[str, object]:
    """Run branch-conditioned inference and return structured outputs."""

    warnings: List[str] = [
        "Diagnostic identifiability score, not evidence of adaptive substitution.",
    ]
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    if sigma_floor < 0:
        raise ValueError("sigma_floor must be >= 0.")
    if neutral_reps < 0:
        raise ValueError("neutral_reps must be >= 0.")
    if min_neutral_group_size < 0:
        raise ValueError("min_neutral_group_size must be >= 0.")
    allowed_modes = {"empirical_monte_carlo", "frozen_reference", "disabled"}
    if pvalue_mode not in allowed_modes:
        raise ValueError(f"Unsupported pvalue_mode: {pvalue_mode}. Allowed: {sorted(allowed_modes)}")

    _set_seed(seed)

    resolved_device = _resolved_device_override if _resolved_device_override is not None else resolve_device(device)
    if _model_override is None:
        model_path = ensure_model(model_tag, offline=offline)
        model = torch.jit.load(str(model_path), map_location=resolved_device)
    else:
        model = _model_override
        model_path = Path(model_status().get("cached_path") or "")
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

    calibration_group = f"L_{L}_K_{K}"
    calibration_source = "disabled_for_neutral_simulation"
    neutral_expected_variance: Optional[float]
    neutral_sd_raw: Optional[float]
    neutral_sd_floored: Optional[float]
    neutral_group_size: Optional[int] = None
    neutral_replicates: List[float] = []
    fallback_flag = False
    fallback_reason: Optional[str] = None
    p_emp_raw = float("nan")

    if pvalue_mode == "disabled":
        neutral_expected_variance = sigma2_obs
        neutral_sd_raw = 0.0
        neutral_group_size = 0
        p_emp_raw = 1.0
    elif pvalue_mode == "empirical_monte_carlo":
        from babappai.tree_calibration import monte_carlo_neutral

        n_empirical = int(neutral_reps if neutral_reps > 0 else N_calibration)
        if n_empirical <= 0:
            raise ValueError("At least one neutral replicate is required for empirical_monte_carlo mode.")
        calibration_source = "tree_conditional_empirical_monte_carlo"
        mu0, sd0, sims = monte_carlo_neutral(
            tree=tree,
            L=L,
            inference_function=run_inference,
            model_tag=model_tag,
            N=n_empirical,
            offline=offline,
            foreground_mode=foreground_mode,
            foreground_list_path=foreground_list_path,
            batch_size=batch_size,
            inference_extra_kwargs={
                "_model_override": model,
                "_resolved_device_override": resolved_device,
                "retain_eii_bands": retain_eii_bands,
                "report_threshold_bands": report_threshold_bands,
            },
        )
        sims_arr = np.asarray(sims, dtype=float)
        neutral_replicates = sims_arr.tolist()
        neutral_expected_variance = float(mu0)
        neutral_sd_raw = float(sd0)
        neutral_group_size = int(sims_arr.size)
        p_emp_raw = empirical_monte_carlo_pvalue(sigma2_obs, neutral_replicates)
    else:
        calibration_source = "frozen_reference_table"
        reference = get_neutral_reference(L=L, K=K, model_tag=model_tag)
        if reference is None:
            neutral_expected_variance = sigma2_obs
            neutral_sd_raw = None
            neutral_group_size = 0
            fallback_flag = True
            fallback_reason = "missing_neutral_reference"
            calibration_source = "fallback_uncalibrated"
            warnings.append(
                "No frozen neutral reference available for this alignment/tree size; "
                "falling back to floor-bounded sigma0 with p_emp unavailable."
            )
        else:
            neutral_expected_variance = float(reference["sigma2_mean"])
            neutral_sd_raw = float(reference["sigma2_sd"])
            if "n_replicates" in reference:
                neutral_group_size = int(reference["n_replicates"])
            elif "n" in reference:
                neutral_group_size = int(reference["n"])
            else:
                neutral_group_size = 0
        warnings.append(
            "pvalue_mode=frozen_reference does not provide empirical Monte Carlo p-values; "
            "use pvalue_mode=empirical_monte_carlo for significance decisions."
        )

    if pvalue_mode != "disabled" and int(neutral_group_size or 0) < int(min_neutral_group_size):
        fallback_flag = True
        if fallback_reason is None:
            fallback_reason = "neutral_group_below_minimum"
        warnings.append(
            "Neutral calibration group size below configured minimum "
            f"({neutral_group_size} < {min_neutral_group_size})."
        )

    if neutral_sd_raw is None or not np.isfinite(neutral_sd_raw) or neutral_sd_raw <= 0:
        if not fallback_flag:
            fallback_flag = True
            fallback_reason = "non_positive_or_invalid_neutral_sd"
        neutral_sd_base = 0.0
    else:
        neutral_sd_base = float(neutral_sd_raw)

    neutral_sd_floored = max(neutral_sd_base, float(sigma_floor))
    if neutral_sd_floored <= 0:
        fallback_flag = True
        if fallback_reason is None:
            fallback_reason = "non_positive_sigma_after_floor"
        neutral_sd_floored = 1e-8

    sigma_floor_applied = bool(sigma_floor > 0 and neutral_sd_floored == float(sigma_floor))

    dispersion_ratio = None
    if neutral_expected_variance and neutral_expected_variance > 0:
        dispersion_ratio = float(sigma2_obs / neutral_expected_variance)

    eii_z_raw = float((sigma2_obs - neutral_expected_variance) / neutral_sd_floored)
    eii_01_raw = float(eii01_from_eiiz(eii_z_raw))
    if np.isfinite(p_emp_raw):
        q_emp = float(p_emp_raw)
    else:
        q_emp = float("nan")
    significant_bool = bool(np.isfinite(q_emp) and q_emp <= float(alpha))
    significance_label = "significant" if significant_bool else "not_significant"

    if fallback_flag and fallback_reason:
        warnings.append(
            "Neutral calibration fallback applied: "
            f"{fallback_reason}; using sigma0={neutral_sd_floored:.6g}."
        )
    ceii_payload: Dict[str, Any] = {
        "ceii_gene": float("nan"),
        "ceii_site": float("nan"),
        "ceii_gene_class": "calibration_unavailable",
        "ceii_site_class": "calibration_unavailable",
        "ceii_gene_identifiable_bool": False,
        "ceii_site_identifiable_bool": False,
        "ceii_ci": {
            "gene": {"lower": float("nan"), "upper": float("nan")},
            "site": {"lower": float("nan"), "upper": float("nan")},
        },
        "domain_shift_or_applicability": "unknown",
        "calibration_version": "unavailable",
    }
    if ceii_enabled:
        try:
            calibration_asset = load_calibration_asset(ceii_asset_path)
            ceii_payload = apply_ceii_calibration(
                eii_z_raw=eii_z_raw,
                n_taxa=ntaxa,
                gene_length_nt=L,
                asset=calibration_asset,
            )
        except Exception as exc:
            warnings.append(
                "cEII calibration unavailable; returning raw EII diagnostics only. "
                f"Reason: {exc}"
            )
    else:
        warnings.append("cEII calibration disabled by caller; returning raw EII diagnostics only.")

    if retain_eii_bands:
        band_extent = str(ceii_payload.get("ceii_gene_class", "calibration_unavailable"))
        band_bool = bool(ceii_payload.get("ceii_gene_identifiable_bool", False))
    else:
        band_extent = "descriptive_band_suppressed"
        band_bool = False

    warnings.append(
        "Raw EII is a dispersion magnitude statistic. Decision-ready identifiability "
        "is represented by cEII outputs calibrated on held-out simulation truth."
    )
    warnings.append(
        "Inferential significance remains separate: use p_emp/q_emp/significant_bool "
        "for matched-neutral exceedance support."
    )
    gene_summary = {
        "D_obs": sigma2_obs,
        "D_obs_definition": D_OBS_DEFINITION,
        "mu0": neutral_expected_variance,
        "sigma0_raw": neutral_sd_raw,
        "sigma0_final": neutral_sd_floored,
        "sigma_floor_used": float(sigma_floor),
        "fallback_flag": fallback_flag,
        "fallback_reason": fallback_reason,
        "neutral_group_size": neutral_group_size,
        "calibration_group": calibration_group,
        "calibration_source": calibration_source,
        "p_emp": p_emp_raw,
        "p_emp_raw": p_emp_raw,
        "q_emp": q_emp,
        "alpha_used": float(alpha),
        "significant_bool": significant_bool,
        "significance_label": significance_label,
        "pvalue_mode": pvalue_mode,
        "neutral_replicates": neutral_replicates,
        "eii_band_descriptive_only": False,
        "retain_eii_bands": bool(retain_eii_bands),
        "report_threshold_bands": bool(report_threshold_bands),
        "observed_variance": sigma2_obs,
        "neutral_expected_variance": neutral_expected_variance,
        "neutral_sd": neutral_sd_floored,
        "neutral_sd_raw": neutral_sd_raw,
        "neutral_sd_floored": neutral_sd_floored,
        "sigma_floor": float(sigma_floor),
        "sigma_floor_applied": sigma_floor_applied,
        "calibration_fallback_flag": fallback_flag,
        "calibration_fallback_reason": fallback_reason,
        "dispersion_ratio": dispersion_ratio,
        "eii_z_raw": eii_z_raw,
        "eii_01_raw": eii_01_raw,
        "EII_z": eii_z_raw,
        "EII_01": eii_01_raw,
        "ceii_gene": float(ceii_payload["ceii_gene"]),
        "ceii_site": float(ceii_payload["ceii_site"]),
        "ceii_gene_class": str(ceii_payload["ceii_gene_class"]),
        "ceii_site_class": str(ceii_payload["ceii_site_class"]),
        "ceii_gene_identifiable_bool": bool(ceii_payload["ceii_gene_identifiable_bool"]),
        "ceii_site_identifiable_bool": bool(ceii_payload["ceii_site_identifiable_bool"]),
        "ceii_ci": ceii_payload["ceii_ci"],
        "domain_shift_or_applicability": str(ceii_payload["domain_shift_or_applicability"]),
        "calibration_version": str(ceii_payload["calibration_version"]),
        "identifiable_bool": band_bool,
        "identifiability_extent": band_extent,
        "identifiable_bool_deprecated": band_bool,
        "identifiability_extent_deprecated": band_extent,
        "model_version": str(model_tag),
        "model_checkpoint_provenance": {
            "model_file_name": MODEL_FILE_NAME,
            "model_doi": MODEL_DOI,
            "model_sha256": MODEL_SHA256,
            "cached_path": str(model_path),
        },
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
        "model_version": str(model_tag),
        "model_checkpoint_provenance": {
            "model_file_name": MODEL_FILE_NAME,
            "model_doi": MODEL_DOI,
            "model_sha256": MODEL_SHA256,
            "cached_path": str(model_path),
        },
        "num_branches": K,
        "num_sites": L,
        "n_sequences": ntaxa,
        "device": str(resolved_device),
        "model_status": {**model_status(), "cached_path": str(model_path), "verified": True},
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
        "neutral_calibration_replicates": neutral_replicates,
        "warnings": warnings,
    }

    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2) + "\n")

    return result
