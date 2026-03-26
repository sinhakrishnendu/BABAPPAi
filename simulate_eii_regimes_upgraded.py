#!/usr/bin/env python3
"""
simulate_eii_regimes_upgraded.py

Upgraded synthetic calibration simulator for BABAPPAi EII thresholds.

What this version adds beyond the original simulator:
1. Neutral-quantile-derived thresholds (EII_z and EII_01).
2. Threshold-performance tables across a dense threshold grid.
3. ROC-style summaries and AUC for user-defined positive classes.
4. Per-threshold confusion summaries at key thresholds.
5. Plots for EII distributions, ROC curve, and threshold trade-offs.

The purpose is to empirically calibrate binary decision thresholds rather than
asserting heuristic cutoffs such as EII_01 >= 0.70.

This remains a score-space simulator, not a full sequence simulator.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def safe_var(x: np.ndarray, ddof: int = 1) -> float:
    if x.size <= ddof:
        return 0.0
    return float(np.var(x, ddof=ddof))


def safe_std(x: np.ndarray, ddof: int = 1) -> float:
    if x.size <= ddof:
        return 0.0
    return float(np.std(x, ddof=ddof))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Regime specification
# -----------------------------------------------------------------------------

@dataclass
class RegimeSpec:
    name: str
    burden_min: float
    burden_max: float
    effect_min: float
    effect_max: float
    branch_frac_min: float
    branch_frac_max: float


REGIMES: Dict[str, RegimeSpec] = {
    "neutral": RegimeSpec(
        name="neutral",
        burden_min=0.00,
        burden_max=0.00,
        effect_min=0.00,
        effect_max=0.00,
        branch_frac_min=0.00,
        branch_frac_max=0.00,
    ),
    "low": RegimeSpec(
        name="low",
        burden_min=0.01,
        burden_max=0.05,
        effect_min=0.25,
        effect_max=0.60,
        branch_frac_min=0.10,
        branch_frac_max=0.25,
    ),
    "medium": RegimeSpec(
        name="medium",
        burden_min=0.06,
        burden_max=0.15,
        effect_min=0.60,
        effect_max=1.10,
        branch_frac_min=0.20,
        branch_frac_max=0.40,
    ),
    "high": RegimeSpec(
        name="high",
        burden_min=0.16,
        burden_max=0.35,
        effect_min=1.10,
        effect_max=1.80,
        branch_frac_min=0.30,
        branch_frac_max=0.60,
    ),
}


@dataclass
class SimConfig:
    n_branches: int = 12
    gene_length_min: int = 200
    gene_length_max: int = 500
    neutral_reps: int = 200
    measurement_noise_sd: float = 0.80
    branch_random_sd: float = 0.50
    site_random_sd: float = 0.40
    baseline_logit_mean: float = -2.2
    baseline_logit_sd: float = 0.20
    sigma_floor: float = 0.05
    random_seed: int = 123


# -----------------------------------------------------------------------------
# Simulation core
# -----------------------------------------------------------------------------

def sample_regime_parameters(spec: RegimeSpec, rng: np.random.Generator) -> Tuple[float, float, float]:
    burden = rng.uniform(spec.burden_min, spec.burden_max)
    effect = rng.uniform(spec.effect_min, spec.effect_max)
    branch_frac = rng.uniform(spec.branch_frac_min, spec.branch_frac_max)
    return burden, effect, branch_frac


def choose_perturbed_structure(
    L: int,
    B: int,
    burden: float,
    branch_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    Y = np.zeros((L, B), dtype=np.int8)
    if burden <= 0.0 or branch_frac <= 0.0:
        return Y

    n_branches_active = max(1, int(round(branch_frac * B)))
    active_branches = rng.choice(B, size=n_branches_active, replace=False)

    target_cells = max(1, int(round(burden * L * B)))
    n_sites = max(1, int(math.ceil(target_cells / n_branches_active)))
    n_sites = min(n_sites, L)
    active_sites = rng.choice(L, size=n_sites, replace=False)

    Y[np.ix_(active_sites, active_branches)] = 1

    current_cells = int(Y.sum())
    if current_cells > target_cells:
        flat_active = np.flatnonzero(Y.ravel() == 1)
        keep = rng.choice(flat_active, size=target_cells, replace=False)
        Y[:] = 0
        Y.ravel()[keep] = 1

    return Y


def generate_logits(
    Y: np.ndarray,
    effect_size: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    L, B = Y.shape
    baseline = rng.normal(cfg.baseline_logit_mean, cfg.baseline_logit_sd)
    site_eff = rng.normal(0.0, cfg.site_random_sd, size=L)[:, None]
    branch_eff = rng.normal(0.0, cfg.branch_random_sd, size=B)[None, :]
    noise = rng.normal(0.0, cfg.measurement_noise_sd, size=(L, B))
    logits = baseline + site_eff + branch_eff + noise + effect_size * Y
    return logits


def site_level_scores(logits: np.ndarray) -> np.ndarray:
    return sigmoid(logits).mean(axis=1)


def branch_level_scores(logits: np.ndarray) -> np.ndarray:
    return sigmoid(logits).mean(axis=0)


def dispersion_statistic(logits: np.ndarray, mode: str) -> float:
    if mode == "site_score_variance":
        return safe_var(site_level_scores(logits), ddof=1)
    if mode == "site_logit_variance":
        return safe_var(logits.mean(axis=1), ddof=1)
    if mode == "full_logit_variance":
        return safe_var(logits.ravel(), ddof=1)
    raise ValueError(f"Unknown dispersion mode: {mode}")


def matched_neutral_reference(
    L: int,
    B: int,
    cfg: SimConfig,
    rng: np.random.Generator,
    mode: str,
) -> Tuple[float, float]:
    Y0 = np.zeros((L, B), dtype=np.int8)
    dvals: List[float] = []
    for _ in range(cfg.neutral_reps):
        logits0 = generate_logits(Y0, effect_size=0.0, cfg=cfg, rng=rng)
        dvals.append(dispersion_statistic(logits0, mode=mode))
    mu0 = float(np.mean(dvals))
    sigma0 = max(safe_std(np.asarray(dvals), ddof=1), cfg.sigma_floor)
    return mu0, sigma0


def categorize_eii01(eii01: float, weak_cut: float, strong_cut: float) -> str:
    if eii01 < weak_cut:
        return "weak_or_ambiguous"
    if eii01 < strong_cut:
        return "identifiable"
    return "strongly_identifiable"


def simulate_one_gene(
    regime_name: str,
    cfg: SimConfig,
    rng: np.random.Generator,
    mode: str,
    weak_cut: float,
    strong_cut: float,
) -> Dict[str, float]:
    spec = REGIMES[regime_name]
    L = int(rng.integers(cfg.gene_length_min, cfg.gene_length_max + 1))
    B = cfg.n_branches

    burden, effect_size, branch_frac = sample_regime_parameters(spec, rng)
    Y = choose_perturbed_structure(L=L, B=B, burden=burden, branch_frac=branch_frac, rng=rng)
    logits = generate_logits(Y=Y, effect_size=effect_size, cfg=cfg, rng=rng)

    d_obs = dispersion_statistic(logits, mode=mode)
    mu0, sigma0 = matched_neutral_reference(L=L, B=B, cfg=cfg, rng=rng, mode=mode)
    eii_z = (d_obs - mu0) / sigma0
    eii_01 = float(sigmoid(eii_z))

    site_scores = site_level_scores(logits)
    branch_scores = branch_level_scores(logits)

    return {
        "regime": regime_name,
        "gene_length": L,
        "n_branches": B,
        "true_burden": float(Y.mean()),
        "effect_size": effect_size,
        "branch_frac": branch_frac,
        "D_obs": d_obs,
        "mu0": mu0,
        "sigma0": sigma0,
        "EII_z": eii_z,
        "EII_01": eii_01,
        "pred_gene_burden": float(site_scores.mean()),
        "site_score_mean": float(site_scores.mean()),
        "site_score_var": safe_var(site_scores, ddof=1),
        "branch_score_mean": float(branch_scores.mean()),
        "branch_score_var": safe_var(branch_scores, ddof=1),
        "identifiability_extent_heuristic": categorize_eii01(eii_01, weak_cut=weak_cut, strong_cut=strong_cut),
        "identifiable_bool_heuristic": int(eii_01 >= weak_cut),
    }


# -----------------------------------------------------------------------------
# Threshold and ROC analysis
# -----------------------------------------------------------------------------

def label_positive(regime_series: pd.Series, target: str) -> pd.Series:
    if target == "any_nonneutral":
        return (regime_series != "neutral").astype(int)
    if target == "medium_high":
        return regime_series.isin(["medium", "high"]).astype(int)
    if target == "high_only":
        return (regime_series == "high").astype(int)
    raise ValueError(f"Unknown target: {target}")


def confusion_from_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    p = tp + fn
    n = tn + fp
    tpr = tp / p if p > 0 else float("nan")
    fpr = fp / n if n > 0 else float("nan")
    tnr = tn / n if n > 0 else float("nan")
    fnr = fn / p if p > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    acc = (tp + tn) / (p + n) if (p + n) > 0 else float("nan")
    bal_acc = np.nanmean([tpr, tnr])
    youden_j = tpr - fpr if (not np.isnan(tpr) and not np.isnan(fpr)) else float("nan")

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "FNR": fnr,
        "precision": precision,
        "NPV": npv,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "youden_j": youden_j,
    }


def threshold_grid_from_scores(score: np.ndarray, n_grid: int = 1001) -> np.ndarray:
    lo = float(np.min(score))
    hi = float(np.max(score))
    grid = np.linspace(lo, hi, n_grid)
    return np.unique(np.concatenate(([lo - 1e-9], grid, [hi + 1e-9])))


def make_threshold_table(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    n_grid: int = 1001,
) -> pd.DataFrame:
    y_true = label_positive(df["regime"], target).to_numpy(dtype=int)
    score = df[score_col].to_numpy(dtype=float)
    rows = [confusion_from_threshold(y_true=y_true, score=score, threshold=t) for t in threshold_grid_from_scores(score, n_grid=n_grid)]
    out = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    out.insert(0, "score_col", score_col)
    out.insert(1, "target", target)
    return out


def auc_from_roc(roc_df: pd.DataFrame) -> float:
    x = roc_df["FPR"].to_numpy(dtype=float)
    y = roc_df["TPR"].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapezoid(y, x))


def roc_table_from_threshold_table(thr_df: pd.DataFrame) -> pd.DataFrame:
    roc_df = thr_df[["threshold", "FPR", "TPR", "balanced_accuracy", "youden_j"]].copy()
    roc_df = roc_df.dropna(subset=["FPR", "TPR"]).drop_duplicates(subset=["FPR", "TPR"]).sort_values(["FPR", "TPR"]).reset_index(drop=True)
    return roc_df


def neutral_quantile_thresholds(df: pd.DataFrame, quantiles: Iterable[float]) -> pd.DataFrame:
    neutral = df.loc[df["regime"] == "neutral"].copy()
    rows = []
    for q in quantiles:
        eii01_thr = float(np.quantile(neutral["EII_01"].to_numpy(), q))
        eii_z_thr = float(np.quantile(neutral["EII_z"].to_numpy(), q))
        rows.append({
            "neutral_quantile": q,
            "threshold_EII_01": eii01_thr,
            "threshold_EII_z": eii_z_thr,
        })
    return pd.DataFrame(rows)


def threshold_perf_at_fixed_values(
    df: pd.DataFrame,
    thresholds: List[Tuple[str, float]],
    target: str,
) -> pd.DataFrame:
    y_true = label_positive(df["regime"], target).to_numpy(dtype=int)
    rows = []
    for score_col, thr in thresholds:
        perf = confusion_from_threshold(y_true=y_true, score=df[score_col].to_numpy(dtype=float), threshold=thr)
        perf["score_col"] = score_col
        rows.append(perf)
    return pd.DataFrame(rows)


def classwise_positive_rates(df: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    for regime, sub in df.groupby("regime", sort=False):
        rate = float(np.mean(sub[score_col].to_numpy() >= threshold))
        rows.append({
            "regime": regime,
            "score_col": score_col,
            "threshold": threshold,
            "positive_rate": rate,
            "n": int(sub.shape[0]),
        })
    return pd.DataFrame(rows)


def choose_operating_points(thr_df: pd.DataFrame, max_fpr_values: List[float]) -> pd.DataFrame:
    rows = []
    clean = thr_df.dropna(subset=["FPR", "TPR", "balanced_accuracy", "youden_j"]).copy()
    if clean.empty:
        return pd.DataFrame(columns=["criterion", "threshold", "FPR", "TPR", "balanced_accuracy", "youden_j"])

    best_j = clean.loc[clean["youden_j"].idxmax()].to_dict()
    best_j["criterion"] = "max_youden_j"
    rows.append(best_j)

    best_bal = clean.loc[clean["balanced_accuracy"].idxmax()].to_dict()
    best_bal["criterion"] = "max_balanced_accuracy"
    rows.append(best_bal)

    for max_fpr in max_fpr_values:
        eligible = clean.loc[clean["FPR"] <= max_fpr].copy()
        if eligible.empty:
            continue
        best = eligible.loc[eligible["TPR"].idxmax()].to_dict()
        best["criterion"] = f"max_tpr_at_fpr_le_{max_fpr:.02f}"
        rows.append(best)

    cols = ["criterion", "threshold", "FPR", "TPR", "balanced_accuracy", "youden_j"]
    return pd.DataFrame(rows)[cols]


# -----------------------------------------------------------------------------
# Summaries and plots
# -----------------------------------------------------------------------------

def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("regime")
        .agg(
            n=("EII_01", "size"),
            true_burden_mean=("true_burden", "mean"),
            EII_01_mean=("EII_01", "mean"),
            EII_01_median=("EII_01", "median"),
            EII_01_sd=("EII_01", "std"),
            EII_z_mean=("EII_z", "mean"),
            EII_z_median=("EII_z", "median"),
            heuristic_identifiable_rate=("identifiable_bool_heuristic", "mean"),
            heuristic_strong_rate=("identifiability_extent_heuristic", lambda s: np.mean(s == "strongly_identifiable")),
        )
        .reset_index()
    )


def plot_eii_histogram(df: pd.DataFrame, quantile_df: pd.DataFrame, outpath: str) -> None:
    plt.figure(figsize=(9, 6))
    order = ["neutral", "low", "medium", "high"]
    for regime in order:
        x = df.loc[df["regime"] == regime, "EII_01"].to_numpy()
        plt.hist(x, bins=40, density=True, alpha=0.45, label=regime)

    for _, row in quantile_df.iterrows():
        plt.axvline(row["threshold_EII_01"], linestyle="--", linewidth=1)

    plt.xlabel("EII_01")
    plt.ylabel("Density")
    plt.title("EII_01 distributions with neutral-quantile thresholds")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_roc(roc_df: pd.DataFrame, auc_val: float, outpath: str) -> None:
    plt.figure(figsize=(6.5, 6.5))
    plt.plot(roc_df["FPR"], roc_df["TPR"], linewidth=1.5)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve (AUC = {auc_val:.3f})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_threshold_tradeoff(thr_df: pd.DataFrame, outpath: str) -> None:
    plt.figure(figsize=(8.5, 6))
    plt.plot(thr_df["threshold"], thr_df["FPR"], label="FPR", linewidth=1.5)
    plt.plot(thr_df["threshold"], thr_df["TPR"], label="TPR", linewidth=1.5)
    plt.plot(thr_df["threshold"], thr_df["balanced_accuracy"], label="Balanced accuracy", linewidth=1.5)
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold trade-off curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_classwise_positive_rates(rate_df: pd.DataFrame, outpath: str, title: str) -> None:
    plt.figure(figsize=(7.5, 5.5))
    x = np.arange(rate_df.shape[0])
    plt.bar(x, rate_df["positive_rate"].to_numpy())
    plt.xticks(x, rate_df["regime"].tolist())
    plt.ylim(0.0, 1.0)
    plt.ylabel("Positive rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upgraded EII calibration simulator with neutral-quantile thresholds and ROC-style summaries."
    )
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--n_per_regime", type=int, default=1000, help="Genes per regime")
    p.add_argument(
        "--dispersion_mode",
        choices=["site_score_variance", "site_logit_variance", "full_logit_variance"],
        default="site_logit_variance",
        help="Dispersion statistic used to form D_obs",
    )
    p.add_argument(
        "--decision_target",
        choices=["any_nonneutral", "medium_high", "high_only"],
        default="medium_high",
        help="Defines the positive class for ROC/threshold analysis",
    )
    p.add_argument("--n_branches", type=int, default=12)
    p.add_argument("--gene_length_min", type=int, default=200)
    p.add_argument("--gene_length_max", type=int, default=500)
    p.add_argument("--neutral_reps", type=int, default=200)
    p.add_argument("--measurement_noise_sd", type=float, default=0.80)
    p.add_argument("--branch_random_sd", type=float, default=0.50)
    p.add_argument("--site_random_sd", type=float, default=0.40)
    p.add_argument("--baseline_logit_mean", type=float, default=-2.2)
    p.add_argument("--baseline_logit_sd", type=float, default=0.20)
    p.add_argument("--sigma_floor", type=float, default=0.05)
    p.add_argument("--heuristic_weak_cut", type=float, default=0.70)
    p.add_argument("--heuristic_strong_cut", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_threshold_grid", type=int, default=1001)
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.outdir)

    cfg = SimConfig(
        n_branches=args.n_branches,
        gene_length_min=args.gene_length_min,
        gene_length_max=args.gene_length_max,
        neutral_reps=args.neutral_reps,
        measurement_noise_sd=args.measurement_noise_sd,
        branch_random_sd=args.branch_random_sd,
        site_random_sd=args.site_random_sd,
        baseline_logit_mean=args.baseline_logit_mean,
        baseline_logit_sd=args.baseline_logit_sd,
        sigma_floor=args.sigma_floor,
        random_seed=args.seed,
    )

    rng = np.random.default_rng(cfg.random_seed)
    rows: List[Dict[str, float]] = []
    for regime in ["neutral", "low", "medium", "high"]:
        for _ in range(args.n_per_regime):
            rows.append(
                simulate_one_gene(
                    regime_name=regime,
                    cfg=cfg,
                    rng=rng,
                    mode=args.dispersion_mode,
                    weak_cut=args.heuristic_weak_cut,
                    strong_cut=args.heuristic_strong_cut,
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "synthetic_eii_regimes.csv"), index=False)

    summary_df = summarize_by_regime(df)
    summary_df.to_csv(os.path.join(args.outdir, "regime_summary.csv"), index=False)

    quantile_df = neutral_quantile_thresholds(df, quantiles=[0.90, 0.95, 0.99])
    quantile_df.to_csv(os.path.join(args.outdir, "neutral_quantile_thresholds.csv"), index=False)

    thresholds_fixed: List[Tuple[str, float]] = [
        ("EII_01", args.heuristic_weak_cut),
        ("EII_01", args.heuristic_strong_cut),
    ]
    for _, row in quantile_df.iterrows():
        q = float(row["neutral_quantile"])
        thresholds_fixed.append(("EII_01", float(row["threshold_EII_01"])))
        thresholds_fixed.append(("EII_z", float(row["threshold_EII_z"])))

    fixed_perf_df = threshold_perf_at_fixed_values(df, thresholds=thresholds_fixed, target=args.decision_target)
    fixed_perf_df.to_csv(os.path.join(args.outdir, "fixed_threshold_performance.csv"), index=False)

    classwise_tables = []
    for _, row in quantile_df.iterrows():
        q = float(row["neutral_quantile"])
        eii01_thr = float(row["threshold_EII_01"])
        tmp = classwise_positive_rates(df, score_col="EII_01", threshold=eii01_thr)
        tmp.insert(0, "criterion", f"neutral_q{int(round(q * 100))}_EII01")
        classwise_tables.append(tmp)
    classwise_df = pd.concat(classwise_tables, ignore_index=True)
    classwise_df.to_csv(os.path.join(args.outdir, "classwise_positive_rates.csv"), index=False)

    thr_eii01 = make_threshold_table(df, score_col="EII_01", target=args.decision_target, n_grid=args.n_threshold_grid)
    thr_eii01.to_csv(os.path.join(args.outdir, "threshold_table_EII_01.csv"), index=False)

    thr_eiiz = make_threshold_table(df, score_col="EII_z", target=args.decision_target, n_grid=args.n_threshold_grid)
    thr_eiiz.to_csv(os.path.join(args.outdir, "threshold_table_EII_z.csv"), index=False)

    roc_eii01 = roc_table_from_threshold_table(thr_eii01)
    roc_eii01.to_csv(os.path.join(args.outdir, "roc_EII_01.csv"), index=False)
    auc_eii01 = auc_from_roc(roc_eii01)

    roc_eiiz = roc_table_from_threshold_table(thr_eiiz)
    roc_eiiz.to_csv(os.path.join(args.outdir, "roc_EII_z.csv"), index=False)
    auc_eiiz = auc_from_roc(roc_eiiz)

    operating_points_eii01 = choose_operating_points(thr_eii01, max_fpr_values=[0.10, 0.05, 0.01])
    operating_points_eii01.to_csv(os.path.join(args.outdir, "operating_points_EII_01.csv"), index=False)

    operating_points_eiiz = choose_operating_points(thr_eiiz, max_fpr_values=[0.10, 0.05, 0.01])
    operating_points_eiiz.to_csv(os.path.join(args.outdir, "operating_points_EII_z.csv"), index=False)

    plot_eii_histogram(df, quantile_df, os.path.join(args.outdir, "eii01_histogram_with_quantiles.png"))
    plot_roc(roc_eii01, auc_eii01, os.path.join(args.outdir, "roc_EII_01.png"))
    plot_threshold_tradeoff(thr_eii01, os.path.join(args.outdir, "threshold_tradeoff_EII_01.png"))

    if not classwise_df.empty:
        q95_rows = classwise_df.loc[classwise_df["criterion"] == "neutral_q95_EII01"].copy()
        if not q95_rows.empty:
            plot_classwise_positive_rates(
                q95_rows,
                os.path.join(args.outdir, "classwise_positive_rates_q95_EII01.png"),
                title="Classwise positive rates at neutral 95th-percentile EII_01 threshold",
            )

    with open(os.path.join(args.outdir, "run_metadata.txt"), "w", encoding="utf-8") as fh:
        fh.write("Upgraded EII calibration run\n")
        fh.write("============================\n")
        fh.write(f"decision_target: {args.decision_target}\n")
        fh.write(f"dispersion_mode: {args.dispersion_mode}\n")
        fh.write(f"n_per_regime: {args.n_per_regime}\n")
        fh.write(f"AUC_EII_01: {auc_eii01:.6f}\n")
        fh.write(f"AUC_EII_z: {auc_eiiz:.6f}\n")
        fh.write("\nSimulation config\n")
        fh.write("-----------------\n")
        for k, v in asdict(cfg).items():
            fh.write(f"{k}: {v}\n")

    print("Done. Wrote outputs to:", args.outdir)
    print("\nRegime summary\n--------------")
    print(summary_df.to_string(index=False))
    print("\nNeutral quantile thresholds\n---------------------------")
    print(quantile_df.to_string(index=False))
    print("\nFixed-threshold performance\n---------------------------")
    print(fixed_perf_df.to_string(index=False))
    print("\nOperating points (EII_01)\n-------------------------")
    if operating_points_eii01.empty:
        print("No operating points available")
    else:
        print(operating_points_eii01.to_string(index=False))
    print(f"\nAUC(EII_01) = {auc_eii01:.4f}")
    print(f"AUC(EII_z)  = {auc_eiiz:.4f}")


if __name__ == "__main__":
    main()
