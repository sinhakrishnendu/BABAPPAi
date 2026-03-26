#!/usr/bin/env python3
"""Run full-pipeline synthetic validation for EII diagnostics and empirical significance calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.validation.full_pipeline_validation import run_full_pipeline_validation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n_per_regime", type=int, default=500)
    p.add_argument("--n_replicates_per_scenario", type=int, default=5)
    p.add_argument("--bootstrap_reps", type=int, default=1000)
    p.add_argument("--default_threshold", type=float, default=0.70)
    p.add_argument(
        "--decision_target",
        choices=["any_nonneutral", "medium_high", "high_only"],
        default="medium_high",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--tree_calibration", action="store_true")
    p.add_argument("--n_calibration", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--sigma_floor", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--pvalue_mode",
        choices=["empirical_monte_carlo", "frozen_reference"],
        default="empirical_monte_carlo",
    )
    p.add_argument("--neutral_reps", type=int, default=200)
    p.add_argument("--min_neutral_group_size", type=int, default=20)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no_overwrite", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    meta = run_full_pipeline_validation(
        outdir=args.outdir,
        n_per_regime=args.n_per_regime,
        n_replicates_per_scenario=args.n_replicates_per_scenario,
        bootstrap_reps=args.bootstrap_reps,
        default_threshold=args.default_threshold,
        decision_target=args.decision_target,
        seed=args.seed,
        tree_calibration=args.tree_calibration,
        n_calibration=args.n_calibration,
        device=args.device,
        batch_size=args.batch_size,
        sigma_floor=args.sigma_floor,
        alpha=args.alpha,
        pvalue_mode=args.pvalue_mode,
        neutral_reps=args.neutral_reps,
        min_neutral_group_size=args.min_neutral_group_size,
        offline=args.offline,
        overwrite=not args.no_overwrite,
    )

    print("Full-pipeline EII validation complete")
    print(f"outdir: {Path(meta['outdir']).resolve()}")
    print("\nConsole Summary")
    print("---------------")
    print(f"global neutral 95th percentile (EII_01): {meta['global_neutral_q95_eii01']:.3f}")
    print(f"global neutral 99th percentile (EII_01): {meta['global_neutral_q99_eii01']:.3f}")
    print(f"FPR at 0.70: {meta['global_fpr_at_070']:.3f}")
    print(f"TPR at 0.70: {meta['global_tpr_at_070']:.3f}")
    print(f"AUC: {meta['global_auc']:.3f}")
    print(f"neutral significant rate at q<=alpha: {meta['neutral_significant_rate_q']:.3f}")
    print(f"medium/high significant rate at q<=alpha: {meta['medium_high_significant_rate_q']:.3f}")
    print(f"q_emp FPR: {meta['q_emp_fpr']:.3f}")
    print(f"q_emp TPR: {meta['q_emp_tpr']:.3f}")
    print(f"neutral p_emp KS: {meta['neutral_p_uniformity_ks']:.3f}")
    print(f"fraction sigma0 at floor: {meta['fraction_sigma0_at_floor']:.3f}")
    print(f"fraction fallback applied: {meta['fraction_fallback_applied']:.3f}")
    raw = meta["sigma0_raw_summary"]
    final = meta["sigma0_final_summary"]
    print(
        "sigma0_raw summary: "
        f"min={float(raw['min']):.4g}, median={float(raw['median']):.4g}, "
        f"q95={float(raw['q95']):.4g}, q99={float(raw['q99']):.4g}, max={float(raw['max']):.4g}"
    )
    print(
        "sigma0_final summary: "
        f"min={float(final['min']):.4g}, median={float(final['median']):.4g}, "
        f"q95={float(final['q95']):.4g}, q99={float(final['q99']):.4g}, max={float(final['max']):.4g}"
    )
    print(f"significance calibration status: {meta['significance_calibration_status']}")

    print("\nKey Outputs")
    print("-----------")
    print(f"metrics_tsv: {meta['metrics_tsv']}")
    print(f"calibration_debug_tsv: {meta['calibration_debug_tsv']}")
    print(f"quantiles_tsv: {meta['quantiles_tsv']}")
    print(f"performance_tsv: {meta['performance_tsv']}")
    print(f"significance_performance_tsv: {meta['significance_performance_tsv']}")
    print(f"significance_regime_rates_tsv: {meta['significance_regime_rates_tsv']}")
    print(f"significance_summary_json: {meta['significance_summary_json']}")
    print(f"bootstrap_summary_tsv: {meta['bootstrap_summary_tsv']}")
    print(f"recoverability_tsv: {meta['recoverability_tsv']}")
    print(f"report_md: {meta['report_md']}")
    print(f"paper_text_md: {meta['paper_text_md']}")
    print(f"methods_md: {meta['methods_md']}")
    print(f"interpretation_txt: {meta['interpretation_txt']}")
    print(f"release_notes_md: {meta['release_notes_md']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
