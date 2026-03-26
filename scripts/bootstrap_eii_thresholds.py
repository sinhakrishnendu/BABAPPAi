#!/usr/bin/env python3
"""Bootstrap uncertainty intervals for EII diagnostics and empirical significance summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.validation.full_pipeline_validation import bootstrap_eii_thresholds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics_tsv", required=True, help="Gene-level full-pipeline metrics table")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--bootstrap_reps", type=int, default=1000)
    p.add_argument("--default_threshold", type=float, default=0.70)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--decision_target",
        choices=["any_nonneutral", "medium_high", "high_only"],
        default="medium_high",
    )
    p.add_argument("--seed", type=int, default=123)
    return p


def main() -> int:
    args = build_parser().parse_args()
    meta = bootstrap_eii_thresholds(
        metrics_tsv=args.metrics_tsv,
        outdir=args.outdir,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        decision_target=args.decision_target,
        default_threshold=args.default_threshold,
        alpha=args.alpha,
    )
    print("Bootstrap calibration complete")
    print(f"  bootstrap_replicates_tsv: {meta['bootstrap_replicates_tsv']}")
    print(f"  bootstrap_summary_tsv: {meta['bootstrap_summary_tsv']}")
    print(f"  outdir: {Path(args.outdir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
