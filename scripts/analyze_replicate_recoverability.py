#!/usr/bin/env python3
"""Analyze replicate-level recoverability and reproducibility for synthetic scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.validation.full_pipeline_validation import analyze_replicate_recoverability


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics_tsv", required=True, help="Gene-level full-pipeline metrics table")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--default_threshold", type=float, default=0.70)
    p.add_argument("--alpha", type=float, default=0.05)
    return p


def main() -> int:
    args = build_parser().parse_args()
    meta = analyze_replicate_recoverability(
        metrics_tsv=args.metrics_tsv,
        outdir=args.outdir,
        default_threshold=args.default_threshold,
        alpha=args.alpha,
    )

    print("Replicate recoverability analysis complete")
    print(f"  scenario_recoverability_tsv: {meta['scenario_recoverability_tsv']}")
    print(f"  pairwise_corr_tsv: {meta['pairwise_corr_tsv']}")
    print(f"  stratified_tsv: {meta['stratified_tsv']}")
    print(f"  outdir: {Path(args.outdir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
