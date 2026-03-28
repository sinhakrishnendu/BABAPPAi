#!/usr/bin/env python3
"""Generate nuisance-stratified synthetic alignment datasets for full-pipeline validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.validation.full_pipeline_validation import simulate_alignment_validation_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True, help="Output directory for simulated dataset")
    p.add_argument("--n_per_regime", type=int, default=100, help="Scenarios per regime")
    p.add_argument(
        "--n_replicates_per_scenario",
        type=int,
        default=5,
        help="Observed replicates generated for each latent scenario",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--dispersion_choices",
        default="site_logit_variance",
        help="Comma-separated dispersion statistic labels. Locked method supports site_logit_variance only.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    choices = [x.strip() for x in args.dispersion_choices.split(",") if x.strip()]
    if not choices:
        raise ValueError("At least one dispersion choice is required.")

    meta = simulate_alignment_validation_dataset(
        outdir=args.outdir,
        n_per_regime=args.n_per_regime,
        n_replicates_per_scenario=args.n_replicates_per_scenario,
        seed=args.seed,
        dispersion_choices=choices,
    )

    print("Synthetic validation dataset generated")
    print(f"  dataset_tsv: {meta['dataset_tsv']}")
    print(f"  scenario_tsv: {meta['scenario_tsv']}")
    print(f"  summary_json: {meta['summary_json']}")
    print(f"  outdir: {Path(args.outdir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
