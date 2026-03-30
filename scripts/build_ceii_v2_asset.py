#!/usr/bin/env python3
"""Build cEII calibration assets from sigma-valid expanded benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.dispersion import PRIMARY_DISPERSION_METHOD, SUPPORTED_DISPERSION_METHODS  # noqa: E402
from babappai.validation.full_pipeline_validation import run_full_pipeline_inference_on_dataset  # noqa: E402


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t") if row]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-tsv",
        default="results/validation/ceii_benchmark_v2_expanded/dataset/synthetic_dataset.tsv",
    )
    p.add_argument("--outdir", default="results/validation/ceii_v3_1")
    p.add_argument("--calibration-version", default="ceii_v3.1")
    p.add_argument("--dispersion-method", default=PRIMARY_DISPERSION_METHOD)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sigma-floor", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--neutral-reps", type=int, default=12)
    p.add_argument("--min-neutral-group-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--write-package-asset", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.dispersion_method not in SUPPORTED_DISPERSION_METHODS:
        raise ValueError(
            f"Unsupported dispersion method: {args.dispersion_method}. "
            f"Supported: {sorted(SUPPORTED_DISPERSION_METHODS)}"
        )

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_tsv = Path(args.dataset_tsv).expanduser().resolve()
    inference_dir = outdir / "inference"
    calibration_dir = outdir / "calibration"

    inference_meta = run_full_pipeline_inference_on_dataset(
        dataset_tsv=str(dataset_tsv),
        outdir=str(inference_dir),
        tree_calibration=False,
        n_calibration=0,
        device=str(args.device),
        batch_size=int(args.batch_size),
        sigma_floor=float(args.sigma_floor),
        alpha=float(args.alpha),
        pvalue_mode="empirical_monte_carlo",
        dispersion_method=str(args.dispersion_method),
        min_neutral_group_size=int(args.min_neutral_group_size),
        neutral_reps=int(args.neutral_reps),
        offline=bool(args.offline),
        overwrite=bool(args.overwrite),
    )

    fit_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "fit_ceii_calibration.py"),
        "--metrics-tsv",
        str(inference_meta["metrics_tsv"]),
        "--outdir",
        str(calibration_dir),
        "--calibration-version",
        str(args.calibration_version),
        "--label-profile",
        "v3_1",
        "--bootstrap-reps",
        "200",
        "--seed",
        str(int(args.seed)),
        "--app-min-n-taxa",
        "8",
        "--app-max-n-taxa",
        "64",
        "--app-min-gene-length-nt",
        "120",
        "--app-max-gene-length-nt",
        "4500",
    ]
    if args.write_package_asset:
        fit_cmd.append("--write-package-asset")
    subprocess.run(fit_cmd, check=True)

    version_slug = str(args.calibration_version)
    # Required publication deliverables with versioned names.
    asset_src = calibration_dir / "ceii_calibration_asset.json"
    split_src = calibration_dir / "ceii_split_performance.tsv"
    rel_src = calibration_dir / "ceii_reliability.tsv"
    app_src = calibration_dir / "ceii_applicability_summary.json"

    asset_dst = outdir / f"{version_slug}_asset.json"
    split_dst = outdir / f"{version_slug}_split_performance.tsv"
    rel_dst = outdir / f"{version_slug}_reliability.tsv"
    app_dst = outdir / f"{version_slug}_applicability_summary.json"
    for src, dst in (
        (asset_src, asset_dst),
        (split_src, split_dst),
        (rel_src, rel_dst),
        (app_src, app_dst),
    ):
        shutil.copy2(src, dst)

    split_rows = _read_tsv(split_dst)
    rel_rows = _read_tsv(rel_dst)
    applicability_payload = json.loads(app_dst.read_text()) if app_dst.exists() else {}

    summary = {
        "calibration_version": version_slug,
        "dataset_tsv": str(dataset_tsv),
        "dispersion_method": str(args.dispersion_method),
        "sigma_floor": float(args.sigma_floor),
        "neutral_reps": int(args.neutral_reps),
        "min_neutral_group_size": int(args.min_neutral_group_size),
        "inference_summary_json": str(inference_meta["summary_json"]),
        "fraction_sigma0_at_floor": float(inference_meta["fraction_sigma0_at_floor"]),
        "fraction_fallback_applied": float(inference_meta["fraction_fallback_applied"]),
        "split_performance_rows": int(len(split_rows)),
        "reliability_rows": int(len(rel_rows)),
        "applicability_coverage": applicability_payload,
        "outputs": {
            "asset": str(asset_dst),
            "split_performance_tsv": str(split_dst),
            "reliability_tsv": str(rel_dst),
            "applicability_summary_json": str(app_dst),
        },
    }
    summary_path = outdir / f"{version_slug}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
