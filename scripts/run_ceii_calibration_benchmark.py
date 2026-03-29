#!/usr/bin/env python3
"""Run end-to-end empirical cEII benchmark and calibration export."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babappai.validation.full_pipeline_validation import (  # noqa: E402
    run_full_pipeline_inference_on_dataset,
    simulate_alignment_validation_dataset,
)
from babappai.dispersion import PRIMARY_DISPERSION_METHOD, SUPPORTED_DISPERSION_METHODS  # noqa: E402


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t") if row]


def _export_figures(calibration_dir: Path, figures_dir: Path) -> Dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(figures_dir / ".mplconfig"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": f"skipped: matplotlib unavailable ({exc})"}

    outputs: Dict[str, str] = {}

    rel_path = calibration_dir / "ceii_reliability.tsv"
    if rel_path.exists():
        rel_rows = _read_tsv(rel_path)
        if rel_rows:
            fig = plt.figure(figsize=(6.2, 4.2))
            ax = fig.add_subplot(111)
            for target, color in (("I_gene", "#1b9e77"), ("I_site", "#d95f02")):
                for split, linestyle in (("calibration", "-"), ("test", "--"), ("ood", ":")):
                    subset = [r for r in rel_rows if r.get("target") == target and r.get("split") == split]
                    if not subset:
                        continue
                    subset.sort(key=lambda r: float(r["mean_pred"]))
                    x = [float(r["mean_pred"]) for r in subset]
                    y = [float(r["empirical_rate"]) for r in subset]
                    ax.plot(x, y, linestyle=linestyle, color=color, linewidth=1.7, label=f"{target} {split}")
            ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, alpha=0.6)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Empirical identifiability rate")
            ax.set_title("cEII reliability")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            out = figures_dir / "figure_ceii_reliability.png"
            fig.savefig(out, dpi=220)
            plt.close(fig)
            outputs["figure_ceii_reliability"] = str(out)

    perf_path = calibration_dir / "ceii_split_performance.tsv"
    if perf_path.exists():
        perf_rows = _read_tsv(perf_path)
        if perf_rows:
            rows = [r for r in perf_rows if r.get("split") in {"test", "ood"}]
            if rows:
                labels = [f"{r['target']}\n{r['split']}" for r in rows]
                ppv = [float(r.get("ppv_at_threshold", "nan")) for r in rows]
                fdr = [float(r.get("fdr_at_threshold", "nan")) for r in rows]
                bal = [float(r.get("balanced_accuracy_at_threshold", "nan")) for r in rows]
                x = list(range(len(rows)))

                fig = plt.figure(figsize=(7.0, 4.0))
                ax = fig.add_subplot(111)
                w = 0.25
                ax.bar([i - w for i in x], ppv, width=w, label="PPV", color="#1b9e77")
                ax.bar(x, bal, width=w, label="Balanced Acc.", color="#7570b3")
                ax.bar([i + w for i in x], fdr, width=w, label="FDR", color="#e7298a")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("Metric value")
                ax.set_title("Operating characteristics at calibrated thresholds")
                ax.legend(fontsize=8)
                fig.tight_layout()
                out = figures_dir / "figure_ceii_operating_characteristics.png"
                fig.savefig(out, dpi=220)
                plt.close(fig)
                outputs["figure_ceii_operating_characteristics"] = str(out)

    if not outputs:
        return {"status": "skipped: no calibration tables found"}
    return outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-per-regime", type=int, default=6)
    p.add_argument("--n-replicates-per-scenario", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sigma-floor", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--pvalue-mode",
        choices=["empirical_monte_carlo", "frozen_reference"],
        default="empirical_monte_carlo",
    )
    p.add_argument(
        "--dispersion-method",
        choices=list(SUPPORTED_DISPERSION_METHODS),
        default=PRIMARY_DISPERSION_METHOD,
    )
    p.add_argument("--neutral-reps", type=int, default=200)
    p.add_argument("--min-neutral-group-size", type=int, default=20)
    p.add_argument("--calibration-version", default="ceii_v2")
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--target-fdr-gene", type=float, default=0.10)
    p.add_argument("--target-fdr-site", type=float, default=0.10)
    p.add_argument("--app-min-n-taxa", type=int, default=8)
    p.add_argument("--app-max-n-taxa", type=int, default=64)
    p.add_argument("--app-min-gene-length-nt", type=int, default=120)
    p.add_argument("--app-max-gene-length-nt", type=int, default=4500)
    p.add_argument("--write-package-asset", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_dir = outdir / "dataset"
    inference_dir = outdir / "inference"
    calibration_dir = outdir / "calibration"
    figures_dir = outdir / "figures"

    dataset_meta = simulate_alignment_validation_dataset(
        outdir=dataset_dir,
        n_per_regime=int(args.n_per_regime),
        n_replicates_per_scenario=int(args.n_replicates_per_scenario),
        seed=int(args.seed),
    )

    inference_meta = run_full_pipeline_inference_on_dataset(
        dataset_tsv=dataset_meta["dataset_tsv"],
        outdir=inference_dir,
        tree_calibration=False,
        n_calibration=0,
        device=args.device,
        batch_size=int(args.batch_size),
        sigma_floor=float(args.sigma_floor),
        alpha=float(args.alpha),
        pvalue_mode=args.pvalue_mode,
        dispersion_method=args.dispersion_method,
        min_neutral_group_size=int(args.min_neutral_group_size),
        neutral_reps=int(args.neutral_reps),
        offline=bool(args.offline),
        overwrite=bool(args.overwrite),
    )

    fit_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "fit_ceii_calibration.py"),
        "--metrics-tsv",
        inference_meta["metrics_tsv"],
        "--outdir",
        str(calibration_dir),
        "--calibration-version",
        str(args.calibration_version),
        "--bootstrap-reps",
        str(int(args.bootstrap_reps)),
        "--target-fdr-gene",
        str(float(args.target_fdr_gene)),
        "--target-fdr-site",
        str(float(args.target_fdr_site)),
        "--seed",
        str(int(args.seed)),
        "--app-min-n-taxa",
        str(int(args.app_min_n_taxa)),
        "--app-max-n-taxa",
        str(int(args.app_max_n_taxa)),
        "--app-min-gene-length-nt",
        str(int(args.app_min_gene_length_nt)),
        "--app-max-gene-length-nt",
        str(int(args.app_max_gene_length_nt)),
    ]
    if args.write_package_asset:
        fit_cmd.append("--write-package-asset")
    subprocess.run(fit_cmd, check=True)

    figure_meta = _export_figures(calibration_dir, figures_dir)

    summary = {
        "dataset": dataset_meta,
        "inference": inference_meta,
        "calibration": {
            "asset": str(calibration_dir / "ceii_calibration_asset.json"),
            "summary": str(calibration_dir / "ceii_calibration_summary.json"),
            "split_performance": str(calibration_dir / "ceii_split_performance.tsv"),
            "reliability": str(calibration_dir / "ceii_reliability.tsv"),
            "recoverability_augmented": str(calibration_dir / "recoverability_augmented.tsv"),
        },
        "figures": figure_meta,
    }

    summary_path = outdir / "ceii_benchmark_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("cEII benchmark + calibration complete")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
