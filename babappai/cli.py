"""Command-line interface for BABAPPAi."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import List, Optional

from babappai import __version__
from babappai.calibration import default_calibration_asset_path, load_calibration_asset
from babappai.calibration.neutral_generator_adapter import run_neutral_generator
from babappai.metadata import (
    MODEL_COMPATIBILITY_NOTE,
    MODEL_DOI,
    MODEL_TAG,
    PACKAGE_NAME,
    SOFTWARE_NAME,
)
from babappai.model_manager import (
    ModelError,
    fetch_model,
    model_status,
    verify_cached_model,
)
from babappai.run_pipeline import run_and_write_outputs, terminal_summary
from babappai.tree import load_tree
from babappai.encoding import encode_alignment
from babappai.validation.empirical_validation import run_empirical_validation
from babappai.validation.orthogroup_selection import select_orthogroups
from babappai.validation.synthetic_validation import run_synthetic_validation
from babappai.validation.validation_reporting import generate_validation_report


def _print_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def _print_model_status(status: dict) -> None:
    print(f"Model tag: {status['model_tag']}")
    print(f"Legacy model name: {status['legacy_model_name']}")
    print(f"Cached: {'YES' if status['cached'] else 'NO'}")
    print(f"Path: {status['cached_path']}")
    print(f"DOI: {status['doi']}")
    print(f"Expected SHA-256: {status['expected_sha256']}")
    print(f"Verified: {'YES' if status['verified'] else 'NO'}")
    if status.get("actual_sha256"):
        print(f"Actual SHA-256: {status['actual_sha256']}")
    print(f"Compatibility: {status['compatibility_note']}")


def cmd_run(args: argparse.Namespace) -> int:
    neutral_meta = None
    try:
        if args.neutral_generator:
            neutral_meta = run_neutral_generator(
                generator_path=args.neutral_generator,
                output_dir=str(Path(args.outdir) / "neutral_generator"),
                model_tag=MODEL_TAG,
                seed=args.seed,
                extra_args=[],
            )

        payload = run_and_write_outputs(
            alignment_path=args.alignment,
            tree_path=args.tree,
            outdir=Path(args.outdir),
            command=" ".join(shlex.quote(a) for a in sys.argv),
            tree_calibration=args.tree_calibration,
            n_calibration=args.n_calibration,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            foreground_mode=args.foreground_mode,
            foreground_list=args.foreground_list,
            offline=args.offline,
            overwrite=args.overwrite,
            sigma_floor=args.sigma_floor,
            alpha=args.alpha,
            pvalue_mode=args.pvalue_mode,
            retain_eii_bands=args.retain_eii_bands,
            report_threshold_bands=args.report_threshold_bands,
            ceii_enabled=args.ceii_enabled,
            ceii_asset_path=args.ceii_asset,
            min_neutral_group_size=args.min_neutral_group_size,
            neutral_reps=args.neutral_reps,
            neutral_generator_metadata=neutral_meta,
        )
    except Exception as exc:
        _print_error(str(exc))
        return 1

    if not args.quiet:
        for line in terminal_summary(payload):
            print(line)
    return 0


def cmd_model_fetch(args: argparse.Namespace) -> int:
    try:
        status = fetch_model(force=args.force, offline=args.offline)
    except ModelError as exc:
        _print_error(str(exc))
        return 1

    if status["downloaded"]:
        print("Model downloaded and verified.")
    else:
        print("Verified cached model already present.")
    _print_model_status(status)
    return 0


def cmd_model_status(args: argparse.Namespace) -> int:
    status = model_status()
    _print_model_status(status)
    return 0


def cmd_model_verify(args: argparse.Namespace) -> int:
    try:
        status = verify_cached_model()
    except ModelError as exc:
        _print_error(str(exc))
        return 1

    print("Model checksum verification: PASS")
    _print_model_status(status)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    problems = []

    print("BABAPPAi doctor")
    print("=" * 40)
    print(f"[OK] package: {PACKAGE_NAME} {__version__}")

    if sys.version_info < (3, 9):
        problems.append("Python >= 3.9 is required.")
        print("[FAIL] Python version is below 3.9")
    else:
        print(f"[OK] Python version: {sys.version.split()[0]}")

    try:
        import torch

        print(f"[OK] torch import: {torch.__version__}")
    except Exception as exc:
        problems.append("PyTorch import failed. Install torch>=2.0.")
        print(f"[FAIL] torch import failed: {exc}")

    status = model_status()
    print(f"[OK] cache path: {status['cached_path']}")

    if not status["cached"]:
        problems.append("Frozen model is not cached.")
        print("[FAIL] frozen model not cached")
        print("       Fix: babappai model fetch")
    elif not status["verified"]:
        problems.append("Frozen model checksum mismatch.")
        print("[FAIL] frozen model checksum mismatch")
        print("       Fix: babappai model fetch --force")
    else:
        print("[OK] frozen model cached and checksum-verified")

    if args.alignment:
        try:
            _, ntaxa, L = encode_alignment(args.alignment)
            print(f"[OK] alignment readable: {args.alignment} (n={ntaxa}, L={L})")
        except Exception as exc:
            problems.append("Alignment parsing failed.")
            print(f"[FAIL] alignment parsing failed: {exc}")

    if args.tree:
        try:
            tree = load_tree(args.tree)
            print(f"[OK] tree readable: {args.tree} ({len(tree.get_leaves())} leaves)")
        except Exception as exc:
            problems.append("Tree parsing failed.")
            print(f"[FAIL] tree parsing failed: {exc}")

    if problems:
        print("\nDoctor found issues:")
        for item in problems:
            print(f"- {item}")
        return 1

    print("\nDoctor checks passed.")
    print(MODEL_COMPATIBILITY_NOTE)
    return 0


def cmd_example_write(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    aln_path = outdir / "aln.fasta"
    tree_path = outdir / "tree.nwk"
    config_path = outdir / "synthetic_grid.json"
    notes_path = outdir / "README.txt"

    if not args.overwrite:
        existing = [p for p in (aln_path, tree_path, config_path, notes_path) if p.exists()]
        if existing:
            _print_error(
                "Example files already exist. Use --overwrite to replace: "
                + ", ".join(str(p) for p in existing)
            )
            return 1

    aln_path.write_text(
        ">taxonA\nATGGCTGCTGCTGCTGCT\n"
        ">taxonB\nATGGCTGCTGCTGATGCT\n"
        ">taxonC\nATGGCTGCTGATGATGCT\n"
        ">taxonD\nATGGCTGCTGCTGCTGAT\n"
    )
    tree_path.write_text("((taxonA:0.1,taxonB:0.1):0.1,(taxonC:0.1,taxonD:0.1):0.1);\n")
    config_path.write_text(
        "{\n"
        "  \"grid\": {\n"
        "    \"n_taxa\": [8, 16],\n"
        "    \"alignment_length\": [300, 900],\n"
        "    \"perturbation_sparsity\": [0.02, 0.1],\n"
        "    \"perturbation_magnitude\": [0.5, 1.5],\n"
        "    \"branch_length_scale\": [0.5, 1.0],\n"
        "    \"recombination_rate\": [0.0],\n"
        "    \"alignment_noise\": [0.0]\n"
        "  }\n"
        "}\n"
    )
    notes_path.write_text(
        "Example files for BABAPPAi.\n\n"
        f"Run inference:\n  babappai run --alignment {aln_path} --tree {tree_path} --outdir {outdir / 'demo_out'}\n\n"
        f"Run orthogroup selection:\n  babappai validate orthogroups select --input <ORTHOGROUP_DIR> --outdir {outdir / 'selection_out'}\n\n"
        f"Run synthetic validation:\n  babappai validate synthetic run --simulator <path/to/simulator.py> --outdir {outdir / 'synthetic_out'}\n"
    )

    print(f"Wrote example files to: {outdir}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    status = model_status()
    calibration_version = "unavailable"
    try:
        calibration_version = str(load_calibration_asset(args.ceii_asset).get("calibration_version", "unknown"))
    except Exception:
        pass
    print(f"software_name={SOFTWARE_NAME}")
    print(f"software_version={__version__}")
    print(f"model_tag={MODEL_TAG}")
    print(f"model_doi={MODEL_DOI}")
    print(f"calibration_version={calibration_version}")
    print("model_cached=" + ("yes" if status["cached"] else "no"))
    print("compatibility_note=" + MODEL_COMPATIBILITY_NOTE)
    return 0


def cmd_validate_orthogroups_select(args: argparse.Namespace) -> int:
    try:
        metadata = select_orthogroups(
            input_dir=args.input,
            outdir=args.outdir,
            target_n=args.target_n,
            min_taxa=args.min_taxa,
            occupancy_threshold=args.occupancy_threshold,
            min_length_nt=args.min_length_nt,
            max_missingness=args.max_missingness,
            enforce_one_to_one=not args.allow_duplicates,
            require_no_internal_stops=not args.allow_internal_stops,
        )
    except Exception as exc:
        _print_error(str(exc))
        return 1

    print(f"Selected {metadata['counts']['selected']} orthogroups")
    print(f"Output: {Path(args.outdir).resolve()}")
    return 0


def cmd_validate_orthogroups_run(args: argparse.Namespace) -> int:
    try:
        metadata = run_empirical_validation(
            selected_input=args.input,
            outdir=args.outdir,
            tree_calibration=args.tree_calibration,
            n_calibration=args.n_calibration,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            foreground_mode=args.foreground_mode,
            foreground_list=args.foreground_list,
            offline=args.offline,
            overwrite=args.overwrite,
            sigma_floor=args.sigma_floor,
            alpha=args.alpha,
            pvalue_mode=args.pvalue_mode,
            min_neutral_group_size=args.min_neutral_group_size,
            neutral_reps=args.neutral_reps,
            robustness_limit=args.robustness_limit,
        )
    except Exception as exc:
        _print_error(str(exc))
        return 1

    print(f"Empirical validation complete for {metadata['n_orthogroups']} orthogroups")
    print(f"Output: {Path(args.outdir).resolve()}")
    return 0


def cmd_validate_synthetic_run(args: argparse.Namespace) -> int:
    try:
        summary = run_synthetic_validation(
            simulator_path=args.simulator,
            neutral_generator_path=args.neutral_generator,
            outdir=args.outdir,
            seed=args.seed,
            tree_calibration=args.tree_calibration,
            n_calibration=args.n_calibration,
            device=args.device,
            batch_size=args.batch_size,
            offline=args.offline,
            overwrite=args.overwrite,
            sigma_floor=args.sigma_floor,
            alpha=args.alpha,
            pvalue_mode=args.pvalue_mode,
            min_neutral_group_size=args.min_neutral_group_size,
            neutral_reps=args.neutral_reps,
            grid_config=args.grid_config,
            replicates_per_cell=args.replicates_per_cell,
            balance_target_per_regime=args.balance_target_per_regime,
            max_replicates=args.max_replicates,
        )
    except Exception as exc:
        _print_error(str(exc))
        return 1

    print(f"Synthetic validation complete: {summary['n_replicates']} replicates")
    print(f"Regime counts: {summary['regime_counts']}")
    print(f"Output: {Path(args.outdir).resolve()}")
    return 0


def cmd_validate_report(args: argparse.Namespace) -> int:
    try:
        metadata = generate_validation_report(input_dir=args.input, outdir=args.outdir)
    except Exception as exc:
        _print_error(str(exc))
        return 1

    print(f"Validation report generated with {metadata['n_master_rows']} rows")
    print(f"Output: {Path(args.outdir).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="babappai",
        description=(
            "BABAPPAi: diagnostic framework for identifiability of episodic "
            "branch-site structure (renamed continuation of BABAPPAΩ)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run branch-conditioned identifiability inference")
    run_parser.add_argument("--alignment", required=True)
    run_parser.add_argument("--tree", required=True)
    run_parser.add_argument("--outdir", required=True)
    run_parser.add_argument("--tree-calibration", action="store_true")
    run_parser.add_argument("--n-calibration", type=int, default=200)
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--foreground-mode",
        choices=["all-leaves", "all-branches", "user-list"],
        default="all-leaves",
    )
    run_parser.add_argument("--foreground-list", default=None)
    run_parser.add_argument("--neutral-generator", default=None)
    run_parser.add_argument("--alpha", type=float, default=0.05)
    run_parser.add_argument(
        "--pvalue-mode",
        choices=["empirical_monte_carlo", "frozen_reference"],
        default="empirical_monte_carlo",
    )
    run_parser.add_argument("--neutral-reps", type=int, default=200)
    run_parser.add_argument("--min-neutral-group-size", type=int, default=20)
    run_parser.add_argument("--sigma-floor", type=float, default=0.05)
    run_parser.add_argument("--retain-eii-bands", dest="retain_eii_bands", action="store_true")
    run_parser.add_argument("--no-retain-eii-bands", dest="retain_eii_bands", action="store_false")
    run_parser.set_defaults(retain_eii_bands=True)
    run_parser.add_argument("--report-threshold-bands", dest="report_threshold_bands", action="store_true")
    run_parser.add_argument("--no-report-threshold-bands", dest="report_threshold_bands", action="store_false")
    run_parser.set_defaults(report_threshold_bands=True)
    run_parser.add_argument("--ceii-enabled", dest="ceii_enabled", action="store_true")
    run_parser.add_argument("--no-ceii-enabled", dest="ceii_enabled", action="store_false")
    run_parser.set_defaults(ceii_enabled=True)
    run_parser.add_argument(
        "--ceii-asset",
        default=str(default_calibration_asset_path()),
        help="Path to cEII calibration asset JSON.",
    )
    run_parser.add_argument("--offline", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    model_parser = subparsers.add_parser("model", help="Manage frozen model cache")
    model_sub = model_parser.add_subparsers(dest="model_command")
    model_fetch = model_sub.add_parser("fetch", help="Download and verify frozen model")
    model_fetch.add_argument("--force", action="store_true")
    model_fetch.add_argument("--offline", action="store_true")
    model_fetch.set_defaults(func=cmd_model_fetch)
    model_status_cmd = model_sub.add_parser("status", help="Report model cache status")
    model_status_cmd.set_defaults(func=cmd_model_status)
    model_verify = model_sub.add_parser("verify", help="Re-hash cached model and verify")
    model_verify.set_defaults(func=cmd_model_verify)

    doctor_parser = subparsers.add_parser("doctor", help="Inspect environment and model status")
    doctor_parser.add_argument("--alignment", default=None)
    doctor_parser.add_argument("--tree", default=None)
    doctor_parser.set_defaults(func=cmd_doctor)

    example_parser = subparsers.add_parser("example", help="Write tiny demo inputs")
    example_sub = example_parser.add_subparsers(dest="example_command")
    example_write = example_sub.add_parser("write")
    example_write.add_argument("--outdir", required=True)
    example_write.add_argument("--overwrite", action="store_true")
    example_write.set_defaults(func=cmd_example_write)

    version_parser = subparsers.add_parser("version", help="Print software/model compatibility version info")
    version_parser.add_argument(
        "--ceii-asset",
        default=str(default_calibration_asset_path()),
        help="Optional cEII calibration asset path for reporting calibration_version.",
    )
    version_parser.set_defaults(func=cmd_version)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validation workflows (orthogroups, synthetic, report)",
    )
    validate_sub = validate_parser.add_subparsers(dest="validate_command")

    og_parser = validate_sub.add_parser("orthogroups", help="Orthogroup selection and empirical validation")
    og_sub = og_parser.add_subparsers(dest="orthogroups_command")

    og_select = og_sub.add_parser("select", help="Select deterministic top-100 orthogroups")
    og_select.add_argument("--input", required=True)
    og_select.add_argument("--outdir", required=True)
    og_select.add_argument("--target-n", type=int, default=100)
    og_select.add_argument("--min-taxa", type=int, default=8)
    og_select.add_argument("--occupancy-threshold", type=float, default=0.7)
    og_select.add_argument("--min-length-nt", type=int, default=300)
    og_select.add_argument("--max-missingness", type=float, default=0.2)
    og_select.add_argument("--allow-duplicates", action="store_true")
    og_select.add_argument("--allow-internal-stops", action="store_true")
    og_select.set_defaults(func=cmd_validate_orthogroups_select)

    og_run = og_sub.add_parser("run", help="Run empirical validation over selected orthogroups")
    og_run.add_argument("--input", required=True)
    og_run.add_argument("--outdir", required=True)
    og_run.add_argument("--tree-calibration", action="store_true")
    og_run.add_argument("--n-calibration", type=int, default=200)
    og_run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    og_run.add_argument("--batch-size", type=int, default=1)
    og_run.add_argument("--seed", type=int, default=42)
    og_run.add_argument(
        "--foreground-mode",
        choices=["all-leaves", "all-branches", "user-list"],
        default="all-leaves",
    )
    og_run.add_argument("--foreground-list", default=None)
    og_run.add_argument("--alpha", type=float, default=0.05)
    og_run.add_argument(
        "--pvalue-mode",
        choices=["empirical_monte_carlo", "frozen_reference"],
        default="empirical_monte_carlo",
    )
    og_run.add_argument("--neutral-reps", type=int, default=200)
    og_run.add_argument("--min-neutral-group-size", type=int, default=20)
    og_run.add_argument("--sigma-floor", type=float, default=0.05)
    og_run.add_argument("--offline", action="store_true")
    og_run.add_argument("--overwrite", action="store_true")
    og_run.add_argument("--robustness-limit", type=int, default=10)
    og_run.set_defaults(func=cmd_validate_orthogroups_run)

    syn_parser = validate_sub.add_parser("synthetic", help="Synthetic benchmarking workflows")
    syn_sub = syn_parser.add_subparsers(dest="synthetic_command")
    syn_run = syn_sub.add_parser("run", help="Run simulator-driven synthetic validation")
    syn_run.add_argument("--simulator", required=True)
    syn_run.add_argument("--neutral-generator", default=None)
    syn_run.add_argument("--outdir", required=True)
    syn_run.add_argument("--grid-config", default=None)
    syn_run.add_argument("--tree-calibration", action="store_true")
    syn_run.add_argument("--n-calibration", type=int, default=200)
    syn_run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    syn_run.add_argument("--batch-size", type=int, default=1)
    syn_run.add_argument("--seed", type=int, default=42)
    syn_run.add_argument("--alpha", type=float, default=0.05)
    syn_run.add_argument(
        "--pvalue-mode",
        choices=["empirical_monte_carlo", "frozen_reference"],
        default="empirical_monte_carlo",
    )
    syn_run.add_argument("--neutral-reps", type=int, default=200)
    syn_run.add_argument("--min-neutral-group-size", type=int, default=20)
    syn_run.add_argument("--sigma-floor", type=float, default=0.05)
    syn_run.add_argument("--offline", action="store_true")
    syn_run.add_argument("--overwrite", action="store_true")
    syn_run.add_argument("--replicates-per-cell", type=int, default=2)
    syn_run.add_argument("--balance-target-per-regime", type=int, default=20)
    syn_run.add_argument("--max-replicates", type=int, default=500)
    syn_run.set_defaults(func=cmd_validate_synthetic_run)

    report_parser = validate_sub.add_parser("report", help="Generate unified validation report")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--outdir", required=True)
    report_parser.set_defaults(func=cmd_validate_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
