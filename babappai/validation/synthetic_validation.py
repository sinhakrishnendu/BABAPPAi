"""Synthetic validation workflow driven by external simulator adapters."""

from __future__ import annotations

import csv
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from babappai.calibration.neutral_generator_adapter import run_neutral_generator
from babappai.run_pipeline import run_and_write_outputs
from babappai.validation.simulator_adapter import run_simulator


REGIMES = [
    "not_identifiable",
    "weak_or_ambiguous",
    "identifiable",
    "strongly_identifiable",
]


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _default_grid() -> Dict[str, List[Any]]:
    return {
        "n_taxa": [8, 16, 32],
        "alignment_length": [300, 900, 1800],
        "perturbation_sparsity": [0.02, 0.10],
        "perturbation_magnitude": [0.5, 1.5],
        "branch_length_scale": [0.5, 1.0],
        "recombination_rate": [0.0, 0.05],
        "alignment_noise": [0.0, 0.02],
    }


def _load_grid(config_path: Optional[str]) -> Dict[str, List[Any]]:
    if not config_path:
        return _default_grid()
    config = json.loads(Path(config_path).read_text())
    return dict(config.get("grid", _default_grid()))


def _iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = sorted(grid.keys())
    values = [grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield {k: v for k, v in zip(keys, combo)}


def run_synthetic_validation(
    *,
    simulator_path: str,
    neutral_generator_path: Optional[str],
    outdir: str,
    seed: int,
    tree_calibration: bool,
    n_calibration: int,
    device: str,
    batch_size: int,
    offline: bool,
    overwrite: bool,
    grid_config: Optional[str] = None,
    replicates_per_cell: int = 2,
    balance_target_per_regime: int = 20,
    max_replicates: int = 500,
) -> Dict[str, Any]:
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    neutral_meta = None
    if neutral_generator_path:
        neutral_meta = run_neutral_generator(
            generator_path=neutral_generator_path,
            output_dir=str(out / "neutral_generator"),
            model_tag="legacy_frozen",
            seed=seed,
            extra_args=[],
        )

    grid = _load_grid(grid_config)
    grid_rows = list(_iter_grid(grid))
    replicate_rows: List[Dict[str, Any]] = []
    regime_counts = defaultdict(int)

    replicate_index = 0
    for params in grid_rows:
        for rep in range(replicates_per_cell):
            if replicate_index >= max_replicates:
                break

            run_seed = seed + replicate_index
            replicate_id = f"rep_{replicate_index:05d}"
            start = time.perf_counter()

            sim_meta = run_simulator(
                simulator_path=simulator_path,
                outdir=str(out / "simulator_runs"),
                replicate_id=replicate_id,
                seed=run_seed,
                params=params,
            )

            inference_payload = run_and_write_outputs(
                alignment_path=sim_meta["alignment_path"],
                tree_path=sim_meta["tree_path"],
                outdir=out / "inference_runs" / replicate_id,
                command=(
                    "babappai validate synthetic run "
                    f"--simulator {simulator_path} --outdir {outdir}"
                ),
                tree_calibration=tree_calibration,
                n_calibration=n_calibration,
                device=device,
                batch_size=batch_size,
                seed=run_seed,
                foreground_mode="all-leaves",
                foreground_list=None,
                offline=offline,
                overwrite=overwrite,
                neutral_generator_metadata=neutral_meta,
            )
            runtime_sec = time.perf_counter() - start

            gene = inference_payload["gene_summary"]
            regime = gene["identifiability_extent"]
            regime_counts[regime] += 1

            replicate_rows.append(
                {
                    "replicate_id": replicate_id,
                    "seed": run_seed,
                    "simulator_path": sim_meta["simulator_path"],
                    "simulator_params_json": json.dumps(params, sort_keys=True),
                    "alignment_path": sim_meta["alignment_path"],
                    "tree_path": sim_meta["tree_path"],
                    "truth_metadata_json": json.dumps(sim_meta["truth_metadata"], sort_keys=True),
                    "neutral_generator_path": neutral_meta["generator_path"] if neutral_meta else "",
                    "neutral_generator_reference_file": neutral_meta["reference_file"] if neutral_meta else "",
                    "EII_z": gene["EII_z"],
                    "EII_01": gene["EII_01"],
                    "identifiable_bool": gene["identifiable_bool"],
                    "identifiability_extent": regime,
                    "runtime_seconds": runtime_sec,
                    "warnings_json": json.dumps(inference_payload.get("warnings", [])),
                    "results_json": str(out / "inference_runs" / replicate_id / "results.json"),
                }
            )

            replicate_index += 1
            if (
                balance_target_per_regime > 0
                and all(regime_counts[label] >= balance_target_per_regime for label in REGIMES)
            ):
                break
        if (
            balance_target_per_regime > 0
            and all(regime_counts[label] >= balance_target_per_regime for label in REGIMES)
        ):
            break
        if replicate_index >= max_replicates:
            break

    _write_tsv(out / "synthetic_replicates.tsv", replicate_rows)
    (out / "synthetic_replicates.json").write_text(
        json.dumps(replicate_rows, indent=2) + "\n"
    )

    summary = {
        "n_replicates": len(replicate_rows),
        "regime_counts": {label: regime_counts[label] for label in REGIMES},
        "balance_target_per_regime": balance_target_per_regime,
        "max_replicates": max_replicates,
        "grid": grid,
        "replicates_per_cell": replicates_per_cell,
        "neutral_generator_metadata": neutral_meta,
        "provenance_note": (
            "BABAPPAi is the renamed continuation of the BABAPPAΩ codebase."
        ),
    }
    (out / "synthetic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

