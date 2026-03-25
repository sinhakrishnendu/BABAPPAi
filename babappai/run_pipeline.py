"""Shared run execution and output writing helpers."""

from __future__ import annotations

import csv
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from babappai import __version__
from babappai.inference import run_inference
from babappai.interpret import interpret_results
from babappai.metadata import (
    MODEL_COMPATIBILITY_NOTE,
    MODEL_DOI,
    MODEL_FILE_NAME,
    MODEL_SHA256,
    SOFTWARE_NAME,
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_string(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_and_write_outputs(
    *,
    alignment_path: str,
    tree_path: str,
    outdir: Path,
    command: str,
    tree_calibration: bool,
    n_calibration: int,
    device: str,
    batch_size: int,
    seed: Optional[int],
    foreground_mode: str,
    foreground_list: Optional[str],
    offline: bool,
    overwrite: bool,
    neutral_generator_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "results_json": outdir / "results.json",
        "branch_summary_tsv": outdir / "branch_summary.tsv",
        "site_summary_tsv": outdir / "site_summary.tsv",
        "interpretation_txt": outdir / "interpretation.txt",
        "run_metadata_json": outdir / "run_metadata.json",
    }

    if not overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --overwrite to replace: "
                + ", ".join(str(path) for path in existing)
            )

    result = run_inference(
        alignment_path=alignment_path,
        tree_path=tree_path,
        model_tag="legacy_frozen",
        tree_calibration=tree_calibration,
        N_calibration=n_calibration,
        device=device,
        batch_size=batch_size,
        seed=seed,
        foreground_mode=foreground_mode,
        foreground_list_path=foreground_list,
        offline=offline,
    )

    gene = result["gene_level_identifiability"]
    mstatus = result["model_status"]

    payload = {
        "software_name": SOFTWARE_NAME,
        "software_version": __version__,
        "command": command,
        "timestamp": utc_timestamp(),
        "model": {
            "file_name": MODEL_FILE_NAME,
            "doi": MODEL_DOI,
            "sha256": MODEL_SHA256,
            "cached_path": mstatus["cached_path"],
            "verified": bool(mstatus["verified"]),
            "compatibility_note": MODEL_COMPATIBILITY_NOTE,
        },
        "input": {
            "alignment_path": str(Path(alignment_path).resolve()),
            "tree_path": str(Path(tree_path).resolve()),
            "n_sequences": int(result["n_sequences"]),
            "alignment_length": int(result["num_sites"]),
        },
        "calibration": {
            "enabled": bool(tree_calibration),
            "n_calibration": int(n_calibration),
            "source": gene.get("calibration_source"),
            "neutral_generator": neutral_generator_metadata,
        },
        "gene_summary": {
            "observed_variance": gene.get("observed_variance"),
            "neutral_expected_variance": gene.get("neutral_expected_variance"),
            "dispersion_ratio": gene.get("dispersion_ratio"),
            "EII_z": gene.get("EII_z"),
            "EII_01": gene.get("EII_01"),
            "identifiable_bool": gene.get("identifiable_bool"),
            "identifiability_extent": gene.get("identifiability_extent"),
        },
        "branch_results": result["branch_results"],
        "site_results": result["site_results"],
        "warnings": result.get("warnings", []),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    _write_json(outputs["results_json"], payload)

    branch_rows = []
    for row in payload["branch_results"]:
        branch_rows.append(
            {
                "branch": row["branch"],
                "background_score": row["background_score"],
                "logit_mean": row["logit_mean"],
                "is_terminal": row["is_terminal"],
                "selected_foreground": row["selected_foreground"],
                "EII_z": payload["gene_summary"]["EII_z"],
                "EII_01": payload["gene_summary"]["EII_01"],
                "identifiable_bool": payload["gene_summary"]["identifiable_bool"],
                "identifiability_extent": payload["gene_summary"]["identifiability_extent"],
            }
        )

    site_rows = []
    for row in payload["site_results"]:
        site_rows.append(
            {
                "site": row["site"],
                "site_score": row["site_score"],
                "site_logit_mean": row["site_logit_mean"],
                "site_logit_var": row["site_logit_var"],
                "EII_z": payload["gene_summary"]["EII_z"],
                "EII_01": payload["gene_summary"]["EII_01"],
                "identifiable_bool": payload["gene_summary"]["identifiable_bool"],
                "identifiability_extent": payload["gene_summary"]["identifiability_extent"],
            }
        )

    _write_tsv(
        outputs["branch_summary_tsv"],
        [
            "branch",
            "background_score",
            "logit_mean",
            "is_terminal",
            "selected_foreground",
            "EII_z",
            "EII_01",
            "identifiable_bool",
            "identifiability_extent",
        ],
        branch_rows,
    )

    _write_tsv(
        outputs["site_summary_tsv"],
        [
            "site",
            "site_score",
            "site_logit_mean",
            "site_logit_var",
            "EII_z",
            "EII_01",
            "identifiable_bool",
            "identifiability_extent",
        ],
        site_rows,
    )

    interpret_results(payload, out_path=outputs["interpretation_txt"])

    run_metadata = {
        "software_name": SOFTWARE_NAME,
        "software_version": __version__,
        "timestamp": utc_timestamp(),
        "command": command,
        "inputs": payload["input"],
        "calibration": payload["calibration"],
        "model": payload["model"],
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    _write_json(outputs["run_metadata_json"], run_metadata)

    return payload


def terminal_summary(payload: Dict[str, Any]) -> list[str]:
    gene = payload["gene_summary"]
    branches = sorted(
        payload["branch_results"],
        key=lambda row: float(row["background_score"]),
        reverse=True,
    )
    sites = sorted(
        payload["site_results"],
        key=lambda row: float(row["site_score"]),
        reverse=True,
    )

    branch_labels = ", ".join(
        f"{row['branch']} ({float(row['background_score']):.3f})"
        for row in branches[:3]
    ) or "n/a"
    site_labels = ", ".join(
        f"{int(row['site'])}:{float(row['site_score']):.3f}"
        for row in sites[:5]
    ) or "n/a"

    return [
        f"Gene-level EII_z: {float(gene['EII_z']):.2f}",
        f"Gene-level EII_01: {float(gene['EII_01']):.2f}",
        f"Identifiable: {'YES' if bool(gene['identifiable_bool']) else 'NO'}",
        f"Extent: {gene['identifiability_extent']}",
        f"Strongest branches: {branch_labels}",
        f"Top elevated sites: {site_labels}",
        "Warning: diagnostic identifiability score, not evidence of adaptive substitution",
    ]

