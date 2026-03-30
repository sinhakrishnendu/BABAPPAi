#!/usr/bin/env python3
"""Fast empirical rerun using precomputed codon alignments/trees.

This script is intended for strict calibration-asset synchronization checks:
- reruns `babappai run` (local upgraded code path) on existing alignments/trees
- pins an explicit cEII asset
- verifies runtime JSON asset/version/hash per gene
- writes per-gene JSON copies and summary TSV/CSV
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_table(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fieldnames:
                val = row.get(key)
                if isinstance(val, (dict, list)):
                    out[key] = json.dumps(val, sort_keys=True)
                elif val is None:
                    out[key] = ""
                elif isinstance(val, bool):
                    out[key] = "true" if val else "false"
                else:
                    out[key] = val
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-workdir", required=True, help="Work dir with per-gene precomputed files.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--ceii-asset", required=True)
    p.add_argument("--genes", required=True, help="Comma-separated gene list.")
    p.add_argument("--cache-dir", default=str(Path.home() / "Library/Caches/babappai"))
    p.add_argument("--n-calibration", type=int, default=20)
    p.add_argument("--neutral-reps", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_workdir = Path(args.source_workdir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ceii_asset = Path(args.ceii_asset).expanduser().resolve()
    genes = [g.strip() for g in str(args.genes).split(",") if g.strip()]
    if not genes:
        raise ValueError("No genes provided.")
    if not source_workdir.exists():
        raise FileNotFoundError(f"source-workdir not found: {source_workdir}")
    if not ceii_asset.exists():
        raise FileNotFoundError(f"cEII asset not found: {ceii_asset}")

    asset_payload = json.loads(ceii_asset.read_text())
    expected_version = str(asset_payload.get("calibration_version", "unknown"))
    expected_hash = _sha256(ceii_asset)

    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_log = outdir / "run_log.txt"
    run_log.write_text(
        f"source_workdir={source_workdir}\n"
        f"ceii_asset={ceii_asset}\n"
        f"ceii_asset_sha256={expected_hash}\n"
        f"expected_calibration_version={expected_version}\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["BABAPPAI_CACHE_DIR"] = str(Path(args.cache_dir).expanduser().resolve())
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HUGGINGFACE_HUB_OFFLINE"] = "1"

    summary_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for gene in genes:
        gene_dir = source_workdir / gene
        aln = gene_dir / f"{gene}.sanitized.codon.aln.fasta"
        tree = gene_dir / f"{gene}.sanitized.codon.aln.fasta.treefile"
        if not aln.exists() or not tree.exists():
            failures.append(f"{gene}: missing precomputed alignment/tree")
            summary_rows.append(
                {
                    "gene": gene,
                    "status": "failed",
                    "error": "missing precomputed alignment/tree",
                }
            )
            continue

        run_out = outdir / "runs" / gene
        run_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "babappai.cli",
            "run",
            "--alignment",
            str(aln),
            "--tree",
            str(tree),
            "--outdir",
            str(run_out),
            "--offline",
            "--device",
            str(args.device),
            "--pvalue-mode",
            "empirical_monte_carlo",
            "--n-calibration",
            str(int(args.n_calibration)),
            "--neutral-reps",
            str(int(args.neutral_reps)),
            "--overwrite",
            "--ceii-asset",
            str(ceii_asset),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        with run_log.open("a") as fh:
            fh.write(f"\n=== {gene} ===\n")
            fh.write("CMD: " + " ".join(cmd) + "\n")
            if proc.stdout:
                fh.write("STDOUT:\n" + proc.stdout + "\n")
            if proc.stderr:
                fh.write("STDERR:\n" + proc.stderr + "\n")
            fh.write(f"EXIT={proc.returncode}\n")
        if proc.returncode != 0:
            failures.append(f"{gene}: run command failed ({proc.returncode})")
            summary_rows.append(
                {
                    "gene": gene,
                    "status": "failed",
                    "error": f"run command failed ({proc.returncode})",
                }
            )
            continue

        result_json = run_out / "results.json"
        if not result_json.exists():
            failures.append(f"{gene}: missing results.json")
            summary_rows.append({"gene": gene, "status": "failed", "error": "missing results.json"})
            continue

        payload = json.loads(result_json.read_text())
        gs = payload.get("gene_summary", {})
        runtime_path = str(gs.get("ceii_asset_path") or payload.get("calibration", {}).get("ceii_asset_path") or "")
        runtime_ver = str(gs.get("calibration_version") or "")
        runtime_hash = str(gs.get("ceii_asset_sha256") or payload.get("calibration", {}).get("ceii_asset_sha256") or "")
        runtime_path_resolved = str(Path(runtime_path).expanduser().resolve()) if runtime_path else ""
        if runtime_path_resolved != str(ceii_asset):
            failures.append(
                f"{gene}: ceii_asset_path mismatch expected={ceii_asset} got={runtime_path_resolved or '<missing>'}"
            )
        if runtime_ver != expected_version:
            failures.append(f"{gene}: calibration_version mismatch expected={expected_version} got={runtime_ver}")
        if runtime_hash != expected_hash:
            failures.append(f"{gene}: ceii_asset_sha256 mismatch expected={expected_hash} got={runtime_hash or '<missing>'}")

        copied = outdir / f"{gene}.results.json"
        copied.write_text(json.dumps(payload, indent=2) + "\n")

        summary_rows.append(
            {
                "gene": gene,
                "status": "ok",
                "error": "",
                "results_json": str(copied),
                "calibration_version": runtime_ver,
                "ceii_asset_path": runtime_path_resolved,
                "ceii_asset_sha256": runtime_hash,
                "applicability_status": gs.get("applicability_status"),
                "eii_01_raw": gs.get("eii_01_raw"),
                "q_emp": gs.get("q_emp"),
                "ceii_gene": gs.get("ceii_gene"),
                "ceii_site": gs.get("ceii_site"),
                "ceii_gene_class": gs.get("ceii_gene_class"),
                "ceii_site_class": gs.get("ceii_site_class"),
                "calibration_unavailable_reason": gs.get("calibration_unavailable_reason"),
            }
        )

    fields = [
        "gene",
        "status",
        "error",
        "results_json",
        "calibration_version",
        "ceii_asset_path",
        "ceii_asset_sha256",
        "applicability_status",
        "eii_01_raw",
        "q_emp",
        "ceii_gene",
        "ceii_site",
        "ceii_gene_class",
        "ceii_site_class",
        "calibration_unavailable_reason",
    ]
    _write_table(outdir / "panel_summary.tsv", summary_rows, fields, "\t")
    _write_table(outdir / "panel_summary.csv", summary_rows, fields, ",")

    summary = {
        "outdir": str(outdir),
        "n_genes": len(genes),
        "n_ok": sum(1 for r in summary_rows if r.get("status") == "ok"),
        "n_failed": sum(1 for r in summary_rows if r.get("status") != "ok"),
        "requested_ceii_asset": str(ceii_asset),
        "requested_ceii_asset_sha256": expected_hash,
        "requested_calibration_version": expected_version,
        "failures": failures,
    }
    (outdir / "panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
