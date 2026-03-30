#!/usr/bin/env python3
"""Assert that empirical result JSONs all used the requested cEII calibration asset/version."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gene",
        "calibration_version",
        "ceii_asset_path",
        "ceii_asset_hash",
        "applicability_status",
        "ceii_gene",
        "ceii_site",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _get_value(payload: Dict[str, Any], key: str) -> Any:
    if key in payload.get("gene_summary", {}):
        return payload["gene_summary"].get(key)
    if key in payload.get("calibration", {}):
        return payload["calibration"].get(key)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, help="Directory containing *.results.json files.")
    p.add_argument("--requested-ceii-asset", required=True, help="Path to expected calibration asset.")
    p.add_argument(
        "--requested-calibration-version",
        default=None,
        help="Expected calibration_version. If omitted, derived from requested asset JSON.",
    )
    p.add_argument(
        "--out-tsv",
        default=None,
        help="Output TSV path (defaults to <results-dir>/ceii_asset_consistency_audit.tsv).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    requested_asset = Path(args.requested_ceii_asset).expanduser().resolve()
    if not requested_asset.exists():
        raise FileNotFoundError(f"Requested cEII asset does not exist: {requested_asset}")

    requested_hash = _sha256(requested_asset)
    requested_payload = json.loads(requested_asset.read_text())
    expected_version = (
        str(args.requested_calibration_version)
        if args.requested_calibration_version is not None
        else str(requested_payload.get("calibration_version", "unknown"))
    )
    if not expected_version:
        raise ValueError("Could not determine expected calibration_version.")

    json_paths = sorted(results_dir.glob("*.results.json"))
    if not json_paths:
        raise FileNotFoundError(f"No *.results.json files found in {results_dir}")

    rows: List[Dict[str, Any]] = []
    mismatches: List[str] = []
    seen_paths: set[str] = set()
    seen_versions: set[str] = set()
    seen_hashes: set[str] = set()

    for path in json_paths:
        payload = json.loads(path.read_text())
        gene = path.stem.replace(".results", "")
        cal_version = str(_get_value(payload, "calibration_version") or "")
        asset_path_raw = _get_value(payload, "ceii_asset_path")
        asset_hash_raw = _get_value(payload, "ceii_asset_sha256")
        app_status = str(_get_value(payload, "applicability_status") or "")
        ceii_gene = _get_value(payload, "ceii_gene")
        ceii_site = _get_value(payload, "ceii_site")

        if not asset_path_raw:
            mismatches.append(f"{gene}: missing ceii_asset_path")
            asset_path = ""
        else:
            asset_path = str(Path(str(asset_path_raw)).expanduser().resolve())

        if not asset_hash_raw:
            mismatches.append(f"{gene}: missing ceii_asset_sha256")
            asset_hash = ""
        else:
            asset_hash = str(asset_hash_raw)

        if cal_version != expected_version:
            mismatches.append(
                f"{gene}: calibration_version mismatch expected={expected_version} got={cal_version}"
            )
        if asset_path and asset_path != str(requested_asset):
            mismatches.append(
                f"{gene}: ceii_asset_path mismatch expected={requested_asset} got={asset_path}"
            )
        if asset_hash and asset_hash != requested_hash:
            mismatches.append(
                f"{gene}: ceii_asset_sha256 mismatch expected={requested_hash} got={asset_hash}"
            )

        if asset_path:
            seen_paths.add(asset_path)
        if cal_version:
            seen_versions.add(cal_version)
        if asset_hash:
            seen_hashes.add(asset_hash)

        rows.append(
            {
                "gene": gene,
                "calibration_version": cal_version,
                "ceii_asset_path": asset_path,
                "ceii_asset_hash": asset_hash,
                "applicability_status": app_status,
                "ceii_gene": ceii_gene,
                "ceii_site": ceii_site,
            }
        )

    out_tsv = (
        Path(args.out_tsv).expanduser().resolve()
        if args.out_tsv
        else (results_dir / "ceii_asset_consistency_audit.tsv")
    )
    _write_tsv(out_tsv, rows)

    if len(seen_paths) > 1:
        mismatches.append(f"Multiple ceii_asset_path values observed: {sorted(seen_paths)}")
    if len(seen_versions) > 1:
        mismatches.append(f"Multiple calibration_version values observed: {sorted(seen_versions)}")
    if len(seen_hashes) > 1:
        mismatches.append(f"Multiple ceii_asset_sha256 values observed: {sorted(seen_hashes)}")

    summary = {
        "results_dir": str(results_dir),
        "n_results": len(rows),
        "requested_ceii_asset": str(requested_asset),
        "requested_ceii_asset_sha256": requested_hash,
        "requested_calibration_version": expected_version,
        "audit_tsv": str(out_tsv),
        "mismatches": mismatches,
    }
    summary_path = out_tsv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if mismatches:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

