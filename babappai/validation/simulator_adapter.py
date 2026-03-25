"""Adapter layer for external simulator scripts used in synthetic validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _discover_single_file(root: Path, patterns: list[str]) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    files = [p for p in matches if p.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No files matching {patterns} were found in simulator output: {root}"
        )
    files.sort()
    return files[0]


def run_simulator(
    *,
    simulator_path: str,
    outdir: str,
    replicate_id: str,
    seed: int,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run simulator script and return generated alignment/tree plus provenance."""

    script = Path(simulator_path).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"Simulator script not found: {script}")

    rep_dir = Path(outdir).expanduser().resolve() / replicate_id
    rep_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(script),
        "--outdir",
        str(rep_dir),
        "--seed",
        str(seed),
        "--n-taxa",
        str(params.get("n_taxa")),
        "--alignment-length",
        str(params.get("alignment_length")),
        "--perturbation-sparsity",
        str(params.get("perturbation_sparsity")),
        "--perturbation-magnitude",
        str(params.get("perturbation_magnitude")),
        "--branch-length-scale",
        str(params.get("branch_length_scale")),
        "--recombination-rate",
        str(params.get("recombination_rate", 0.0)),
        "--alignment-noise",
        str(params.get("alignment_noise", 0.0)),
    ]

    first = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=str(rep_dir),
        check=False,
    )

    attempted_commands = [command]
    result = first

    if first.returncode != 0:
        fallback = [sys.executable, str(script)]
        attempted_commands.append(fallback)
        second = subprocess.run(
            fallback,
            text=True,
            capture_output=True,
            cwd=str(rep_dir),
            check=False,
        )
        result = second

    if result.returncode != 0:
        raise RuntimeError(
            "Simulator failed. "
            f"Attempted commands: {attempted_commands}. "
            f"stderr: {result.stderr.strip()}"
        )

    alignment = _discover_single_file(rep_dir, ["*.fasta", "*.fa", "*.fas"])
    tree = _discover_single_file(rep_dir, ["*.nwk", "*.newick", "*.tree"])

    truth_metadata = {}
    truth_file = rep_dir / "truth_metadata.json"
    if truth_file.exists():
        truth_metadata = json.loads(truth_file.read_text())

    metadata = {
        "simulator_path": str(script),
        "attempted_commands": attempted_commands,
        "replicate_id": replicate_id,
        "seed": seed,
        "params": params,
        "alignment_path": str(alignment),
        "tree_path": str(tree),
        "truth_metadata": truth_metadata,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (rep_dir / "simulator_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata

