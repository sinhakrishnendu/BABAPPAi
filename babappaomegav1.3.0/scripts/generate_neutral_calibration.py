#!/usr/bin/env python3
"""Generate neutral calibration reference tables for BABAPPAi."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from babappai.inference import run_inference


CODONS = [
    "TTT", "TTC", "TTA", "TTG", "CTT", "CTC", "CTA", "CTG",
    "ATT", "ATC", "ATA", "ATG", "GTT", "GTC", "GTA", "GTG",
    "TCT", "TCC", "TCA", "TCG", "CCT", "CCC", "CCA", "CCG",
    "ACT", "ACC", "ACA", "ACG", "GCT", "GCC", "GCA", "GCG",
    "TAT", "TAC", "CAT", "CAC", "CAA", "CAG", "AAT", "AAC",
    "AAA", "AAG", "GAT", "GAC", "GAA", "GAG", "TGT", "TGC",
    "TGG", "CGT", "CGC", "CGA", "CGG", "AGT", "AGC", "AGA",
    "AGG", "GGT", "GGC", "GGA", "GGG",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-replicates", type=int, default=30)
    parser.add_argument("--model-tag", default="legacy_frozen")
    parser.add_argument("--length-grid", default="200,400,800,1000")
    parser.add_argument("--taxa-grid", default="8,16,32,48,64")
    return parser.parse_args()


def _simulate_neutral_alignment(path: Path, n_taxa: int, n_codons: int, seed: int) -> None:
    rng = random.Random(seed)
    records = []
    for t in range(n_taxa):
        seq = "".join(rng.choice(CODONS) for _ in range(n_codons))
        records.append(SeqRecord(Seq(seq), id=f"t{t+1}", description=""))
    SeqIO.write(records, path, "fasta")


def _write_star_tree(path: Path, n_taxa: int) -> None:
    leaves = ",".join(f"t{t+1}:1.0" for t in range(n_taxa))
    path.write_text(f"({leaves});\n")


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    lengths = [int(x) for x in args.length_grid.split(",") if x]
    taxa_counts = [int(x) for x in args.taxa_grid.split(",") if x]

    table = {}
    for L in lengths:
        for K in taxa_counts:
            variances = []
            for rep in range(args.n_replicates):
                rep_dir = outdir / f"tmp_L{L}_K{K}_R{rep}"
                rep_dir.mkdir(parents=True, exist_ok=True)
                aln = rep_dir / "alignment.fasta"
                tree = rep_dir / "tree.nwk"

                _simulate_neutral_alignment(aln, n_taxa=K, n_codons=L, seed=args.seed + rep)
                _write_star_tree(tree, K)

                result = run_inference(
                    alignment_path=str(aln),
                    tree_path=str(tree),
                    model_tag=args.model_tag,
                    tree_calibration=False,
                    seed=args.seed + rep,
                    offline=False,
                )
                variances.append(result["gene_level_identifiability"]["observed_variance"])

            table[f"L_{L}_K_{K}"] = {
                "sigma2_mean": float(np.mean(variances)),
                "sigma2_sd": float(np.std(variances, ddof=1)) if len(variances) > 1 else 0.0,
            }

    out_file = outdir / "neutral_reference_frozen.json"
    out_file.write_text(json.dumps(table, indent=2) + "\n")

    metadata = {
        "seed": args.seed,
        "n_replicates": args.n_replicates,
        "model_tag": args.model_tag,
        "lengths": lengths,
        "taxa_counts": taxa_counts,
        "output_file": str(out_file),
    }
    (outdir / "neutral_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"[OK] Wrote neutral reference table to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
