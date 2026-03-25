#!/usr/bin/env python3
"""Reference synthetic simulator for BABAPPAi validation grids.

This lightweight simulator is intended as an adapter-friendly stand-in for the
training simulator interface. It emits:
- alignment.fasta
- tree.nwk
- truth_metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


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
    parser.add_argument("--n-taxa", type=int, default=16)
    parser.add_argument("--alignment-length", type=int, default=900)
    parser.add_argument("--perturbation-sparsity", type=float, default=0.05)
    parser.add_argument("--perturbation-magnitude", type=float, default=1.0)
    parser.add_argument("--branch-length-scale", type=float, default=1.0)
    parser.add_argument("--recombination-rate", type=float, default=0.0)
    parser.add_argument("--alignment-noise", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    n_taxa = max(4, int(args.n_taxa))
    n_codons = max(10, int(args.alignment_length // 3))

    # Simulate background consensus codon trajectory.
    root_codons = [rng.choice(CODONS) for _ in range(n_codons)]

    records = []
    perturbed_taxa = set(rng.sample(range(n_taxa), k=max(1, n_taxa // 4)))
    for t in range(n_taxa):
        codons = root_codons.copy()

        # Baseline branch divergence controlled by branch-length scale.
        base_mut = min(0.95, 0.01 * args.branch_length_scale)
        for i in range(n_codons):
            if rng.random() < base_mut:
                codons[i] = rng.choice(CODONS)

        # Episodic perturbation on selected taxa.
        if t in perturbed_taxa:
            p = min(0.95, args.perturbation_sparsity * args.perturbation_magnitude)
            for i in range(n_codons):
                if rng.random() < p:
                    codons[i] = rng.choice(CODONS)

        seq = list("".join(codons))

        # Optional alignment noise.
        if args.alignment_noise > 0:
            for i, ch in enumerate(seq):
                if rng.random() < args.alignment_noise:
                    seq[i] = rng.choice(["A", "C", "G", "T"])

        records.append(
            SeqRecord(Seq("".join(seq)), id=f"taxon{t+1:02d}", description="")
        )

    aln_path = outdir / "alignment.fasta"
    SeqIO.write(records, aln_path, "fasta")

    leaves = ",".join(
        f"taxon{t+1:02d}:{0.05 + 0.05 * args.branch_length_scale:.3f}"
        for t in range(n_taxa)
    )
    tree_path = outdir / "tree.nwk"
    tree_path.write_text(f"({leaves});\n")

    truth = {
        "seed": args.seed,
        "n_taxa": n_taxa,
        "alignment_length_nt": n_codons * 3,
        "perturbation_sparsity": args.perturbation_sparsity,
        "perturbation_magnitude": args.perturbation_magnitude,
        "branch_length_scale": args.branch_length_scale,
        "recombination_rate": args.recombination_rate,
        "alignment_noise": args.alignment_noise,
        "perturbed_taxa": sorted(f"taxon{t+1:02d}" for t in perturbed_taxa),
    }
    (outdir / "truth_metadata.json").write_text(json.dumps(truth, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
