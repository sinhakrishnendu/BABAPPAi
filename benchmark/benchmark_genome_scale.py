#!/usr/bin/env python3
"""
BABAPPAΩ genome-scale identifiability benchmark
==============================================

Correct per-gene inference benchmark.
"""

# --------------------------------------------------
# OpenMP safety (macOS / conda / torch)
# --------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# --------------------------------------------------
# Imports
# --------------------------------------------------
import json
import csv
import random
import subprocess
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
SEEDS = range(10)
N_GENES = 1000
GENE_LEN = 400
NTAXA = 12
TREE = "tree_sim.nwk"

P_EPI_REGIMES = {
    "neutral": 0.00,
    "low": 0.02,
    "medium": 0.05,
    "high": 0.10,
}

CODONS = [
    "TTT","TTC","TTA","TTG","CTT","CTC","CTA","CTG",
    "ATT","ATC","ATA","ATG","GTT","GTC","GTA","GTG",
    "TCT","TCC","TCA","TCG","CCT","CCC","CCA","CCG",
    "ACT","ACC","ACA","ACG","GCT","GCC","GCA","GCG",
    "TAT","TAC","CAT","CAC","CAA","CAG","AAT","AAC",
    "AAA","AAG","GAT","GAC","GAA","GAG","TGT","TGC",
    "TGG","CGT","CGC","CGA","CGG","AGT","AGC","AGA",
    "AGG","GGT","GGC","GGA","GGG"
]

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
LOG = open("genome_scale.log", "w")
def log(msg):
    print(msg)
    LOG.write(msg + "\n")

log("="*60)
log("BABAPPAΩ GENOME-SCALE BENCHMARK")
log("="*60)
log(f"Timestamp : {datetime.now().isoformat()}")
log(f"Genes     : {N_GENES}")
log(f"Codons/g  : {GENE_LEN}")
log(f"Taxa      : {NTAXA}")
log("")

# --------------------------------------------------
# TREE (STAR)
# --------------------------------------------------
if not os.path.exists(TREE):
    with open(TREE, "w") as f:
        leaves = ",".join(f"t{i}:1.0" for i in range(NTAXA))
        f.write(f"({leaves});\n")
    log(f"[INFO] Generated tree: {TREE}")

# --------------------------------------------------
# SIMULATE ONE GENE
# --------------------------------------------------
def simulate_gene(p_epi, seed, gid):
    random.seed(seed + gid)
    np.random.seed(seed + gid)

    anc = [random.choice(CODONS) for _ in range(GENE_LEN)]
    epi = np.random.binomial(1, p_epi, GENE_LEN)
    true_burden = epi.mean()

    records = []
    for t in range(NTAXA):
        seq = anc.copy()
        for i in range(GENE_LEN):
            if epi[i]:
                seq[i] = random.choice(CODONS)
        records.append(
            SeqRecord(
                Seq("".join(seq)),
                id=f"gene{gid}_taxon{t}",
                description=""
            )
        )

    fasta = f"gene_s{seed}_g{gid}_p{p_epi}.fasta"
    SeqIO.write(records, fasta, "fasta")
    return fasta, true_burden

# --------------------------------------------------
# RUN BABAPPAΩ
# --------------------------------------------------
def infer_gene(fasta, out_json):
    subprocess.run(
        [
            "babappaomega",
            "--alignment", fasta,
            "--tree", TREE,
            "--out", out_json,
            "--no-interpretation"
        ],
        check=True,
        stdout=subprocess.DEVNULL
    )

# --------------------------------------------------
# MAIN BENCHMARK
# --------------------------------------------------
rows = []

for label, p in P_EPI_REGIMES.items():
    log(f"[RUN] Regime: {label} (p_epi={p})")

    for seed in SEEDS:
        true_vals = []
        pred_vals = []

        for g in range(N_GENES):
            fasta, true_b = simulate_gene(p, seed, g)
            out = f"{fasta}.json"
            infer_gene(fasta, out)

            with open(out) as f:
                pred = np.array(json.load(f)["site_logit_mean"])

            pred_b = pred.mean()

            true_vals.append(true_b)
            pred_vals.append(pred_b)

        if np.var(true_vals) == 0:
            rho = np.nan
        else:
            rho, _ = spearmanr(true_vals, pred_vals)

        rows.append([label, p, seed, rho])

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
with open("genome_scale_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["regime", "p_epi", "seed", "spearman_rho"])
    w.writerows(rows)

summary = {}
for r in rows:
    summary.setdefault(r[0], []).append(r[3])

with open("genome_scale_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["regime", "mean_spearman_rho"])
    for k, v in summary.items():
        v = [x for x in v if not np.isnan(x)]
        w.writerow([k, np.mean(v) if v else "NaN"])

log("")
log("[OK] Genome-scale benchmark completed successfully")
LOG.close()
