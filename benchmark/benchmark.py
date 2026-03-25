#!/usr/bin/env python3
import os, sys, json, subprocess, random, csv, platform
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ------------------------------------------------------------
# SAFETY (macOS OpenMP)
# ------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
N_SEEDS = 10
SEEDS = list(range(1, N_SEEDS + 1))

P_EPI_VALUES = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
NTAXA = 10
NCODONS = 400

TREE_SIM = "tree_sim.nwk"
TREE_EMP = "empirical.nwk"
EMP_FASTA = "empirical.fasta"

LOG_FILE = "benchmark_results.log"

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

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

def cliffs_delta(x, y):
    gt = sum(a > b for a in x for b in y)
    lt = sum(a < b for a in x for b in y)
    return (gt - lt) / (len(x) * len(y))

def sig_label(p, alpha=0.05):
    return "SIGNIFICANT" if p < alpha else "NOT SIGNIFICANT"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ------------------------------------------------------------
# ENV LOGGING
# ------------------------------------------------------------
def log_env():
    import babappaomega
    log("============================================================")
    log("BABAPPAΩ FULL BENCHMARK")
    log("============================================================")
    log(f"Timestamp           : {datetime.now().isoformat()}")
    log(f"Python              : {sys.version}")
    log(f"Platform            : {platform.platform()}")
    log(f"babappaomega ver.   : {babappaomega.__version__}")
    log("------------------------------------------------------------")

# ------------------------------------------------------------
# TREE GENERATION (SIMULATION)
# ------------------------------------------------------------
def ensure_sim_tree():
    if os.path.exists(TREE_SIM):
        return
    taxa = [f"taxon_{i}" for i in range(NTAXA)]
    newick = "(" + ",".join(f"{t}:1.0" for t in taxa) + ");"
    with open(TREE_SIM, "w") as f:
        f.write(newick)
    log(f"[INFO] Generated simulation tree: {TREE_SIM}")

# ------------------------------------------------------------
# SIMULATED ALIGNMENT
# ------------------------------------------------------------
def simulate_alignment(p_epi, seed, out_fasta):
    random.seed(seed)
    anc = [random.choice(CODONS) for _ in range(NCODONS)]
    records = []
    for t in range(NTAXA):
        seq = anc.copy()
        for i in range(NCODONS):
            if random.random() < p_epi:
                seq[i] = random.choice(CODONS)
        records.append(
            SeqRecord(Seq("".join(seq)), id=f"taxon_{t}", description="")
        )
    SeqIO.write(records, out_fasta, "fasta")

# ------------------------------------------------------------
# LOGIT EXTRACTION
# ------------------------------------------------------------
def extract_logits(json_path):
    with open(json_path) as f:
        d = json.load(f)
    x = np.array(d["site_logit_mean"])
    return x.mean(), x.var()

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    open(LOG_FILE, "w").close()
    log_env()
    ensure_sim_tree()

    if not os.path.exists(TREE_EMP):
        raise FileNotFoundError("empirical.nwk not found")
    if not os.path.exists(EMP_FASTA):
        raise FileNotFoundError("empirical.fasta not found")

    sim_rows = []
    emp_rows = []

    # ================= SIMULATION =================
    log("\n[SIMULATION] Scale sensitivity benchmark")
    for seed in SEEDS:
        for p in P_EPI_VALUES:
            fasta = f"sim_s{seed}_p{p}.fasta"
            out = f"sim_s{seed}_p{p}.json"

            simulate_alignment(p, seed, fasta)
            run([
                "babappaomega",
                "--alignment", fasta,
                "--tree", TREE_SIM,
                "--out", out,
                "--no-interpretation"
            ])
            m, v = extract_logits(out)
            sim_rows.append([seed, p, m, v])

    with open("simulation_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "p_epi", "mean_logit", "var_logit"])
        w.writerows(sim_rows)

    sim = np.array(sim_rows, dtype=float)
    pvals = np.unique(sim[:,1])
    mean_m = [sim[sim[:,1]==p][:,2].mean() for p in pvals]
    mean_v = [sim[sim[:,1]==p][:,3].mean() for p in pvals]

    rho_m, p_m = spearmanr(pvals, mean_m)
    rho_v, p_v = spearmanr(pvals, mean_v)

    log(f"Mean logit vs p_epi : Spearman rho = {rho_m:.3f}, p = {p_m:.3e} → {sig_label(p_m)}")
    log(f"Var  logit vs p_epi : Spearman rho = {rho_v:.3f}, p = {p_v:.3e} → {sig_label(p_v)}")

    # ================= EMPIRICAL =================
    log("\n[EMPIRICAL] Stability and sanity benchmark")
    for seed in SEEDS:
        random.seed(seed)

        orig = f"emp_orig_{seed}.json"
        run(["babappaomega","--alignment",EMP_FASTA,"--tree",TREE_EMP,"--out",orig,"--no-interpretation"])
        _, v0 = extract_logits(orig)

        recs = list(SeqIO.parse(EMP_FASTA, "fasta"))
        cols = list(zip(*[str(r.seq) for r in recs]))

        random.shuffle(cols)
        for i,r in enumerate(recs):
            r.seq = Seq("".join(c[i] for c in cols))
        SeqIO.write(recs,"_shuf.fasta","fasta")

        sh = f"emp_shuf_{seed}.json"
        run(["babappaomega","--alignment","_shuf.fasta","--tree",TREE_EMP,"--out",sh,"--no-interpretation"])
        _, vs = extract_logits(sh)

        boot = [random.choice(cols) for _ in cols]
        for i,r in enumerate(recs):
            r.seq = Seq("".join(b[i] for b in boot))
        SeqIO.write(recs,"_boot.fasta","fasta")

        bt = f"emp_boot_{seed}.json"
        run(["babappaomega","--alignment","_boot.fasta","--tree",TREE_EMP,"--out",bt,"--no-interpretation"])
        _, vb = extract_logits(bt)

        emp_rows.append([seed, v0, vs, vb])

    with open("empirical_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "orig_var", "shuffle_var", "bootstrap_var"])
        w.writerows(emp_rows)

    emp = np.array(emp_rows, dtype=float)
    orig, shuf, boot = emp[:,1], emp[:,2], emp[:,3]

    w_s = wilcoxon(orig, shuf)
    w_b = wilcoxon(orig, boot)

    delta_shuf = cliffs_delta(orig, shuf)
    delta_boot = cliffs_delta(orig, boot)

    log(f"Original vs Shuffle : Wilcoxon p = {w_s.pvalue:.3e}, Cliff's δ = {delta_shuf:.2f} → {sig_label(w_s.pvalue)}")
    log(f"Original vs Bootstrap: Wilcoxon p = {w_b.pvalue:.3e}, Cliff's δ = {delta_boot:.2f} → {sig_label(w_b.pvalue)}")

    log("\n[OK] Benchmark completed successfully")

if __name__ == "__main__":
    main()
