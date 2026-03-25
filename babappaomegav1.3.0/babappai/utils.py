# babappai/utils.py
# ============================================================
# Core utilities for BABAPPAi
# ============================================================

from Bio import Phylo, SeqIO
from io import StringIO
import torch
import numpy as np


# ============================================================
# DEVICE RESOLUTION
# ============================================================

def resolve_device(device=None):
    """
    Resolve execution device for BABAPPAi inference.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device in ("cpu", "cuda"):
        return torch.device(device)

    raise ValueError(
        f"Invalid device '{device}'. Expected 'auto', 'cpu', or 'cuda'."
    )


# ============================================================
# TREE PARSING WITH STABLE BRANCH IDS
# ============================================================

def parse_tree(newick):
    """
    Parse a Newick tree and return:
    - tree object
    - stable branch identifiers
    - mapping from branch index → descendant taxa
    """

    handle = StringIO(newick)
    tree = Phylo.read(handle, "newick")

    branches = []
    branch_to_taxa = {}
    internal_counter = 0

    for clade in tree.find_clades(order="preorder"):
        if clade.branch_length is None:
            continue

        # stable branch ID
        if clade.name:
            bid = clade.name
        else:
            bid = f"node_{internal_counter}"
            internal_counter += 1

        branches.append(bid)

        # collect descendant taxa (leaf names)
        taxa = [t.name for t in clade.get_terminals()]
        branch_to_taxa[bid] = taxa

    return tree, branches, branch_to_taxa


# ============================================================
# ALIGNMENT LOADING
# ============================================================

def load_alignment(alignment_path):
    """
    Load a codon alignment from FASTA.
    """
    records = SeqIO.parse(alignment_path, "fasta")
    alignment = {}

    for rec in records:
        seq = str(rec.seq).upper()
        if len(seq) % 3 != 0:
            raise ValueError(
                f"Sequence length not divisible by 3 for {rec.id}"
            )
        alignment[rec.id] = seq

    if not alignment:
        raise ValueError("Empty alignment")

    return alignment


# ============================================================
# BRANCH-STRATIFIED, LIKELIHOOD-FREE ENCODER
# ============================================================

def encode_parent_child_from_alignment(alignment, branches, branch_to_taxa):
    """
    Biologically meaningful, likelihood-free encoding.

    Core idea
    ---------
    - Parent = per-site global consensus codon
    - Child  = branch-specific codon disagreement
    - Episodic signal = uneven distribution of mismatches across branches

    This unlocks the model's learned branch-aware representation
    WITHOUT ancestral reconstruction.
    """

    taxa = list(alignment.keys())
    seqs = {t: alignment[t] for t in taxa}
    ntaxa = len(taxa)

    L = len(next(iter(seqs.values()))) // 3
    K = len(branches)

    # --------------------------------------------------------
    # GLOBAL CONSENSUS CODON (SITE-WISE)
    # --------------------------------------------------------

    consensus = []
    for i in range(L):
        codons = [seqs[t][3*i:3*i+3] for t in taxa]
        counts = {}
        for c in codons:
            counts[c] = counts.get(c, 0) + 1
        consensus.append(max(counts, key=counts.get))

    # --------------------------------------------------------
    # BRANCH-STRATIFIED DISAGREEMENT
    # --------------------------------------------------------

    parent = torch.zeros((1, K, L), dtype=torch.long)
    child  = torch.zeros((1, K, L), dtype=torch.long)

    for b, bid in enumerate(branches):
        desc = branch_to_taxa[bid]
        if not desc:
            continue

        for i in range(L):
            mismatches = sum(
                seqs[t][3*i:3*i+3] != consensus[i]
                for t in desc
            )
            parent[0, b, i] = len(desc) - mismatches
            child[0, b, i]  = mismatches

    # --------------------------------------------------------
    # BRANCH LENGTHS (WEAK WEIGHTS)
    # --------------------------------------------------------

    branch_length = torch.ones((1, K), dtype=torch.float32)

    return parent, child, branch_length
