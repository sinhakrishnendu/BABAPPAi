# babappai/tree_neutral.py
# ============================================================
# TREE-CONDITIONAL NEUTRAL SIMULATION
# ============================================================

import numpy as np

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

NC = len(CODONS)


def simulate_neutral_alignment(tree, L, seed=None):
    """
    Simulate neutral codon evolution on a given tree.
    ω = 1, simple mutation model preserving ancestry.
    """

    if seed is not None:
        np.random.seed(seed)

    root_codons = np.random.randint(0, NC, size=L)

    for node in tree.traverse("preorder"):
        if node.is_root():
            node.codons = root_codons.copy()
        else:
            parent = node.up
            t = node.dist if node.dist else 1.0

            mutation_prob = 1 - np.exp(-t)

            new_codons = parent.codons.copy()
            mutate_mask = np.random.rand(L) < mutation_prob
            new_codons[mutate_mask] = np.random.randint(0, NC, mutate_mask.sum())

            node.codons = new_codons

    sequences = {}
    for leaf in tree.get_leaves():
        sequences[leaf.name] = "".join(
            CODONS[i] for i in leaf.codons
        )

    return sequences
