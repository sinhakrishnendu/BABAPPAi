import numpy as np
from Bio import SeqIO

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

CODON_TO_ID = {c: i for i, c in enumerate(CODONS)}
PAD_ID = len(CODONS)
VOCAB_SIZE = PAD_ID + 1


def encode_alignment(alignment_path):
    """
    Encode a codon alignment FASTA file for BABAPPAi inference.

    Parameters
    ----------
    alignment_path : str
        Path to codon alignment in FASTA format.

    Returns
    -------
    X : np.ndarray
        Encoded alignment of shape (ntaxa, L), dtype=int64.
    ntaxa : int
        Number of taxa.
    L : int
        Alignment length in codons.
    """
    records = list(SeqIO.parse(alignment_path, "fasta"))
    if not records:
        raise ValueError("Alignment file is empty")

    seqs = [str(r.seq).upper() for r in records]

    # enforce codon alignment
    for s in seqs:
        if len(s) % 3 != 0:
            raise ValueError("Alignment length is not divisible by 3")

    ntaxa = len(seqs)
    L = max(len(s) for s in seqs) // 3

    X = np.full((ntaxa, L), PAD_ID, dtype=np.int64)

    invalid_examples = []
    invalid_count = 0
    for i, s in enumerate(seqs):
        for j in range(0, len(s), 3):
            codon = s[j:j+3]
            if codon in CODON_TO_ID:
                X[i, j // 3] = CODON_TO_ID[codon]
                continue
            if codon == "---":
                X[i, j // 3] = PAD_ID
                continue
            invalid_count += 1
            if len(invalid_examples) < 10:
                invalid_examples.append(
                    f"{records[i].id}:site={j // 3 + 1}:codon={codon}"
                )

    if invalid_count > 0:
        raise ValueError(
            "Alignment contains unsupported codon triplets. Allowed: 61 sense codons and '---' PAD gap triplets. "
            f"invalid_triplets={invalid_count}; examples={'; '.join(invalid_examples)}"
        )

    return X, ntaxa, L
