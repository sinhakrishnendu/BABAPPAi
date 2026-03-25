# babappai/tree_calibration.py
# ============================================================
# TREE-CONDITIONAL MONTE CARLO CALIBRATION
# ============================================================

import numpy as np
import tempfile
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from babappai.tree_neutral import simulate_neutral_alignment
from babappai.metadata import MODEL_TAG


def monte_carlo_neutral(
    tree,
    L,
    inference_function,
    model_tag=MODEL_TAG,
    N=200,
    offline=False,
    foreground_mode="all-leaves",
    foreground_list_path=None,
    batch_size=1,
):
    """
    Perform tree-conditional Monte Carlo calibration.
    """

    simulated_variances = []

    for r in range(N):

        sequences = simulate_neutral_alignment(tree, L, seed=r)

        with tempfile.NamedTemporaryFile(suffix=".fasta") as tmp:

            records = [
                SeqRecord(Seq(seq), id=name, description="")
                for name, seq in sequences.items()
            ]
            SeqIO.write(records, tmp.name, "fasta")

            result = inference_function(
                alignment_path=tmp.name,
                tree_obj=tree,
                model_tag=model_tag,
                tree_calibration=False,  # avoid recursion
                offline=offline,
                seed=r,
                foreground_mode=foreground_mode,
                foreground_list_path=foreground_list_path,
                batch_size=batch_size,
            )

        simulated_variances.append(
            result["gene_level_identifiability"]["observed_variance"]
        )

    simulated_variances = np.array(simulated_variances)

    mu0 = simulated_variances.mean()
    sd0 = simulated_variances.std(ddof=1)

    return mu0, sd0, simulated_variances
