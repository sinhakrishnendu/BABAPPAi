"""Full-pipeline empirical validation utilities for EII threshold calibration.

This module extends BABAPPAi validation without altering core inference logic.
It treats EII as a diagnostic of recoverability/identifiability, not direct proof
of adaptation.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from babappai.run_pipeline import run_and_write_outputs
from babappai.stats import annotate_bh_qvalues, bh_adjust


CODONS: List[str] = [
    "TTT", "TTC", "TTA", "TTG", "CTT", "CTC", "CTA", "CTG",
    "ATT", "ATC", "ATA", "ATG", "GTT", "GTC", "GTA", "GTG",
    "TCT", "TCC", "TCA", "TCG", "CCT", "CCC", "CCA", "CCG",
    "ACT", "ACC", "ACA", "ACG", "GCT", "GCC", "GCA", "GCG",
    "TAT", "TAC", "CAT", "CAC", "CAA", "CAG", "AAT", "AAC",
    "AAA", "AAG", "GAT", "GAC", "GAA", "GAG", "TGT", "TGC",
    "TGG", "CGT", "CGC", "CGA", "CGG", "AGT", "AGC", "AGA",
    "AGG", "GGT", "GGC", "GGA", "GGG",
]

NUCLEOTIDES = np.array(["A", "C", "G", "T"], dtype="U1")


# Regime specification kept faithful to existing neutral/low/medium/high framing.
REGIME_SPECS: Dict[str, Dict[str, float]] = {
    "neutral": {
        "burden_min": 0.00,
        "burden_max": 0.00,
        "effect_min": 0.00,
        "effect_max": 0.00,
        "branch_frac_min": 0.00,
        "branch_frac_max": 0.00,
    },
    "low": {
        "burden_min": 0.01,
        "burden_max": 0.05,
        "effect_min": 0.04,
        "effect_max": 0.10,
        "branch_frac_min": 0.10,
        "branch_frac_max": 0.25,
    },
    "medium": {
        "burden_min": 0.06,
        "burden_max": 0.15,
        "effect_min": 0.10,
        "effect_max": 0.20,
        "branch_frac_min": 0.20,
        "branch_frac_max": 0.40,
    },
    "high": {
        "burden_min": 0.16,
        "burden_max": 0.35,
        "effect_min": 0.20,
        "effect_max": 0.35,
        "branch_frac_min": 0.30,
        "branch_frac_max": 0.60,
    },
}


LENGTH_BINS: Dict[str, Tuple[int, int]] = {
    "short": (240, 480),
    "medium": (720, 1200),
    "long": (1500, 2400),
}

TREE_BINS: Dict[str, Dict[str, float]] = {
    "small_shallow": {"n_taxa": 8, "depth_scale": 0.6},
    "medium": {"n_taxa": 16, "depth_scale": 1.0},
    "large_deep": {"n_taxa": 24, "depth_scale": 1.6},
}

RECOMBINATION_BINS: Dict[str, float] = {
    "none": 0.00,
    "moderate": 0.03,
    "high": 0.08,
}

ALIGNMENT_NOISE_BINS: Dict[str, float] = {
    "none": 0.00,
    "mild": 0.005,
    "high": 0.02,
}

RATE_HETEROGENEITY_BINS: Dict[str, float] = {
    "low": 0.10,
    "moderate": 0.35,
    "high": 0.70,
}

DEFAULT_DISPERSION_CHOICES = ("site_logit_variance", "site_score_variance")


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    regime: str
    gene_length_bin: str
    tree_bin: str
    recombination_bin: str
    alignment_noise_bin: str
    mutation_rate_heterogeneity_bin: str
    dispersion_statistic: str
    n_taxa: int
    depth_scale: float
    gene_length_nt: int
    recombination_rate: float
    alignment_noise_rate: float
    rate_heterogeneity_sigma: float
    burden_fraction: float
    branch_fraction: float
    burden_effect: float
    base_mutation_rate: float
    scenario_seed: int


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy compatibility helper (np.trapezoid on new versions, np.trapz older)."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_tsv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open() as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t") if row]


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# Tree + sequence simulation helpers
# ---------------------------------------------------------------------------


def _import_tree_class():
    # Keep local import so scripts that only analyze tables do not require ete3.
    from ete3 import Tree

    return Tree


def _build_random_binary_tree(n_taxa: int, depth_scale: float, seed: int):
    Tree = _import_tree_class()
    rng = random.Random(seed)
    nodes = [Tree(name=f"taxon{i + 1:02d}") for i in range(n_taxa)]

    while len(nodes) > 1:
        rng.shuffle(nodes)
        next_nodes = []
        i = 0
        while i < len(nodes):
            if i == len(nodes) - 1:
                next_nodes.append(nodes[i])
                i += 1
                continue
            left = nodes[i]
            right = nodes[i + 1]
            i += 2
            parent = Tree()
            left.dist = max(0.01, rng.lognormvariate(-2.2, 0.35) * depth_scale)
            right.dist = max(0.01, rng.lognormvariate(-2.2, 0.35) * depth_scale)
            parent.add_child(left)
            parent.add_child(right)
            next_nodes.append(parent)
        nodes = next_nodes

    tree = nodes[0]
    tree.name = "root"
    return tree


def _enumerate_tree_branches(tree) -> List[Tuple[Any, str]]:
    branches: List[Tuple[Any, str]] = []
    internal_counter = 0
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue
        if node.is_leaf():
            name = node.name
        else:
            internal_counter += 1
            name = f"internal_{internal_counter}"
        branches.append((node, name))
    return branches


def _sample_regime_parameters(regime: str, rng: np.random.Generator) -> Tuple[float, float, float]:
    spec = REGIME_SPECS[regime]
    burden = rng.uniform(spec["burden_min"], spec["burden_max"])
    effect = rng.uniform(spec["effect_min"], spec["effect_max"])
    branch_fraction = rng.uniform(spec["branch_frac_min"], spec["branch_frac_max"])
    return float(burden), float(effect), float(branch_fraction)


def _choose_latent_burden_matrix(
    n_branches: int,
    n_codons: int,
    burden_fraction: float,
    branch_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = np.zeros((n_branches, n_codons), dtype=np.int8)
    if burden_fraction <= 0.0 or branch_fraction <= 0.0:
        return matrix

    n_active_branches = max(1, int(round(branch_fraction * n_branches)))
    active_branches = rng.choice(n_branches, size=n_active_branches, replace=False)

    target_cells = max(1, int(round(burden_fraction * n_branches * n_codons)))
    n_sites = max(1, min(n_codons, int(math.ceil(target_cells / n_active_branches))))
    active_sites = rng.choice(n_codons, size=n_sites, replace=False)

    matrix[np.ix_(active_branches, active_sites)] = 1

    # Exact-match target density if rounding overshoots.
    active = np.flatnonzero(matrix.ravel() == 1)
    if active.size > target_cells:
        keep = rng.choice(active, size=target_cells, replace=False)
        matrix[:] = 0
        matrix.ravel()[keep] = 1
    return matrix


def _mutate_codon_indices(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Simple substitution model: re-sample codons uniformly.
    return rng.integers(0, len(CODONS), size=indices.shape[0], endpoint=False)


def _simulate_leaf_codon_matrix(
    tree,
    branch_rows: Sequence[Tuple[Any, str]],
    n_codons: int,
    latent_matrix: np.ndarray,
    base_mutation_rate: float,
    burden_effect: float,
    rate_heterogeneity_sigma: float,
    replicate_seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(replicate_seed)

    root_codons = rng.integers(0, len(CODONS), size=n_codons, endpoint=False)
    if rate_heterogeneity_sigma > 0:
        # Log-normal with mean ~1.0
        site_rate = rng.lognormal(
            mean=-(rate_heterogeneity_sigma ** 2) / 2.0,
            sigma=rate_heterogeneity_sigma,
            size=n_codons,
        )
    else:
        site_rate = np.ones(n_codons, dtype=float)

    branch_index_by_name = {name: idx for idx, (_, name) in enumerate(branch_rows)}

    for node in tree.traverse("preorder"):
        if node.is_root():
            node._codon_states = root_codons.copy()
            continue

        parent_states = node.up._codon_states
        states = parent_states.copy()
        branch_name = None
        if node.is_leaf():
            branch_name = node.name
        else:
            # Must match branch enumeration logic.
            for candidate, candidate_name in branch_rows:
                if candidate is node:
                    branch_name = candidate_name
                    break
        if branch_name is None:
            raise RuntimeError("Failed to resolve branch name during simulation.")

        bidx = branch_index_by_name[branch_name]
        branch_mask = latent_matrix[bidx].astype(float)

        neutral_component = base_mutation_rate * max(float(node.dist), 0.01) * site_rate
        burden_component = burden_effect * branch_mask
        p_mut = np.clip(neutral_component + burden_component, 0.0, 0.95)

        mutate = rng.random(n_codons) < p_mut
        if np.any(mutate):
            states[mutate] = _mutate_codon_indices(states[mutate], rng)
        node._codon_states = states

    leaves: Dict[str, np.ndarray] = {}
    for leaf in tree.iter_leaves():
        leaves[leaf.name] = leaf._codon_states.copy()
    return leaves


def _apply_recombination(
    leaf_codons: Dict[str, np.ndarray],
    recombination_rate: float,
    replicate_seed: int,
) -> Dict[str, np.ndarray]:
    if recombination_rate <= 0.0:
        return leaf_codons

    rng = np.random.default_rng(replicate_seed + 800_001)
    taxa = sorted(leaf_codons.keys())
    out = {taxon: arr.copy() for taxon, arr in leaf_codons.items()}
    if len(taxa) < 3:
        return out

    n_codons = next(iter(out.values())).shape[0]
    for recipient in taxa:
        donor_pool = [name for name in taxa if name != recipient]
        if not donor_pool:
            continue
        mask = rng.random(n_codons) < recombination_rate
        indices = np.flatnonzero(mask)
        for codon_idx in indices:
            donor = donor_pool[int(rng.integers(0, len(donor_pool)))]
            out[recipient][codon_idx] = out[donor][codon_idx]
    return out


def _codon_matrix_to_sequences(
    leaf_codons: Dict[str, np.ndarray],
    alignment_noise_rate: float,
    replicate_seed: int,
) -> Dict[str, str]:
    rng = np.random.default_rng(replicate_seed + 1_700_003)
    sequences: Dict[str, str] = {}
    for taxon, codon_indices in leaf_codons.items():
        seq_chars = list("".join(CODONS[idx] for idx in codon_indices.tolist()))
        if alignment_noise_rate > 0:
            for i, char in enumerate(seq_chars):
                if rng.random() < alignment_noise_rate:
                    pool = NUCLEOTIDES[NUCLEOTIDES != char]
                    seq_chars[i] = str(pool[int(rng.integers(0, pool.shape[0]))])
        sequences[taxon] = "".join(seq_chars)
    return sequences


def _write_alignment(path: Path, sequences: Mapping[str, str]) -> None:
    records = [SeqRecord(Seq(seq), id=taxon, description="") for taxon, seq in sorted(sequences.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, path, "fasta")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _scenario_grid(dispersion_choices: Sequence[str]) -> List[Tuple[str, str, str, str, str, str]]:
    combos = list(
        itertools.product(
            LENGTH_BINS.keys(),
            TREE_BINS.keys(),
            RECOMBINATION_BINS.keys(),
            ALIGNMENT_NOISE_BINS.keys(),
            RATE_HETEROGENEITY_BINS.keys(),
            dispersion_choices,
        )
    )
    return [(str(a), str(b), str(c), str(d), str(e), str(f)) for a, b, c, d, e, f in combos]


def _sample_gene_length(length_bin: str, rng: np.random.Generator) -> int:
    lo, hi = LENGTH_BINS[length_bin]
    val = int(rng.integers(lo, hi + 1))
    # Keep codon length valid.
    return max(3, val - (val % 3))


def simulate_alignment_validation_dataset(
    *,
    outdir: str | Path,
    n_per_regime: int,
    n_replicates_per_scenario: int,
    seed: int,
    dispersion_choices: Sequence[str] = DEFAULT_DISPERSION_CHOICES,
) -> Dict[str, Any]:
    if n_per_regime <= 0:
        raise ValueError("n_per_regime must be > 0.")
    if n_replicates_per_scenario <= 0:
        raise ValueError("n_replicates_per_scenario must be > 0.")

    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    dataset_rows: List[Dict[str, Any]] = []
    scenario_rows: List[Dict[str, Any]] = []

    master_rng = np.random.default_rng(seed)
    strata = _scenario_grid(dispersion_choices)
    master_rng.shuffle(strata)

    scenario_counter = 0
    for regime in ("neutral", "low", "medium", "high"):
        for i in range(n_per_regime):
            stratum = strata[(scenario_counter + i) % len(strata)]
            (
                length_bin,
                tree_bin,
                recomb_bin,
                noise_bin,
                rate_bin,
                dispersion_mode,
            ) = stratum

            tree_cfg = TREE_BINS[tree_bin]
            n_taxa = int(tree_cfg["n_taxa"])
            depth_scale = float(tree_cfg["depth_scale"])

            scenario_seed = int(master_rng.integers(1, 2**31 - 1))
            scenario_rng = np.random.default_rng(scenario_seed)
            gene_length_nt = _sample_gene_length(length_bin, scenario_rng)
            burden_fraction, burden_effect, branch_fraction = _sample_regime_parameters(regime, scenario_rng)

            spec = ScenarioSpec(
                scenario_id=f"scenario_{scenario_counter:05d}",
                regime=regime,
                gene_length_bin=length_bin,
                tree_bin=tree_bin,
                recombination_bin=recomb_bin,
                alignment_noise_bin=noise_bin,
                mutation_rate_heterogeneity_bin=rate_bin,
                dispersion_statistic=dispersion_mode,
                n_taxa=n_taxa,
                depth_scale=depth_scale,
                gene_length_nt=gene_length_nt,
                recombination_rate=float(RECOMBINATION_BINS[recomb_bin]),
                alignment_noise_rate=float(ALIGNMENT_NOISE_BINS[noise_bin]),
                rate_heterogeneity_sigma=float(RATE_HETEROGENEITY_BINS[rate_bin]),
                burden_fraction=burden_fraction,
                branch_fraction=branch_fraction,
                burden_effect=burden_effect,
                base_mutation_rate=float(0.04 * depth_scale),
                scenario_seed=scenario_seed,
            )

            scenario_dir = out / "scenarios" / spec.scenario_id
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Fixed latent structure for this scenario.
            tree = _build_random_binary_tree(spec.n_taxa, spec.depth_scale, spec.scenario_seed)
            tree_path = scenario_dir / "tree.nwk"
            tree.write(outfile=str(tree_path), format=1)
            branch_rows = _enumerate_tree_branches(tree)
            n_branches = len(branch_rows)
            n_codons = spec.gene_length_nt // 3

            latent_matrix = _choose_latent_burden_matrix(
                n_branches=n_branches,
                n_codons=n_codons,
                burden_fraction=spec.burden_fraction,
                branch_fraction=spec.branch_fraction,
                rng=np.random.default_rng(spec.scenario_seed + 97),
            )

            active_branch_names = [
                name
                for idx, (_, name) in enumerate(branch_rows)
                if int(latent_matrix[idx].sum()) > 0
            ]
            active_site_count = int(np.any(latent_matrix > 0, axis=0).sum())

            scenario_payload = {
                **spec.__dict__,
                "n_branches": n_branches,
                "n_codons": n_codons,
                "active_branch_names": active_branch_names,
                "active_branch_count": len(active_branch_names),
                "active_site_count": active_site_count,
                "latent_cell_fraction_realized": float(latent_matrix.mean()),
            }
            _write_json(scenario_dir / "scenario_metadata.json", scenario_payload)
            scenario_rows.append(scenario_payload)

            for rep in range(n_replicates_per_scenario):
                replicate_id = f"rep_{rep:03d}"
                rep_seed = int(spec.scenario_seed + 10_000 + rep)
                rep_dir = scenario_dir / "replicates" / replicate_id
                rep_dir.mkdir(parents=True, exist_ok=True)

                # Re-load tree to keep simulation deterministic and independent.
                tree_rep = _import_tree_class()(str(tree_path), format=1)
                branch_rows_rep = _enumerate_tree_branches(tree_rep)

                leaf_codons = _simulate_leaf_codon_matrix(
                    tree=tree_rep,
                    branch_rows=branch_rows_rep,
                    n_codons=n_codons,
                    latent_matrix=latent_matrix,
                    base_mutation_rate=spec.base_mutation_rate,
                    burden_effect=spec.burden_effect,
                    rate_heterogeneity_sigma=spec.rate_heterogeneity_sigma,
                    replicate_seed=rep_seed,
                )
                recombined = _apply_recombination(
                    leaf_codons,
                    recombination_rate=spec.recombination_rate,
                    replicate_seed=rep_seed,
                )
                sequences = _codon_matrix_to_sequences(
                    recombined,
                    alignment_noise_rate=spec.alignment_noise_rate,
                    replicate_seed=rep_seed,
                )

                aln_path = rep_dir / "alignment.fasta"
                _write_alignment(aln_path, sequences)
                rep_tree_path = rep_dir / "tree.nwk"
                rep_tree_path.write_text(tree_path.read_text())

                truth_payload = {
                    "scenario_id": spec.scenario_id,
                    "replicate_id": replicate_id,
                    "regime": spec.regime,
                    "scenario_seed": spec.scenario_seed,
                    "replicate_seed": rep_seed,
                    "gene_length_nt": spec.gene_length_nt,
                    "n_codons": n_codons,
                    "n_taxa": spec.n_taxa,
                    "n_branches": n_branches,
                    "active_branch_count": len(active_branch_names),
                    "active_site_count": active_site_count,
                    "latent_cell_fraction_realized": float(latent_matrix.mean()),
                    "active_branch_names": active_branch_names,
                    "nuisance_bins": {
                        "gene_length_bin": spec.gene_length_bin,
                        "tree_bin": spec.tree_bin,
                        "recombination_bin": spec.recombination_bin,
                        "alignment_noise_bin": spec.alignment_noise_bin,
                        "mutation_rate_heterogeneity_bin": spec.mutation_rate_heterogeneity_bin,
                        "dispersion_statistic": spec.dispersion_statistic,
                    },
                }
                _write_json(rep_dir / "truth_metadata.json", truth_payload)

                dataset_rows.append(
                    {
                        "scenario_id": spec.scenario_id,
                        "replicate_id": replicate_id,
                        "regime": spec.regime,
                        "scenario_seed": spec.scenario_seed,
                        "replicate_seed": rep_seed,
                        "gene_length_bin": spec.gene_length_bin,
                        "tree_bin": spec.tree_bin,
                        "recombination_bin": spec.recombination_bin,
                        "alignment_noise_bin": spec.alignment_noise_bin,
                        "mutation_rate_heterogeneity_bin": spec.mutation_rate_heterogeneity_bin,
                        "dispersion_statistic": spec.dispersion_statistic,
                        "gene_length_nt": spec.gene_length_nt,
                        "n_taxa": spec.n_taxa,
                        "depth_scale": spec.depth_scale,
                        "recombination_rate": spec.recombination_rate,
                        "alignment_noise_rate": spec.alignment_noise_rate,
                        "rate_heterogeneity_sigma": spec.rate_heterogeneity_sigma,
                        "burden_fraction": spec.burden_fraction,
                        "branch_fraction": spec.branch_fraction,
                        "burden_effect": spec.burden_effect,
                        "base_mutation_rate": spec.base_mutation_rate,
                        "latent_cell_fraction_realized": float(latent_matrix.mean()),
                        "active_branch_count": len(active_branch_names),
                        "active_site_count": active_site_count,
                        "stratum_id": "|".join(
                            [
                                spec.gene_length_bin,
                                spec.tree_bin,
                                spec.recombination_bin,
                                spec.alignment_noise_bin,
                                spec.mutation_rate_heterogeneity_bin,
                                spec.dispersion_statistic,
                            ]
                        ),
                        "alignment_path": str(aln_path),
                        "tree_path": str(rep_tree_path),
                        "truth_metadata_path": str(rep_dir / "truth_metadata.json"),
                    }
                )

            scenario_counter += 1

    _write_tsv(out / "synthetic_dataset.tsv", dataset_rows)
    _write_json(
        out / "dataset_summary.json",
        {
            "n_rows": len(dataset_rows),
            "n_scenarios": len(scenario_rows),
            "n_per_regime": n_per_regime,
            "n_replicates_per_scenario": n_replicates_per_scenario,
            "seed": seed,
            "regime_counts": {
                regime: int(sum(1 for row in scenario_rows if row["regime"] == regime))
                for regime in ("neutral", "low", "medium", "high")
            },
        },
    )
    _write_tsv(out / "scenario_table.tsv", scenario_rows)

    return {
        "dataset_tsv": str(out / "synthetic_dataset.tsv"),
        "scenario_tsv": str(out / "scenario_table.tsv"),
        "summary_json": str(out / "dataset_summary.json"),
    }


# ---------------------------------------------------------------------------
# Full-pipeline inference aggregation
# ---------------------------------------------------------------------------


def _site_summary_mean_score(site_summary_path: Path) -> float:
    scores = []
    with site_summary_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            scores.append(float(row["site_score"]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _compute_dispersion_from_site_summary(site_summary_path: Path, mode: str) -> float:
    site_scores: List[float] = []
    site_logits: List[float] = []
    with site_summary_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            site_scores.append(float(row["site_score"]))
            site_logits.append(float(row["site_logit_mean"]))

    if len(site_scores) <= 1:
        return 0.0

    if mode == "site_logit_variance":
        return float(np.var(np.asarray(site_logits), ddof=1))
    if mode == "site_score_variance":
        return float(np.var(np.asarray(site_scores), ddof=1))
    raise ValueError(f"Unsupported dispersion statistic: {mode}")


def _coerce_finite_float(value: Any, *, field: str, run_label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{run_label}: non-numeric `{field}` value: {value!r}") from exc
    if not np.isfinite(out):
        raise RuntimeError(f"{run_label}: non-finite `{field}` value: {out!r}")
    return out


def _coerce_optional_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": int(arr.size),
            "finite_count": 0,
            "min": float("nan"),
            "median": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def run_full_pipeline_inference_on_dataset(
    *,
    dataset_tsv: str | Path,
    outdir: str | Path,
    tree_calibration: bool,
    n_calibration: int,
    device: str,
    batch_size: int,
    sigma_floor: float = 0.05,
    alpha: float = 0.05,
    pvalue_mode: str = "empirical_monte_carlo",
    min_neutral_group_size: int = 20,
    neutral_reps: int = 200,
    offline: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    rows = read_tsv(dataset_tsv)
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    inference_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    for row in rows:
        scenario_id = row["scenario_id"]
        replicate_id = row["replicate_id"]
        run_label = f"{scenario_id}:{replicate_id}"
        run_out = out / "inference_runs" / scenario_id / replicate_id

        payload = run_and_write_outputs(
            alignment_path=row["alignment_path"],
            tree_path=row["tree_path"],
            outdir=run_out,
            command=(
                "python scripts/run_full_pipeline_validation.py "
                f"--dataset-row {scenario_id}:{replicate_id}"
            ),
            tree_calibration=tree_calibration,
            n_calibration=n_calibration,
            device=device,
            batch_size=batch_size,
            seed=int(row["replicate_seed"]),
            foreground_mode="all-leaves",
            foreground_list=None,
            sigma_floor=sigma_floor,
            alpha=alpha,
            pvalue_mode=pvalue_mode,
            min_neutral_group_size=min_neutral_group_size,
            neutral_reps=neutral_reps,
            offline=offline,
            overwrite=overwrite,
        )

        gene = payload["gene_summary"]
        site_summary_path = run_out / "site_summary.tsv"
        requested_dispersion = row.get("dispersion_statistic", "site_logit_variance")
        d_obs_requested = _compute_dispersion_from_site_summary(site_summary_path, requested_dispersion)

        d_obs = _coerce_finite_float(
            gene.get("D_obs", gene.get("observed_variance")),
            field="D_obs",
            run_label=run_label,
        )
        mu0 = _coerce_finite_float(
            gene.get("mu0", gene.get("neutral_expected_variance")),
            field="mu0",
            run_label=run_label,
        )
        eii_z = _coerce_finite_float(gene.get("EII_z"), field="EII_z", run_label=run_label)

        raw_sigma0 = _coerce_optional_float(gene.get("sigma0_raw", gene.get("neutral_sd_raw")))
        sigma0_candidates = [
            gene.get("sigma0_final"),
            gene.get("neutral_sd_floored"),
            gene.get("neutral_sd"),
        ]
        sigma0 = float("nan")
        for candidate in sigma0_candidates:
            value = _coerce_optional_float(candidate)
            if np.isfinite(value) and value > 0:
                sigma0 = value
                break
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise RuntimeError(
                f"{run_label}: missing valid floored sigma0 in inference output; "
                "cannot safely compute EII calibration."
            )

        if sigma0 + 1e-12 < float(sigma_floor):
            raise RuntimeError(
                f"{run_label}: sigma0={sigma0:.6g} is below requested sigma_floor={sigma_floor:.6g}."
            )

        expected_eii_z = float((d_obs - mu0) / sigma0)
        if not np.isclose(expected_eii_z, eii_z, rtol=1e-6, atol=1e-8):
            raise RuntimeError(
                f"{run_label}: inconsistent EII calibration values. "
                f"(D_obs-mu0)/sigma0={expected_eii_z:.8g}, EII_z={eii_z:.8g}"
            )

        fallback_flag = int(bool(gene.get("fallback_flag", gene.get("calibration_fallback_flag"))))
        fallback_reason = str(gene.get("fallback_reason", gene.get("calibration_fallback_reason")) or "")
        neutral_group_size_raw = gene.get("neutral_group_size")
        neutral_group_size = (
            int(neutral_group_size_raw)
            if neutral_group_size_raw is not None and str(neutral_group_size_raw) != ""
            else 0
        )
        sigma_floor_applied = int(bool(gene.get("sigma_floor_applied", sigma0 <= sigma_floor + 1e-12)))
        calibration_group = str(gene.get("calibration_group", row.get("stratum_id", "global")))
        calibration_source = str(gene.get("calibration_source") or "")
        p_emp = _coerce_optional_float(gene.get("p_emp"))
        q_emp = _coerce_optional_float(gene.get("q_emp"))
        significant_bool = int(bool(gene.get("significant_bool")))
        significance_label = str(gene.get("significance_label") or "not_significant")
        requested_matches_calibration = int(requested_dispersion == "site_logit_variance")

        inference_rows.append(
            {
                **row,
                "results_json": str(run_out / "results.json"),
                "branch_summary_tsv": str(run_out / "branch_summary.tsv"),
                "site_summary_tsv": str(site_summary_path),
                "run_metadata_json": str(run_out / "run_metadata.json"),
                "D_obs": d_obs,
                "D_obs_requested_stat": d_obs_requested,
                "mu0": mu0,
                "sigma0": sigma0,
                "raw_sigma0": raw_sigma0,
                "floored_sigma0": sigma0,
                "EII_z": eii_z,
                "EII_01": _coerce_finite_float(gene.get("EII_01"), field="EII_01", run_label=run_label),
                "p_emp": p_emp,
                "q_emp": q_emp,
                "alpha_used": float(alpha),
                "significant_bool": significant_bool,
                "significance_label": significance_label,
                "identifiable_bool": int(bool(gene["identifiable_bool"])),
                "identifiability_extent": str(gene["identifiability_extent"]),
                "gene_burden_score": _site_summary_mean_score(site_summary_path),
                "calibration_group": calibration_group,
                "calibration_source": calibration_source,
                "calibration_fallback_flag": fallback_flag,
                "calibration_fallback_reason": fallback_reason,
                "neutral_group_size": neutral_group_size,
                "sigma_floor_requested": float(sigma_floor),
                "sigma_floor_applied": sigma_floor_applied,
                "requested_dispersion_matches_calibration_stat": requested_matches_calibration,
            }
        )
        debug_rows.append(
            {
                "scenario_id": scenario_id,
                "replicate_id": replicate_id,
                "stratum_id": calibration_group,
                "calibration_group": calibration_group,
                "dispersion_statistic_requested": requested_dispersion,
                "calibration_dispersion_statistic": "site_logit_variance",
                "requested_dispersion_matches_calibration_stat": requested_matches_calibration,
                "D_obs": d_obs,
                "mu0": mu0,
                "raw_sigma0": raw_sigma0,
                "floored_sigma0": sigma0,
                "EII_z": eii_z,
                "EII_z_recomputed": expected_eii_z,
                "p_emp": p_emp,
                "q_emp": q_emp,
                "fallback_flag": fallback_flag,
                "fallback_reason": fallback_reason,
                "neutral_group_size": neutral_group_size,
                "calibration_source": calibration_source,
                "sigma_floor_requested": float(sigma_floor),
                "sigma_floor_applied": sigma_floor_applied,
            }
        )

    annotate_bh_qvalues(inference_rows, p_key="p_emp", q_key="q_emp")
    for row, debug in zip(inference_rows, debug_rows):
        qv = _coerce_optional_float(row.get("q_emp"))
        row["alpha_used"] = float(alpha)
        row["significant_bool"] = int(np.isfinite(qv) and qv <= float(alpha))
        row["significance_label"] = "significant" if row["significant_bool"] else "not_significant"
        debug["q_emp"] = row["q_emp"]
        debug["alpha_used"] = float(alpha)
        debug["significant_bool"] = row["significant_bool"]
        debug["significance_label"] = row["significance_label"]

    _write_tsv(out / "full_pipeline_gene_metrics.tsv", inference_rows)
    _write_tsv(out / "full_pipeline_calibration_debug.tsv", debug_rows)

    floored_sigma0_values = [float(row["floored_sigma0"]) for row in debug_rows]
    raw_sigma0_values = [float(row["raw_sigma0"]) for row in debug_rows]
    fallback_flags = np.asarray([int(row["fallback_flag"]) for row in debug_rows], dtype=float)
    if sigma_floor > 0:
        at_floor = np.isclose(
            np.asarray(floored_sigma0_values, dtype=float),
            float(sigma_floor),
            rtol=0.0,
            atol=1e-12,
        )
        fraction_sigma0_at_floor = float(np.mean(at_floor)) if at_floor.size > 0 else float("nan")
    else:
        fraction_sigma0_at_floor = 0.0
    fraction_fallback = float(np.mean(fallback_flags)) if fallback_flags.size > 0 else float("nan")

    sigma_diagnostics = {
        "sigma_floor_requested": float(sigma_floor),
        "alpha_used": float(alpha),
        "pvalue_mode": str(pvalue_mode),
        "neutral_reps": int(neutral_reps),
        "min_neutral_group_size": int(min_neutral_group_size),
        "fraction_sigma0_at_floor": fraction_sigma0_at_floor,
        "fraction_fallback_applied": fraction_fallback,
        "sigma0_before_floor_summary": _summary_stats(raw_sigma0_values),
        "sigma0_after_floor_summary": _summary_stats(floored_sigma0_values),
    }
    _write_json(
        out / "full_pipeline_inference_summary.json",
        {
            "n_runs": len(inference_rows),
            "dataset_tsv": str(Path(dataset_tsv).resolve()),
            "output_table": str(out / "full_pipeline_gene_metrics.tsv"),
            "debug_table": str(out / "full_pipeline_calibration_debug.tsv"),
            "sigma_diagnostics": sigma_diagnostics,
        },
    )
    return {
        "metrics_tsv": str(out / "full_pipeline_gene_metrics.tsv"),
        "debug_tsv": str(out / "full_pipeline_calibration_debug.tsv"),
        "summary_json": str(out / "full_pipeline_inference_summary.json"),
        "fraction_sigma0_at_floor": fraction_sigma0_at_floor,
        "fraction_fallback_applied": fraction_fallback,
        "sigma0_before_floor_summary": sigma_diagnostics["sigma0_before_floor_summary"],
        "sigma0_after_floor_summary": sigma_diagnostics["sigma0_after_floor_summary"],
    }


# ---------------------------------------------------------------------------
# Threshold calibration and performance
# ---------------------------------------------------------------------------


def label_positive(regime_values: Sequence[str], decision_target: str) -> np.ndarray:
    arr = np.asarray(regime_values, dtype=object)
    if decision_target == "any_nonneutral":
        return (arr != "neutral").astype(int)
    if decision_target == "medium_high":
        return np.isin(arr, ["medium", "high"]).astype(int)
    if decision_target == "high_only":
        return (arr == "high").astype(int)
    raise ValueError(f"Unknown decision_target: {decision_target}")


def confusion_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    p = tp + fn
    n = tn + fp

    tpr = tp / p if p > 0 else float("nan")
    fpr = fp / n if n > 0 else float("nan")
    tnr = tn / n if n > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    bal_acc = np.nanmean([tpr, tnr])
    youden_j = tpr - fpr if not (math.isnan(tpr) or math.isnan(fpr)) else float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "TPR": tpr,
        "FPR": fpr,
        "precision": precision,
        "NPV": npv,
        "balanced_accuracy": bal_acc,
        "youden_j": youden_j,
    }


def _threshold_grid(score: np.ndarray, n_grid: int = 801) -> np.ndarray:
    lo = float(np.min(score))
    hi = float(np.max(score))
    base = np.linspace(lo, hi, n_grid)
    return np.unique(np.concatenate(([lo - 1e-9], base, [hi + 1e-9])))


def _threshold_table(y_true: np.ndarray, score: np.ndarray, n_grid: int = 801) -> List[Dict[str, float]]:
    rows = []
    for thr in _threshold_grid(score, n_grid=n_grid):
        rows.append(confusion_metrics(y_true, score, float(thr)))
    return rows


def _roc_table(threshold_rows: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
    seen: set[Tuple[float, float]] = set()
    rows: List[Dict[str, float]] = []
    for row in threshold_rows:
        fpr = float(row["FPR"])
        tpr = float(row["TPR"])
        if math.isnan(fpr) or math.isnan(tpr):
            continue
        key = (round(fpr, 12), round(tpr, 12))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"FPR": fpr, "TPR": tpr, "threshold": float(row["threshold"])})
    rows.sort(key=lambda r: (r["FPR"], r["TPR"]))
    return rows


def _pr_table(threshold_rows: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
    seen: set[Tuple[float, float]] = set()
    rows: List[Dict[str, float]] = []
    for row in threshold_rows:
        recall = float(row["TPR"])
        precision = float(row["precision"])
        if math.isnan(recall) or math.isnan(precision):
            continue
        key = (round(recall, 12), round(precision, 12))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"recall": recall, "precision": precision, "threshold": float(row["threshold"])})
    rows.sort(key=lambda r: (r["recall"], r["precision"]))
    return rows


def _auc_from_roc(roc_rows: Sequence[Mapping[str, float]]) -> float:
    if len(roc_rows) < 2:
        return float("nan")
    x = np.asarray([float(r["FPR"]) for r in roc_rows], dtype=float)
    y = np.asarray([float(r["TPR"]) for r in roc_rows], dtype=float)
    order = np.argsort(x)
    return _trapz_compat(y[order], x[order])


def _neutral_quantiles(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    neutral_eii01 = np.asarray([float(r["EII_01"]) for r in rows if r["regime"] == "neutral"], dtype=float)
    neutral_eiiz = np.asarray([float(r["EII_z"]) for r in rows if r["regime"] == "neutral"], dtype=float)
    if neutral_eii01.size == 0 or neutral_eiiz.size == 0:
        return {
            "neutral_q90_EII_01": float("nan"),
            "neutral_q95_EII_01": float("nan"),
            "neutral_q99_EII_01": float("nan"),
            "neutral_q90_EII_z": float("nan"),
            "neutral_q95_EII_z": float("nan"),
            "neutral_q99_EII_z": float("nan"),
        }
    return {
        "neutral_q90_EII_01": float(np.quantile(neutral_eii01, 0.90)),
        "neutral_q95_EII_01": float(np.quantile(neutral_eii01, 0.95)),
        "neutral_q99_EII_01": float(np.quantile(neutral_eii01, 0.99)),
        "neutral_q90_EII_z": float(np.quantile(neutral_eiiz, 0.90)),
        "neutral_q95_EII_z": float(np.quantile(neutral_eiiz, 0.95)),
        "neutral_q99_EII_z": float(np.quantile(neutral_eiiz, 0.99)),
    }


def _find_operating_thresholds(threshold_rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    clean = [
        row
        for row in threshold_rows
        if not (
            math.isnan(float(row["FPR"]))
            or math.isnan(float(row["TPR"]))
            or math.isnan(float(row["youden_j"]))
        )
    ]
    if not clean:
        return {
            "max_youden": float("nan"),
            "max_tpr_fpr_le_0.10": float("nan"),
            "max_tpr_fpr_le_0.05": float("nan"),
            "max_tpr_fpr_le_0.01": float("nan"),
        }

    by_youden = max(clean, key=lambda r: (float(r["youden_j"]), float(r["TPR"]), -float(r["FPR"])))

    def _max_tpr_under(limit: float) -> float:
        eligible = [r for r in clean if float(r["FPR"]) <= limit]
        if not eligible:
            return float("nan")
        best = max(eligible, key=lambda r: (float(r["TPR"]), -float(r["FPR"]), -float(r["threshold"])))
        return float(best["threshold"])

    return {
        "max_youden": float(by_youden["threshold"]),
        "max_tpr_fpr_le_0.10": _max_tpr_under(0.10),
        "max_tpr_fpr_le_0.05": _max_tpr_under(0.05),
        "max_tpr_fpr_le_0.01": _max_tpr_under(0.01),
    }


def _calibrate_group(
    *,
    rows: Sequence[Mapping[str, Any]],
    group_label: str,
    decision_target: str,
    default_threshold: float,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError(f"No rows available for calibration group: {group_label}")

    regimes = [str(r["regime"]) for r in rows]
    y_true = label_positive(regimes, decision_target)
    score_eii01 = np.asarray([float(r["EII_01"]) for r in rows], dtype=float)

    threshold_rows = _threshold_table(y_true, score_eii01)
    roc_rows = _roc_table(threshold_rows)
    pr_rows = _pr_table(threshold_rows)
    auc = _auc_from_roc(roc_rows)

    quant = _neutral_quantiles(rows)
    ops = _find_operating_thresholds(threshold_rows)

    candidate_thresholds = {
        "fixed_0.70": 0.70,
        "fixed_0.90": 0.90,
        "neutral_q95": quant["neutral_q95_EII_01"],
        "neutral_q99": quant["neutral_q99_EII_01"],
        "max_youden": ops["max_youden"],
        "max_tpr_fpr_le_0.10": ops["max_tpr_fpr_le_0.10"],
        "max_tpr_fpr_le_0.05": ops["max_tpr_fpr_le_0.05"],
        "max_tpr_fpr_le_0.01": ops["max_tpr_fpr_le_0.01"],
    }

    performance_rows = []
    for name, threshold in candidate_thresholds.items():
        if math.isnan(float(threshold)):
            continue
        perf = confusion_metrics(y_true, score_eii01, float(threshold))
        perf.update(
            {
                "group": group_label,
                "threshold_name": name,
                "threshold_value": float(threshold),
                "decision_target": decision_target,
                "default_threshold": default_threshold,
                "n": len(rows),
                "n_neutral": int(np.sum(np.asarray(regimes) == "neutral")),
                "AUC": auc,
            }
        )
        performance_rows.append(perf)

    return {
        "group": group_label,
        "n": len(rows),
        "n_neutral": int(np.sum(np.asarray(regimes) == "neutral")),
        "AUC": auc,
        "neutral_quantiles": quant,
        "performance_rows": performance_rows,
        "roc_rows": [{"group": group_label, **r} for r in roc_rows],
        "pr_rows": [{"group": group_label, **r} for r in pr_rows],
    }


def compute_threshold_calibration(
    *,
    metrics_tsv: str | Path,
    outdir: str | Path,
    decision_target: str,
    default_threshold: float,
) -> Dict[str, Any]:
    rows = read_tsv(metrics_tsv)
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, str]]] = {"global": rows}
    for row in rows:
        key = str(row["stratum_id"])
        grouped.setdefault(key, []).append(row)

    quantile_rows: List[Dict[str, Any]] = []
    performance_rows: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    pr_rows: List[Dict[str, Any]] = []

    for group_name, group_rows in grouped.items():
        result = _calibrate_group(
            rows=group_rows,
            group_label=group_name,
            decision_target=decision_target,
            default_threshold=default_threshold,
        )
        quant = result["neutral_quantiles"]
        quantile_rows.append(
            {
                "group": group_name,
                "n": result["n"],
                "n_neutral": result["n_neutral"],
                **quant,
            }
        )
        performance_rows.extend(result["performance_rows"])
        roc_rows.extend(result["roc_rows"])
        pr_rows.extend(result["pr_rows"])

    _write_tsv(out / "neutral_quantiles_by_group.tsv", quantile_rows)
    _write_tsv(out / "threshold_performance_by_group.tsv", performance_rows)
    _write_tsv(out / "roc_table_by_group.tsv", roc_rows)
    _write_tsv(out / "pr_table_by_group.tsv", pr_rows)

    return {
        "quantiles_tsv": str(out / "neutral_quantiles_by_group.tsv"),
        "performance_tsv": str(out / "threshold_performance_by_group.tsv"),
        "roc_tsv": str(out / "roc_table_by_group.tsv"),
        "pr_tsv": str(out / "pr_table_by_group.tsv"),
    }


def _binary_confusion_from_pred(y_true: np.ndarray, pred_bool: np.ndarray) -> Dict[str, float]:
    pred = pred_bool.astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    p = tp + fn
    n = tn + fp
    tpr = tp / p if p > 0 else float("nan")
    fpr = fp / n if n > 0 else float("nan")
    tnr = tn / n if n > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    bal_acc = np.nanmean([tpr, tnr])
    youden_j = tpr - fpr if not (math.isnan(tpr) or math.isnan(fpr)) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "TPR": tpr,
        "FPR": fpr,
        "precision": precision,
        "NPV": npv,
        "balanced_accuracy": bal_acc,
        "youden_j": youden_j,
    }


def _neutral_uniformity_ks(pvals: np.ndarray) -> float:
    if pvals.size == 0:
        return float("nan")
    clean = np.sort(np.asarray(pvals, dtype=float))
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    n = clean.size
    ecdf = np.arange(1, n + 1, dtype=float) / n
    return float(np.max(np.abs(ecdf - clean)))


def compute_significance_calibration(
    *,
    metrics_tsv: str | Path,
    outdir: str | Path,
    decision_target: str,
    alpha: float,
    default_threshold: float,
) -> Dict[str, Any]:
    rows = read_tsv(metrics_tsv)
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows available for significance calibration.")

    regimes = np.asarray([str(r["regime"]) for r in rows], dtype=object)
    y_true = label_positive(regimes, decision_target)
    eii = np.asarray([float(r["EII_01"]) for r in rows], dtype=float)
    p_emp = np.asarray([float(r.get("p_emp", float("nan"))) for r in rows], dtype=float)
    q_emp = np.asarray([float(r.get("q_emp", float("nan"))) for r in rows], dtype=float)
    sig_q = np.asarray([bool(float(q) <= alpha) if np.isfinite(q) else False for q in q_emp], dtype=bool)
    sig_eii = np.asarray([bool(float(v) >= float(default_threshold)) for v in eii], dtype=bool)

    q_metrics = _binary_confusion_from_pred(y_true, sig_q)
    eii_metrics = _binary_confusion_from_pred(y_true, sig_eii)

    regime_rows: List[Dict[str, Any]] = []
    for regime in ("neutral", "low", "medium", "high"):
        mask = regimes == regime
        if not np.any(mask):
            continue
        regime_rows.append(
            {
                "regime": regime,
                "n": int(np.sum(mask)),
                "significant_rate_q": float(np.mean(sig_q[mask])),
                "significant_rate_eii_threshold": float(np.mean(sig_eii[mask])),
                "mean_EII_01": float(np.mean(eii[mask])),
                "mean_p_emp": float(np.nanmean(p_emp[mask])),
            }
        )

    neutral_mask = regimes == "neutral"
    neutral_p = p_emp[neutral_mask]
    neutral_p = neutral_p[np.isfinite(neutral_p)]
    if neutral_p.size > 0:
        hist_counts, hist_edges = np.histogram(neutral_p, bins=np.linspace(0.0, 1.0, 11))
        hist_rows = [
            {
                "bin_left": float(hist_edges[i]),
                "bin_right": float(hist_edges[i + 1]),
                "count": int(hist_counts[i]),
            }
            for i in range(len(hist_counts))
        ]
        qgrid = np.linspace(0.05, 0.95, 19)
        qq_rows = []
        for q in qgrid:
            qq_rows.append(
                {
                    "expected_quantile": float(q),
                    "observed_quantile": float(np.quantile(neutral_p, q)),
                }
            )
    else:
        hist_rows = []
        qq_rows = []

    _write_tsv(out / "significance_regime_rates.tsv", regime_rows)
    _write_tsv(
        out / "significance_performance.tsv",
        [
            {"method": "q_emp", "alpha": float(alpha), **q_metrics},
            {"method": f"EII_01_ge_{default_threshold:.2f}", "alpha": float(alpha), **eii_metrics},
        ],
    )
    _write_tsv(out / "neutral_p_emp_histogram.tsv", hist_rows)
    _write_tsv(out / "neutral_p_emp_qq.tsv", qq_rows)

    comparison_summary = {
        "neutral_significant_rate_q": float(np.mean(sig_q[neutral_mask])) if np.any(neutral_mask) else float("nan"),
        "medium_high_significant_rate_q": float(np.mean(sig_q[np.isin(regimes, ["medium", "high"])]))
        if np.any(np.isin(regimes, ["medium", "high"]))
        else float("nan"),
        "neutral_p_uniformity_ks": _neutral_uniformity_ks(neutral_p),
        "q_emp_metrics": q_metrics,
        "eii_threshold_metrics": eii_metrics,
        "q_vs_eii_delta_balanced_accuracy": float(q_metrics["balanced_accuracy"] - eii_metrics["balanced_accuracy"]),
        "q_vs_eii_delta_fpr": float(q_metrics["FPR"] - eii_metrics["FPR"]),
        "q_vs_eii_delta_tpr": float(q_metrics["TPR"] - eii_metrics["TPR"]),
    }
    _write_json(out / "significance_summary.json", comparison_summary)

    return {
        "regime_rates_tsv": str(out / "significance_regime_rates.tsv"),
        "performance_tsv": str(out / "significance_performance.tsv"),
        "neutral_hist_tsv": str(out / "neutral_p_emp_histogram.tsv"),
        "neutral_qq_tsv": str(out / "neutral_p_emp_qq.tsv"),
        "summary_json": str(out / "significance_summary.json"),
        "summary": comparison_summary,
    }


# ---------------------------------------------------------------------------
# Bootstrap uncertainty
# ---------------------------------------------------------------------------


def bootstrap_eii_thresholds(
    *,
    metrics_tsv: str | Path,
    outdir: str | Path,
    bootstrap_reps: int,
    seed: int,
    decision_target: str,
    default_threshold: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be > 0.")

    rows = read_tsv(metrics_tsv)
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows available for bootstrap.")

    rng = np.random.default_rng(seed)
    n = len(rows)

    replicate_rows: List[Dict[str, Any]] = []
    for rep in range(bootstrap_reps):
        idx = rng.integers(0, n, size=n)
        sample = [rows[i] for i in idx.tolist()]

        quant = _neutral_quantiles(sample)
        regimes = [str(r["regime"]) for r in sample]
        y_true = label_positive(regimes, decision_target)
        score = np.asarray([float(r["EII_01"]) for r in sample], dtype=float)
        p_emp = np.asarray([float(r.get("p_emp", float("nan"))) for r in sample], dtype=float)
        q_emp = bh_adjust(p_emp)

        thr_rows = _threshold_table(y_true, score, n_grid=301)
        roc_rows = _roc_table(thr_rows)
        auc = _auc_from_roc(roc_rows)
        perf_070 = confusion_metrics(y_true, score, default_threshold)
        sig_pred = np.asarray([bool(q <= alpha) if np.isfinite(q) else False for q in q_emp], dtype=bool)
        sig_perf = _binary_confusion_from_pred(y_true, sig_pred)
        regimes_arr = np.asarray(regimes, dtype=object)
        neutral_mask = regimes_arr == "neutral"

        replicate_rows.append(
            {
                "bootstrap_rep": rep,
                "neutral_q95_EII_01": quant["neutral_q95_EII_01"],
                "neutral_q99_EII_01": quant["neutral_q99_EII_01"],
                "AUC": auc,
                "FPR_at_default": perf_070["FPR"],
                "TPR_at_default": perf_070["TPR"],
                "balanced_accuracy_at_default": perf_070["balanced_accuracy"],
                "FPR_q_alpha": sig_perf["FPR"],
                "TPR_q_alpha": sig_perf["TPR"],
                "balanced_accuracy_q_alpha": sig_perf["balanced_accuracy"],
                "neutral_significant_rate_q": float(np.mean(sig_pred[neutral_mask])) if np.any(neutral_mask) else float("nan"),
                "neutral_p_uniformity_ks": _neutral_uniformity_ks(p_emp[neutral_mask]),
            }
        )

    metrics = [
        "neutral_q95_EII_01",
        "neutral_q99_EII_01",
        "AUC",
        "FPR_at_default",
        "TPR_at_default",
        "balanced_accuracy_at_default",
        "FPR_q_alpha",
        "TPR_q_alpha",
        "balanced_accuracy_q_alpha",
        "neutral_significant_rate_q",
        "neutral_p_uniformity_ks",
    ]
    summary_rows = []
    for metric in metrics:
        values = np.asarray([float(r[metric]) for r in replicate_rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary_rows.append(
                {
                    "metric": metric,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "ci_lower_2_5": float("nan"),
                    "ci_upper_97_5": float("nan"),
                }
            )
            continue
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "ci_lower_2_5": float(np.quantile(finite, 0.025)),
                "ci_upper_97_5": float(np.quantile(finite, 0.975)),
            }
        )

    _write_tsv(out / "bootstrap_replicates.tsv", replicate_rows)
    _write_tsv(out / "bootstrap_summary.tsv", summary_rows)

    return {
        "bootstrap_replicates_tsv": str(out / "bootstrap_replicates.tsv"),
        "bootstrap_summary_tsv": str(out / "bootstrap_summary.tsv"),
    }


# ---------------------------------------------------------------------------
# Reproducibility / recoverability analyses
# ---------------------------------------------------------------------------


def _pairwise_replicate_correlations(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_scenario: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    replicate_ids: set[str] = set()
    for row in rows:
        sid = str(row["scenario_id"])
        rid = str(row["replicate_id"])
        replicate_ids.add(rid)
        by_scenario.setdefault(sid, {})[rid] = row

    replicate_ids_sorted = sorted(replicate_ids)
    out: List[Dict[str, Any]] = []
    for i, r1 in enumerate(replicate_ids_sorted):
        for r2 in replicate_ids_sorted[i + 1 :]:
            x = []
            y = []
            for sid, rep_map in by_scenario.items():
                if r1 in rep_map and r2 in rep_map:
                    x.append(float(rep_map[r1]["gene_burden_score"]))
                    y.append(float(rep_map[r2]["gene_burden_score"]))
            if len(x) < 2:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])
            out.append(
                {
                    "replicate_a": r1,
                    "replicate_b": r2,
                    "n_scenarios": len(x),
                    "gene_burden_score_correlation": corr,
                }
            )
    return out


def analyze_replicate_recoverability(
    *,
    metrics_tsv: str | Path,
    outdir: str | Path,
    default_threshold: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    rows = read_tsv(metrics_tsv)
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    by_scenario: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)

    scenario_rows: List[Dict[str, Any]] = []
    for scenario_id, scenario_data in sorted(by_scenario.items()):
        eii = np.asarray([float(r["EII_01"]) for r in scenario_data], dtype=float)
        q_emp = np.asarray([float(r.get("q_emp", float("nan"))) for r in scenario_data], dtype=float)
        burden = np.asarray([float(r["gene_burden_score"]) for r in scenario_data], dtype=float)
        above = (eii >= default_threshold).astype(int)
        significant = np.asarray([1 if np.isfinite(q) and q <= alpha else 0 for q in q_emp], dtype=int)

        # Pairwise threshold-consistency rate among replicates.
        pair_matches = []
        pair_sig_matches = []
        if above.size >= 2:
            for i in range(above.size):
                for j in range(i + 1, above.size):
                    pair_matches.append(1.0 if above[i] == above[j] else 0.0)
                    pair_sig_matches.append(1.0 if significant[i] == significant[j] else 0.0)
        stability = float(np.mean(pair_matches)) if pair_matches else 1.0
        sig_stability = float(np.mean(pair_sig_matches)) if pair_sig_matches else 1.0

        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "regime": scenario_data[0]["regime"],
                "stratum_id": scenario_data[0]["stratum_id"],
                "n_replicates": len(scenario_data),
                "mean_EII_01": float(np.mean(eii)),
                "var_EII_01": float(np.var(eii, ddof=1)) if eii.size > 1 else 0.0,
                "mean_q_emp": float(np.nanmean(q_emp)),
                "mean_gene_burden_score": float(np.mean(burden)),
                "var_gene_burden_score": float(np.var(burden, ddof=1)) if burden.size > 1 else 0.0,
                "probability_above_threshold": float(np.mean(above)),
                "probability_significant_q": float(np.mean(significant)),
                "always_above_threshold": int(np.all(above == 1)),
                "always_significant_q": int(np.all(significant == 1)),
                "pairwise_threshold_consistency": stability,
                "pairwise_significance_consistency": sig_stability,
                "mean_eii_ge_threshold": int(float(np.mean(eii)) >= default_threshold),
                "mean_q_significant": int(float(np.mean(significant)) >= 0.5),
            }
        )

    pair_corr_rows = _pairwise_replicate_correlations(rows)

    # Stratified summary requested by threshold split.
    strat_rows: List[Dict[str, Any]] = []
    for label, subset in (
        ("EII_01_ge_threshold", [r for r in scenario_rows if r["mean_eii_ge_threshold"] == 1]),
        ("EII_01_lt_threshold", [r for r in scenario_rows if r["mean_eii_ge_threshold"] == 0]),
        ("q_significant_majority", [r for r in scenario_rows if r["mean_q_significant"] == 1]),
        ("q_not_significant_majority", [r for r in scenario_rows if r["mean_q_significant"] == 0]),
    ):
        if subset:
            strat_rows.append(
                {
                    "stratum": label,
                    "n_scenarios": len(subset),
                    "mean_probability_above_threshold": float(np.mean([float(r["probability_above_threshold"]) for r in subset])),
                    "mean_probability_significant_q": float(np.mean([float(r["probability_significant_q"]) for r in subset])),
                    "mean_var_EII_01": float(np.mean([float(r["var_EII_01"]) for r in subset])),
                    "mean_var_gene_burden_score": float(np.mean([float(r["var_gene_burden_score"]) for r in subset])),
                    "mean_pairwise_threshold_consistency": float(np.mean([float(r["pairwise_threshold_consistency"]) for r in subset])),
                    "mean_pairwise_significance_consistency": float(np.mean([float(r["pairwise_significance_consistency"]) for r in subset])),
                }
            )
        else:
            strat_rows.append(
                {
                    "stratum": label,
                    "n_scenarios": 0,
                    "mean_probability_above_threshold": float("nan"),
                    "mean_probability_significant_q": float("nan"),
                    "mean_var_EII_01": float("nan"),
                    "mean_var_gene_burden_score": float("nan"),
                    "mean_pairwise_threshold_consistency": float("nan"),
                    "mean_pairwise_significance_consistency": float("nan"),
                }
            )

    _write_tsv(out / "scenario_recoverability.tsv", scenario_rows)
    _write_tsv(out / "replicate_pair_correlations.tsv", pair_corr_rows)
    _write_tsv(out / "recoverability_stratified_summary.tsv", strat_rows)

    return {
        "scenario_recoverability_tsv": str(out / "scenario_recoverability.tsv"),
        "pairwise_corr_tsv": str(out / "replicate_pair_correlations.tsv"),
        "stratified_tsv": str(out / "recoverability_stratified_summary.tsv"),
    }


# ---------------------------------------------------------------------------
# Plots and manuscript-facing reporting
# ---------------------------------------------------------------------------


def make_validation_figures(
    *,
    metrics_tsv: str | Path,
    roc_tsv: str | Path,
    threshold_perf_tsv: str | Path,
    significance_regime_tsv: str | Path,
    neutral_p_hist_tsv: str | Path,
    neutral_p_qq_tsv: str | Path,
    recoverability_tsv: str | Path,
    outdir: str | Path,
    default_threshold: float,
) -> Dict[str, Any]:
    plt = _get_pyplot()
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(metrics_tsv)
    roc_rows = [r for r in read_tsv(roc_tsv) if r["group"] == "global"]
    perf_rows = [r for r in read_tsv(threshold_perf_tsv) if r["group"] == "global"]
    sig_regime_rows = read_tsv(significance_regime_tsv)
    neutral_hist_rows = read_tsv(neutral_p_hist_tsv)
    neutral_qq_rows = read_tsv(neutral_p_qq_tsv)
    rec_rows = read_tsv(recoverability_tsv)

    # Figure 1: EII_01 distribution by regime (grayscale-safe hist lines).
    regimes = ["neutral", "low", "medium", "high"]
    colors = {"neutral": "#000000", "low": "#444444", "medium": "#777777", "high": "#aaaaaa"}
    plt.figure(figsize=(8.0, 5.5))
    for regime in regimes:
        vals = np.asarray([float(r["EII_01"]) for r in rows if r["regime"] == regime], dtype=float)
        if vals.size == 0:
            continue
        plt.hist(
            vals,
            bins=25,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=colors[regime],
            label=regime,
        )
    plt.axvline(default_threshold, linestyle="--", color="#222222", linewidth=1.2, label=f"threshold={default_threshold:.2f}")
    plt.xlabel("EII_01")
    plt.ylabel("Density")
    plt.title("Global EII_01 distributions by latent regime")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig1 = out / "figure_eii01_distribution.png"
    plt.savefig(fig1, dpi=300)
    plt.close()

    # Figure 2: Global ROC.
    fpr = np.asarray([float(r["FPR"]) for r in roc_rows], dtype=float)
    tpr = np.asarray([float(r["TPR"]) for r in roc_rows], dtype=float)
    auc = _trapz_compat(tpr[np.argsort(fpr)], fpr[np.argsort(fpr)]) if fpr.size >= 2 else float("nan")
    plt.figure(figsize=(5.8, 5.8))
    plt.plot(fpr, tpr, color="#111111", linewidth=1.8)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#888888")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Global ROC (AUC={auc:.3f})")
    plt.tight_layout()
    fig2 = out / "figure_global_roc.png"
    plt.savefig(fig2, dpi=300)
    plt.close()

    # Figure 3: Recoverability stability vs mean EII_01.
    eii = np.asarray([float(r["mean_EII_01"]) for r in rec_rows], dtype=float)
    stability = np.asarray([float(r["pairwise_threshold_consistency"]) for r in rec_rows], dtype=float)
    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(eii, stability, s=20, c="#222222", alpha=0.7)
    plt.axvline(default_threshold, linestyle="--", color="#666666", linewidth=1.0)
    plt.xlabel("Scenario mean EII_01")
    plt.ylabel("Pairwise threshold consistency")
    plt.title("Replicate recoverability vs EII_01")
    plt.tight_layout()
    fig3 = out / "figure_recoverability_scatter.png"
    plt.savefig(fig3, dpi=300)
    plt.close()

    # Figure 4: Bar chart of selected threshold metrics (global).
    targets = [
        ("fixed_0.70", "0.70"),
        ("fixed_0.90", "0.90"),
        ("neutral_q95", "neutral95"),
        ("neutral_q99", "neutral99"),
        ("max_youden", "maxJ"),
    ]
    selected = []
    for key, label in targets:
        row = next((r for r in perf_rows if r["threshold_name"] == key), None)
        if row is None:
            continue
        selected.append((label, float(row["balanced_accuracy"]), float(row["TPR"]), float(row["FPR"])))

    if selected:
        x = np.arange(len(selected))
        bal = np.asarray([s[1] for s in selected], dtype=float)
        tpr_v = np.asarray([s[2] for s in selected], dtype=float)
        fpr_v = np.asarray([s[3] for s in selected], dtype=float)

        width = 0.25
        plt.figure(figsize=(8.2, 5.4))
        plt.bar(x - width, bal, width=width, color="#222222", label="Balanced acc")
        plt.bar(x, tpr_v, width=width, color="#777777", label="TPR")
        plt.bar(x + width, fpr_v, width=width, color="#bbbbbb", label="FPR")
        plt.xticks(x, [s[0] for s in selected])
        plt.ylim(0, 1)
        plt.ylabel("Metric value")
        plt.title("Global threshold performance summary")
        plt.legend(frameon=False)
        plt.tight_layout()
        fig4 = out / "figure_threshold_metrics.png"
        plt.savefig(fig4, dpi=300)
        plt.close()
    else:
        fig4 = out / "figure_threshold_metrics.png"

    # Figure 5: q_emp significant rates by latent regime.
    if sig_regime_rows:
        labels = [r["regime"] for r in sig_regime_rows]
        vals = np.asarray([float(r["significant_rate_q"]) for r in sig_regime_rows], dtype=float)
        plt.figure(figsize=(7.0, 4.6))
        plt.bar(np.arange(len(labels)), vals, color="#3a3a3a", width=0.65)
        plt.ylim(0, 1)
        plt.xticks(np.arange(len(labels)), labels)
        plt.ylabel("Significant rate (q_emp <= alpha)")
        plt.title("Significance rates by latent regime")
        plt.tight_layout()
        fig5 = out / "figure_significance_rates.png"
        plt.savefig(fig5, dpi=300)
        plt.close()
    else:
        fig5 = out / "figure_significance_rates.png"

    # Figure 6: Neutral p_emp diagnostics (histogram + QQ).
    if neutral_hist_rows and neutral_qq_rows:
        left = np.asarray([float(r["bin_left"]) for r in neutral_hist_rows], dtype=float)
        right = np.asarray([float(r["bin_right"]) for r in neutral_hist_rows], dtype=float)
        counts = np.asarray([float(r["count"]) for r in neutral_hist_rows], dtype=float)
        expected = np.asarray([float(r["expected_quantile"]) for r in neutral_qq_rows], dtype=float)
        observed = np.asarray([float(r["observed_quantile"]) for r in neutral_qq_rows], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
        axes[0].bar(left, counts, width=(right - left), align="edge", color="#555555", edgecolor="#222222")
        axes[0].set_xlabel("neutral p_emp")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Neutral p_emp histogram")

        axes[1].plot(expected, observed, color="#111111", linewidth=1.4)
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0)
        axes[1].set_xlabel("Expected quantile (Uniform)")
        axes[1].set_ylabel("Observed neutral p_emp quantile")
        axes[1].set_title("Neutral p_emp QQ")
        fig.tight_layout()
        fig6 = out / "figure_neutral_pemp_diagnostics.png"
        fig.savefig(fig6, dpi=300)
        plt.close(fig)
    else:
        fig6 = out / "figure_neutral_pemp_diagnostics.png"

    return {
        "figure_eii01_distribution": str(fig1),
        "figure_global_roc": str(fig2),
        "figure_recoverability_scatter": str(fig3),
        "figure_threshold_metrics": str(fig4),
        "figure_significance_rates": str(fig5),
        "figure_neutral_pemp_diagnostics": str(fig6),
    }


def _global_row(rows: Sequence[Mapping[str, str]], threshold_name: str) -> Optional[Mapping[str, str]]:
    for row in rows:
        if row.get("group") == "global" and row.get("threshold_name") == threshold_name:
            return row
    return None


def generate_manuscript_outputs(
    *,
    metrics_tsv: str | Path,
    quantiles_tsv: str | Path,
    threshold_perf_tsv: str | Path,
    significance_performance_tsv: str | Path,
    significance_regime_rates_tsv: str | Path,
    significance_summary_json: str | Path,
    bootstrap_summary_tsv: str | Path,
    recoverability_stratified_tsv: str | Path,
    outdir: str | Path,
    default_threshold: float,
    alpha: float,
    decision_target: str,
) -> Dict[str, Any]:
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    quant_rows = read_tsv(quantiles_tsv)
    perf_rows = read_tsv(threshold_perf_tsv)
    sig_perf_rows = read_tsv(significance_performance_tsv)
    sig_regime_rows = read_tsv(significance_regime_rates_tsv)
    sig_summary = json.loads(Path(significance_summary_json).read_text())
    boot_rows = read_tsv(bootstrap_summary_tsv)
    rec_rows = read_tsv(recoverability_stratified_tsv)

    global_quant = next((r for r in quant_rows if r["group"] == "global"), None)
    if global_quant is None:
        raise RuntimeError("Global quantile row missing.")

    perf_070 = _global_row(perf_rows, "fixed_0.70")
    perf_090 = _global_row(perf_rows, "fixed_0.90")
    perf_q95 = _global_row(perf_rows, "neutral_q95")
    perf_q99 = _global_row(perf_rows, "neutral_q99")

    if perf_070 is None:
        raise RuntimeError("Global fixed_0.70 performance row missing.")

    auc_global = float(perf_070["AUC"])
    q95 = float(global_quant["neutral_q95_EII_01"])
    q99 = float(global_quant["neutral_q99_EII_01"])
    fpr_070 = float(perf_070["FPR"])
    tpr_070 = float(perf_070["TPR"])
    bal_070 = float(perf_070["balanced_accuracy"])
    q_perf = next((r for r in sig_perf_rows if r.get("method") == "q_emp"), None)
    if q_perf is None:
        raise RuntimeError("q_emp performance row missing.")

    neutral_sig_rate = float(sig_summary["neutral_significant_rate_q"])
    mh_sig_rate = float(sig_summary["medium_high_significant_rate_q"])
    neutral_ks = float(sig_summary["neutral_p_uniformity_ks"])
    q_fpr = float(sig_summary["q_emp_metrics"]["FPR"])
    q_tpr = float(sig_summary["q_emp_metrics"]["TPR"])
    q_bal = float(sig_summary["q_emp_metrics"]["balanced_accuracy"])
    threshold_bal = float(sig_summary["eii_threshold_metrics"]["balanced_accuracy"])
    delta_bal = float(sig_summary["q_vs_eii_delta_balanced_accuracy"])

    support_flag = (
        "acceptable"
        if (
            np.isfinite(neutral_sig_rate)
            and np.isfinite(neutral_ks)
            and neutral_sig_rate <= alpha * 1.5
            and neutral_ks <= 0.25
            and mh_sig_rate >= neutral_sig_rate
        )
        else "needs_review"
    )
    support_text = (
        f"q_emp significance calibration is {support_flag} at alpha={alpha:.2f}; "
        f"neutral significant rate={neutral_sig_rate:.3f}, medium/high significant rate={mh_sig_rate:.3f}, "
        f"neutral p_emp KS deviation={neutral_ks:.3f}."
    )

    methods_text = "\n".join(
        [
            "### Validation calibration methods",
            "Full-pipeline synthetic alignments were simulated under explicit latent burden regimes (neutral/low/medium/high) and nuisance strata.",
            "For each replicate, BABAPPAi inference was run without replacing core inference code.",
            "The observed dispersion statistic D_obs was extracted from model outputs; neutral calibration parameters (mu0, sigma0) were derived from matched neutral replicate distributions, with sigma0 floored at a user-configurable minimum before EII_z was computed.",
            "Empirical p-values were computed as p_emp=(1+count(D0>=D_obs))/(M+1), where D0 are matched neutral replicate dispersion values.",
            "Multiple testing control used Benjamini-Hochberg q_emp across genes in the analysis set, with significance defined as q_emp<=alpha.",
            "Threshold-based EII_01 summaries were retained only as descriptive/legacy comparisons.",
            "Bootstrap uncertainty intervals were computed by gene-level resampling with replacement.",
            "EII was interpreted as an identifiability/recoverability diagnostic, not direct evidence of adaptive substitution.",
        ]
    )

    perf_090_text = (
        ""
        if perf_090 is None
        else f"FPR={float(perf_090['FPR']):.3f}, TPR={float(perf_090['TPR']):.3f}"
    )
    perf_q95_text = (
        ""
        if perf_q95 is None
        else f"FPR={float(perf_q95['FPR']):.3f}, TPR={float(perf_q95['TPR']):.3f}"
    )
    perf_q99_text = (
        ""
        if perf_q99 is None
        else f"FPR={float(perf_q99['FPR']):.3f}, TPR={float(perf_q99['TPR']):.3f}"
    )

    report_lines = [
        "# Full-Pipeline Validation Report",
        "",
        "## Global calibration summary",
        f"- Decision target: `{decision_target}`",
        f"- Neutral 95th percentile (EII_01): `{q95:.3f}`",
        f"- Neutral 99th percentile (EII_01): `{q99:.3f}`",
        f"- FPR at 0.70: `{fpr_070:.3f}`",
        f"- TPR at 0.70: `{tpr_070:.3f}`",
        f"- Balanced accuracy at 0.70: `{bal_070:.3f}`",
        f"- AUC: `{auc_global:.3f}`",
        "## Significance framework summary",
        f"- alpha: `{alpha:.2f}`",
        f"- neutral significant rate (q_emp <= alpha): `{neutral_sig_rate:.3f}`",
        f"- medium/high significant rate (q_emp <= alpha): `{mh_sig_rate:.3f}`",
        f"- q_emp FPR: `{q_fpr:.3f}`",
        f"- q_emp TPR: `{q_tpr:.3f}`",
        f"- q_emp balanced accuracy: `{q_bal:.3f}`",
        f"- neutral p_emp KS deviation from Uniform(0,1): `{neutral_ks:.3f}`",
        f"- significance calibration status: `{support_flag}`",
        "",
        "## Legacy threshold comparison (descriptive only)",
        f"- {support_text}",
        f"- EII threshold balanced accuracy: `{threshold_bal:.3f}`",
        f"- Delta balanced accuracy (q_emp - EII threshold): `{delta_bal:.3f}`",
        "",
        "## Comparison thresholds",
        f"- 0.90: `{perf_090_text}`",
        f"- neutral95: `{perf_q95_text}`",
        f"- neutral99: `{perf_q99_text}`",
        "",
        "## Bootstrap uncertainty (95% intervals)",
    ]
    for row in boot_rows:
        report_lines.append(
            f"- {row['metric']}: mean={float(row['mean']):.3f}, CI=[{float(row['ci_lower_2_5']):.3f}, {float(row['ci_upper_97_5']):.3f}]"
        )

    report_lines.extend(
        [
            "",
            "## Reproducibility summary (scenario-level)",
        ]
    )
    for row in rec_rows:
        report_lines.append(
            "- "
            f"{row['stratum']}: n={int(float(row['n_scenarios']))}, "
            f"mean_threshold_consistency={float(row['mean_pairwise_threshold_consistency']):.3f}, "
            f"mean_significance_consistency={float(row.get('mean_pairwise_significance_consistency', float('nan'))):.3f}, "
            f"mean_var_EII_01={float(row['mean_var_EII_01']):.4f}"
        )

    report_lines.extend(
        [
            "",
            "## Limitations",
            "- Synthetic data use simplified codon-level simulation and are not intended as complete biological realism.",
            "- Some fine-grained strata may have limited sample size in modest benchmark runs.",
            "- Empirical p-values and q-values are conditional on the matched neutral simulator and calibration adequacy.",
            "- Significance indicates excess dispersion relative to matched neutral expectation, not proof of adaptive substitution.",
        ]
    )

    report_path = out / "validation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    methods_path = out / "methods_text_block.md"
    methods_path.write_text(methods_text + "\n")

    paper_paragraph = (
        "In full-pipeline synthetic validation across nuisance-stratified scenarios, "
        f"empirical significance at q_emp<= {alpha:.2f} yielded a neutral significant rate of {neutral_sig_rate:.2f} "
        f"and a medium/high significant rate of {mh_sig_rate:.2f}, with AUC={auc_global:.2f} for EII-based ranking "
        f"under the {decision_target} decision target. Threshold-style EII_01 summaries were retained only as descriptive "
        "comparators. Consistent with framework intent, EII was treated as a recoverability magnitude diagnostic, while "
        "p_emp/q_emp quantified excess dispersion relative to matched neutral calibration."
    )
    paper_path = out / "manuscript_ready_results_text.md"
    paper_path.write_text(paper_paragraph + "\n")

    interpretation = (
        f"Significance calibration ({support_flag}) at q_emp<= {alpha:.2f}: {support_text} "
        "EII thresholds are descriptive only and are not used as the primary inferential decision rule."
    )
    interp_path = out / "threshold_interpretation.txt"
    interp_path.write_text(interpretation + "\n")

    release_notes_text = "\n".join(
        [
            "# BABAPPAi 2.1.0 Validation Release Notes",
            "",
            "- Primary inferential decision layer now uses empirical Monte Carlo p-values and BH-adjusted q-values.",
            "- EII metrics remain diagnostic effect-size style outputs.",
            f"- Validation neutral significant rate at q<=alpha: {neutral_sig_rate:.3f}",
            f"- Validation medium/high significant rate at q<=alpha: {mh_sig_rate:.3f}",
            f"- q_emp FPR={q_fpr:.3f}, TPR={q_tpr:.3f}, balanced_accuracy={q_bal:.3f}",
            "- Interpretation guardrail: significance means excess dispersion relative to matched neutral calibration, not proof of adaptation.",
        ]
    )
    release_notes_path = out / "release_notes.md"
    release_notes_path.write_text(release_notes_text + "\n")

    return {
        "report_md": str(report_path),
        "methods_md": str(methods_path),
        "paper_text_md": str(paper_path),
        "interpretation_txt": str(interp_path),
        "release_notes_md": str(release_notes_path),
        "support_flag": support_flag,
    }


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def run_full_pipeline_validation(
    *,
    outdir: str | Path,
    n_per_regime: int,
    n_replicates_per_scenario: int,
    bootstrap_reps: int,
    default_threshold: float,
    decision_target: str,
    seed: int,
    tree_calibration: bool = False,
    n_calibration: int = 200,
    device: str = "cpu",
    batch_size: int = 1,
    sigma_floor: float = 0.05,
    alpha: float = 0.05,
    pvalue_mode: str = "empirical_monte_carlo",
    min_neutral_group_size: int = 20,
    neutral_reps: int = 200,
    offline: bool = False,
    overwrite: bool = True,
) -> Dict[str, Any]:
    if not (0.0 <= default_threshold <= 1.0):
        raise ValueError("default_threshold must be in [0, 1].")
    if n_per_regime <= 0:
        raise ValueError("n_per_regime must be > 0.")
    if n_replicates_per_scenario <= 0:
        raise ValueError("n_replicates_per_scenario must be > 0.")
    if sigma_floor < 0:
        raise ValueError("sigma_floor must be >= 0.")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    dataset_dir = out / "dataset"
    inference_dir = out / "inference"
    calibration_dir = out / "calibration"
    significance_dir = out / "significance"
    bootstrap_dir = out / "bootstrap"
    reproducibility_dir = out / "reproducibility"
    figures_dir = out / "figures"
    report_dir = out / "report"

    dataset_meta = simulate_alignment_validation_dataset(
        outdir=dataset_dir,
        n_per_regime=n_per_regime,
        n_replicates_per_scenario=n_replicates_per_scenario,
        seed=seed,
    )

    inference_meta = run_full_pipeline_inference_on_dataset(
        dataset_tsv=dataset_meta["dataset_tsv"],
        outdir=inference_dir,
        tree_calibration=tree_calibration,
        n_calibration=n_calibration,
        device=device,
        batch_size=batch_size,
        sigma_floor=sigma_floor,
        alpha=alpha,
        pvalue_mode=pvalue_mode,
        min_neutral_group_size=min_neutral_group_size,
        neutral_reps=neutral_reps,
        offline=offline,
        overwrite=overwrite,
    )

    calibration_meta = compute_threshold_calibration(
        metrics_tsv=inference_meta["metrics_tsv"],
        outdir=calibration_dir,
        decision_target=decision_target,
        default_threshold=default_threshold,
    )

    significance_meta = compute_significance_calibration(
        metrics_tsv=inference_meta["metrics_tsv"],
        outdir=significance_dir,
        decision_target=decision_target,
        alpha=alpha,
        default_threshold=default_threshold,
    )

    bootstrap_meta = bootstrap_eii_thresholds(
        metrics_tsv=inference_meta["metrics_tsv"],
        outdir=bootstrap_dir,
        bootstrap_reps=bootstrap_reps,
        seed=seed + 71,
        decision_target=decision_target,
        default_threshold=default_threshold,
        alpha=alpha,
    )

    reproducibility_meta = analyze_replicate_recoverability(
        metrics_tsv=inference_meta["metrics_tsv"],
        outdir=reproducibility_dir,
        default_threshold=default_threshold,
        alpha=alpha,
    )

    figures_meta = make_validation_figures(
        metrics_tsv=inference_meta["metrics_tsv"],
        roc_tsv=calibration_meta["roc_tsv"],
        threshold_perf_tsv=calibration_meta["performance_tsv"],
        significance_regime_tsv=significance_meta["regime_rates_tsv"],
        neutral_p_hist_tsv=significance_meta["neutral_hist_tsv"],
        neutral_p_qq_tsv=significance_meta["neutral_qq_tsv"],
        recoverability_tsv=reproducibility_meta["scenario_recoverability_tsv"],
        outdir=figures_dir,
        default_threshold=default_threshold,
    )

    report_meta = generate_manuscript_outputs(
        metrics_tsv=inference_meta["metrics_tsv"],
        quantiles_tsv=calibration_meta["quantiles_tsv"],
        threshold_perf_tsv=calibration_meta["performance_tsv"],
        significance_performance_tsv=significance_meta["performance_tsv"],
        significance_regime_rates_tsv=significance_meta["regime_rates_tsv"],
        significance_summary_json=significance_meta["summary_json"],
        bootstrap_summary_tsv=bootstrap_meta["bootstrap_summary_tsv"],
        recoverability_stratified_tsv=reproducibility_meta["stratified_tsv"],
        outdir=report_dir,
        default_threshold=default_threshold,
        alpha=alpha,
        decision_target=decision_target,
    )

    perf_rows = read_tsv(calibration_meta["performance_tsv"])
    quant_rows = read_tsv(calibration_meta["quantiles_tsv"])
    global_q = next((r for r in quant_rows if r["group"] == "global"), None)
    global_070 = next((r for r in perf_rows if r["group"] == "global" and r["threshold_name"] == "fixed_0.70"), None)

    if global_q is None or global_070 is None:
        raise RuntimeError("Global summary rows missing from calibration outputs.")

    summary = {
        "global_neutral_q95_eii01": float(global_q["neutral_q95_EII_01"]),
        "global_neutral_q99_eii01": float(global_q["neutral_q99_EII_01"]),
        "global_fpr_at_070": float(global_070["FPR"]),
        "global_tpr_at_070": float(global_070["TPR"]),
        "global_auc": float(global_070["AUC"]),
        "neutral_significant_rate_q": float(significance_meta["summary"]["neutral_significant_rate_q"]),
        "medium_high_significant_rate_q": float(significance_meta["summary"]["medium_high_significant_rate_q"]),
        "neutral_p_uniformity_ks": float(significance_meta["summary"]["neutral_p_uniformity_ks"]),
        "fraction_sigma0_at_floor": float(inference_meta["fraction_sigma0_at_floor"]),
        "fraction_fallback_applied": float(inference_meta["fraction_fallback_applied"]),
        "q_emp_fpr": float(significance_meta["summary"]["q_emp_metrics"]["FPR"]),
        "q_emp_tpr": float(significance_meta["summary"]["q_emp_metrics"]["TPR"]),
        "threshold_support_flag": report_meta["support_flag"],
        "sigma0_raw_summary": inference_meta["sigma0_before_floor_summary"],
        "sigma0_final_summary": inference_meta["sigma0_after_floor_summary"],
    }
    summary["significance_calibration_status"] = str(report_meta["support_flag"])

    _write_json(
        out / "run_manifest.json",
        {
            "dataset": dataset_meta,
            "inference": inference_meta,
            "calibration": calibration_meta,
            "significance": significance_meta,
            "bootstrap": bootstrap_meta,
            "reproducibility": reproducibility_meta,
            "figures": figures_meta,
            "report": report_meta,
            "console_summary": summary,
        },
    )

    return {
        **summary,
        "outdir": str(out),
        "dataset_tsv": dataset_meta["dataset_tsv"],
        "metrics_tsv": inference_meta["metrics_tsv"],
        "calibration_debug_tsv": inference_meta["debug_tsv"],
        "quantiles_tsv": calibration_meta["quantiles_tsv"],
        "performance_tsv": calibration_meta["performance_tsv"],
        "significance_summary_json": significance_meta["summary_json"],
        "significance_performance_tsv": significance_meta["performance_tsv"],
        "significance_regime_rates_tsv": significance_meta["regime_rates_tsv"],
        "bootstrap_summary_tsv": bootstrap_meta["bootstrap_summary_tsv"],
        "recoverability_tsv": reproducibility_meta["stratified_tsv"],
        "report_md": report_meta["report_md"],
        "paper_text_md": report_meta["paper_text_md"],
        "methods_md": report_meta["methods_md"],
        "interpretation_txt": report_meta["interpretation_txt"],
        "release_notes_md": report_meta["release_notes_md"],
    }


__all__ = [
    "analyze_replicate_recoverability",
    "bootstrap_eii_thresholds",
    "compute_significance_calibration",
    "compute_threshold_calibration",
    "generate_manuscript_outputs",
    "make_validation_figures",
    "read_tsv",
    "run_full_pipeline_inference_on_dataset",
    "run_full_pipeline_validation",
    "simulate_alignment_validation_dataset",
]
