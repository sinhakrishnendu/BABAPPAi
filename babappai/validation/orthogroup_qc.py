"""Orthogroup quality-control utilities for empirical validation selection."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from Bio import SeqIO


SUPPORTED_ALIGNMENT_SUFFIXES = (
    ".fasta",
    ".fa",
    ".fas",
    ".fna",
    ".ffn",
    ".aln",
)


def list_orthogroup_alignments(input_dir: str) -> List[Path]:
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input orthogroup directory not found: {root}")
    alignments = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_ALIGNMENT_SUFFIXES
    ]
    return sorted(alignments)


def orthogroup_id_from_path(path: Path) -> str:
    suffixes = "".join(path.suffixes)
    if suffixes:
        return path.name[: -len(suffixes)]
    return path.stem


def species_from_header(header: str) -> str:
    token = header.strip().split()[0]
    if "|" in token:
        return token.split("|")[0]
    if "__" in token:
        return token.split("__")[0]
    return token.split("_")[0]


def _pairwise_divergence(seqs: List[str]) -> Tuple[float, float]:
    if len(seqs) < 2:
        return 0.0, 0.0

    distances = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            a, b = seqs[i], seqs[j]
            L = min(len(a), len(b))
            if L == 0:
                continue
            mismatch = sum(1 for k in range(L) if a[k] != b[k])
            distances.append(mismatch / L)

    if not distances:
        return 0.0, 0.0
    return float(statistics.mean(distances)), float(statistics.pstdev(distances))


def _internal_stop_count(seq: str) -> int:
    stop_codons = {"TAA", "TAG", "TGA"}
    if len(seq) < 6 or len(seq) % 3 != 0:
        return 0
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    return sum(1 for codon in codons[:-1] if codon in stop_codons)


def compute_orthogroup_metrics(path: Path, dataset_max_taxa: int) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "orthogroup_id": orthogroup_id_from_path(path),
        "alignment_path": str(path),
        "readable_fasta": False,
        "n_sequences": 0,
        "n_unique_species": 0,
        "duplicate_species_count": 0,
        "occupancy": 0.0,
        "median_length_nt": 0.0,
        "missingness": 1.0,
        "internal_stop_count": 0,
        "mean_divergence": 0.0,
        "divergence_sd": 0.0,
        "divergence_balance": 0.0,
        "provisional_alignment_quality": 0.0,
        "tree_recoverability": 0.0,
        "qc_score": 0.0,
        "tree_path": "",
    }

    try:
        records = list(SeqIO.parse(path, "fasta"))
    except Exception:
        return metrics

    if not records:
        return metrics

    seqs = [str(rec.seq).upper() for rec in records]
    ids = [rec.id for rec in records]

    species = [species_from_header(identifier) for identifier in ids]
    unique_species = sorted(set(species))
    duplicate_species_count = len(species) - len(unique_species)

    lengths = [len(seq) for seq in seqs]
    median_length = float(statistics.median(lengths))

    missing_chars = set("N?-")
    total_chars = sum(len(seq) for seq in seqs) or 1
    missing = sum(sum(1 for ch in seq if ch in missing_chars) for seq in seqs)
    missingness = missing / total_chars

    internal_stop_count = sum(_internal_stop_count(seq) for seq in seqs)
    mean_div, sd_div = _pairwise_divergence(seqs)

    # Favor moderate divergence (avoid both near-identical and over-saturated loci).
    divergence_balance = max(0.0, 1.0 - abs(mean_div - 0.15) / 0.15)
    provisional_alignment_quality = max(0.0, 1.0 - missingness)
    tree_recoverability = max(0.0, min(1.0, (mean_div * 4.0)))

    occupancy = len(seqs) / max(dataset_max_taxa, 1)

    qc_score = (
        0.24 * occupancy
        + 0.18 * (1.0 - min(1.0, duplicate_species_count / max(len(seqs), 1)))
        + 0.18 * min(1.0, median_length / 1500.0)
        + 0.14 * (1.0 - missingness)
        + 0.14 * divergence_balance
        + 0.12 * tree_recoverability
    )

    metrics.update(
        {
            "readable_fasta": True,
            "n_sequences": len(seqs),
            "n_unique_species": len(unique_species),
            "duplicate_species_count": duplicate_species_count,
            "occupancy": occupancy,
            "median_length_nt": median_length,
            "missingness": missingness,
            "internal_stop_count": internal_stop_count,
            "mean_divergence": mean_div,
            "divergence_sd": sd_div,
            "divergence_balance": divergence_balance,
            "provisional_alignment_quality": provisional_alignment_quality,
            "tree_recoverability": tree_recoverability,
            "qc_score": qc_score,
        }
    )
    return metrics


def apply_hard_filters(
    *,
    metrics: Dict[str, object],
    min_taxa: int,
    occupancy_threshold: float,
    min_length_nt: int,
    max_missingness: float,
    enforce_one_to_one: bool,
    require_no_internal_stops: bool,
) -> Dict[str, object]:
    reasons = []
    if not metrics["readable_fasta"]:
        reasons.append("unreadable_fasta")
    if int(metrics["n_sequences"]) < min_taxa:
        reasons.append("min_taxa")
    if float(metrics["occupancy"]) < occupancy_threshold:
        reasons.append("occupancy_threshold")
    if float(metrics["median_length_nt"]) < min_length_nt:
        reasons.append("length_threshold")
    if float(metrics["missingness"]) > max_missingness:
        reasons.append("missingness_threshold")
    if enforce_one_to_one and int(metrics["duplicate_species_count"]) > 0:
        reasons.append("duplicate_species")
    if require_no_internal_stops and int(metrics["internal_stop_count"]) > 0:
        reasons.append("internal_stop_codons")

    return {
        **metrics,
        "hard_filter_pass": not reasons,
        "rejection_reasons": ",".join(reasons),
    }
