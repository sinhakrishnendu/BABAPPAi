"""Input quality-control helpers for empirical and inference preflight checks."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from babappai.encoding import CODON_TO_ID, PAD_ID, encode_alignment
from babappai.tree import enumerate_branches, load_tree

STOP_CODONS = {"TAA", "TAG", "TGA"}
_BASES = {"A", "C", "G", "T"}
_ACC_RE = re.compile(r"^[A-Z]{2,4}_(\d+)(?:\.\d+)?$")
_BRACKET_SPECIES_RE = re.compile(r"\[([A-Za-z][A-Za-z _.-]+)\]")


class InputPreflightError(ValueError):
    """Raised when alignment/tree inputs fail deterministic preflight checks."""


@dataclass(frozen=True)
class SpeciesKey:
    key: str
    method: str


def _normalize_seq(seq: str) -> str:
    return seq.upper().replace("U", "T")


def infer_species_key(
    record_id: str,
    description: str = "",
    *,
    accession_prefix_len: int = 6,
) -> SpeciesKey:
    """Infer a stable species key from a FASTA header.

    If no explicit species label can be found, accession-like IDs fall back to a
    deterministic numeric-prefix proxy so duplicate isoforms/paralogs from the
    same accession block can still be collapsed reproducibly.
    """

    token = (record_id or "").strip() or "unknown_record"
    desc = (description or "").strip()

    bracket_match = _BRACKET_SPECIES_RE.search(desc)
    if bracket_match:
        species = bracket_match.group(1).strip().lower().replace(" ", "_")
        return SpeciesKey(species, "description_brackets")

    for delimiter, method in (("|", "token_pipe"), ("__", "token_double_underscore")):
        if delimiter in token:
            left = token.split(delimiter, 1)[0].strip()
            if left and re.search(r"[A-Za-z]", left):
                return SpeciesKey(left.lower(), method)

    if "_" in token:
        left = token.split("_", 1)[0].strip()
        if left and len(left) >= 3 and re.search(r"[A-Za-z]", left):
            return SpeciesKey(left.lower(), "token_prefix")

    acc_match = _ACC_RE.match(token)
    if acc_match:
        digits = acc_match.group(1)
        prefix = digits[: max(1, int(accession_prefix_len))]
        return SpeciesKey(f"accprefix_{prefix}", f"accession_prefix_{int(accession_prefix_len)}")

    return SpeciesKey(token.lower(), "record_id_fallback")


def _record_quality_key(rec: SeqRecord) -> Tuple[int, int, str]:
    seq = _normalize_seq(str(rec.seq))
    ambiguous = sum(ch not in _BASES for ch in seq)
    return (ambiguous, -len(seq), rec.id)


def deduplicate_species_records(
    records: Sequence[SeqRecord],
    *,
    gene_name: str,
    accession_prefix_len: int = 6,
) -> Tuple[List[SeqRecord], List[Dict[str, Any]]]:
    """Enforce at most one sequence per inferred species key.

    Selection rule is deterministic: fewest ambiguous bases, then longest CDS,
    then lexicographically smallest record ID.
    """

    by_species: Dict[str, List[Tuple[SeqRecord, SpeciesKey]]] = defaultdict(list)
    for rec in records:
        sk = infer_species_key(rec.id, rec.description, accession_prefix_len=accession_prefix_len)
        by_species[sk.key].append((rec, sk))

    kept: List[SeqRecord] = []
    audit_rows: List[Dict[str, Any]] = []

    for species_key in sorted(by_species):
        group = by_species[species_key]
        ranked = sorted(group, key=lambda item: _record_quality_key(item[0]))
        chosen_rec, chosen_sk = ranked[0]
        kept.append(chosen_rec)
        if len(ranked) == 1:
            continue
        for discarded_rec, discarded_sk in ranked[1:]:
            audit_rows.append(
                {
                    "gene_name": gene_name,
                    "species_key": species_key,
                    "species_key_method": discarded_sk.method,
                    "kept_record_id": chosen_rec.id,
                    "discarded_record_id": discarded_rec.id,
                    "discard_reason": "duplicate_species_or_isoform",
                    "selection_rule": "fewest_ambiguous_then_longest_then_lexicographic",
                    "kept_length_nt": len(str(chosen_rec.seq)),
                    "discarded_length_nt": len(str(discarded_rec.seq)),
                }
            )

    kept_sorted = sorted(kept, key=lambda r: r.id)
    return kept_sorted, audit_rows


def cap_records_for_model_support(
    records: Sequence[SeqRecord],
    *,
    gene_name: str,
    max_sequences: int,
) -> Tuple[List[SeqRecord], List[Dict[str, Any]]]:
    """Deterministically cap record count to the model-supported maximum."""

    if max_sequences <= 0:
        raise ValueError("max_sequences must be > 0")
    ranked = sorted(records, key=_record_quality_key)
    kept = ranked[:max_sequences]
    dropped = ranked[max_sequences:]
    audit_rows: List[Dict[str, Any]] = []
    for rec in dropped:
        audit_rows.append(
            {
                "gene_name": gene_name,
                "species_key": "",
                "species_key_method": "",
                "kept_record_id": "",
                "discarded_record_id": rec.id,
                "discard_reason": "model_support_taxon_cap",
                "selection_rule": (
                    f"max_sequences={max_sequences}; kept_by=fewest_ambiguous_then_longest_then_lexicographic"
                ),
                "kept_length_nt": "",
                "discarded_length_nt": len(str(rec.seq)),
            }
        )
    return sorted(kept, key=lambda r: r.id), audit_rows


def audit_codon_records(
    records: Sequence[SeqRecord],
    *,
    gene_name: str,
    stage: str,
    allow_gap_triplet: bool = True,
    max_examples: int = 12,
) -> Dict[str, Any]:
    """Audit codon cleanliness for a sequence collection."""

    sense_codons = set(CODON_TO_ID.keys())
    codon_triplets_total = 0
    sense_codon_count = 0
    gap_triplet_count = 0
    stop_codon_count = 0
    ambiguous_codon_count = 0
    partial_codon_count = 0
    mixed_gap_codon_count = 0
    oov_codon_count = 0
    invalid_seq_ids: set[str] = set()
    invalid_examples: List[str] = []

    for rec in records:
        seq = _normalize_seq(str(rec.seq))
        if len(seq) % 3 != 0:
            partial_codon_count += 1
            invalid_seq_ids.add(rec.id)
            if len(invalid_examples) < max_examples:
                invalid_examples.append(f"{rec.id}:PARTIAL_LENGTH:{len(seq)}")
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            codon_triplets_total += 1
            if codon in sense_codons:
                sense_codon_count += 1
                continue
            if allow_gap_triplet and codon == "---":
                gap_triplet_count += 1
                continue
            if codon in STOP_CODONS:
                stop_codon_count += 1
                invalid_seq_ids.add(rec.id)
                if len(invalid_examples) < max_examples:
                    invalid_examples.append(f"{rec.id}:{i // 3 + 1}:{codon}:stop")
                continue
            if "-" in codon:
                mixed_gap_codon_count += 1
                invalid_seq_ids.add(rec.id)
                if len(invalid_examples) < max_examples:
                    invalid_examples.append(f"{rec.id}:{i // 3 + 1}:{codon}:mixed_gap")
                continue
            if any(ch not in _BASES for ch in codon):
                ambiguous_codon_count += 1
                invalid_seq_ids.add(rec.id)
                if len(invalid_examples) < max_examples:
                    invalid_examples.append(f"{rec.id}:{i // 3 + 1}:{codon}:ambiguous")
                continue
            oov_codon_count += 1
            invalid_seq_ids.add(rec.id)
            if len(invalid_examples) < max_examples:
                invalid_examples.append(f"{rec.id}:{i // 3 + 1}:{codon}:oov")

    invalid_total = (
        stop_codon_count
        + ambiguous_codon_count
        + partial_codon_count
        + mixed_gap_codon_count
        + oov_codon_count
    )
    return {
        "gene_name": gene_name,
        "stage": stage,
        "n_sequences": len(records),
        "codon_triplets_total": codon_triplets_total,
        "sense_codon_count": sense_codon_count,
        "gap_triplet_count": gap_triplet_count,
        "stop_codon_count": stop_codon_count,
        "ambiguous_codon_count": ambiguous_codon_count,
        "partial_codon_count": partial_codon_count,
        "mixed_gap_codon_count": mixed_gap_codon_count,
        "oov_codon_count": oov_codon_count,
        "invalid_total": invalid_total,
        "invalid_sequence_count": len(invalid_seq_ids),
        "invalid_examples": "; ".join(invalid_examples),
        "allow_gap_triplet": bool(allow_gap_triplet),
        "valid_bool": bool(invalid_total == 0),
    }


def audit_codon_fasta(
    fasta_path: Path | str,
    *,
    gene_name: str,
    stage: str,
    allow_gap_triplet: bool = True,
    max_examples: int = 12,
) -> Dict[str, Any]:
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        return {
            "gene_name": gene_name,
            "stage": stage,
            "n_sequences": 0,
            "codon_triplets_total": 0,
            "sense_codon_count": 0,
            "gap_triplet_count": 0,
            "stop_codon_count": 0,
            "ambiguous_codon_count": 0,
            "partial_codon_count": 0,
            "mixed_gap_codon_count": 0,
            "oov_codon_count": 0,
            "invalid_total": 1,
            "invalid_sequence_count": 0,
            "invalid_examples": "empty_alignment",
            "allow_gap_triplet": bool(allow_gap_triplet),
            "valid_bool": False,
        }
    return audit_codon_records(
        records,
        gene_name=gene_name,
        stage=stage,
        allow_gap_triplet=allow_gap_triplet,
        max_examples=max_examples,
    )


def _build_branch_to_taxa_indices(tree) -> List[List[int]]:
    taxon_names = [leaf.name for leaf in tree.get_leaves()]
    taxon_to_idx = {name: idx for idx, name in enumerate(taxon_names)}
    branch_to_taxa: List[List[int]] = []
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue
        descendants = [
            taxon_to_idx[leaf.name]
            for leaf in node.get_leaves()
            if leaf.name in taxon_to_idx
        ]
        branch_to_taxa.append(descendants)
    return branch_to_taxa


def compute_feature_index_stats(
    *,
    alignment_path: Path | str,
    tree_path: Path | str,
) -> Dict[str, Any]:
    """Compute parent/child feature index ranges used by the frozen model."""

    X, ntaxa, L = encode_alignment(str(alignment_path))
    tree = load_tree(str(tree_path))
    leaves = [leaf.name for leaf in tree.get_leaves()]
    if len(leaves) != ntaxa:
        raise InputPreflightError(
            "Tree/alignment taxon count mismatch before inference: "
            f"alignment_ntaxa={ntaxa}, tree_leaves={len(leaves)}."
        )
    branches = enumerate_branches(tree)
    branch_to_taxa = _build_branch_to_taxa_indices(tree)
    if len(branch_to_taxa) != len(branches):
        raise InputPreflightError(
            "Internal branch-index construction mismatch: "
            f"branches={len(branches)}, branch_to_taxa={len(branch_to_taxa)}."
        )

    max_descendants = 0
    max_desc_branch = ""
    for bid, taxa in zip(branches, branch_to_taxa):
        if len(taxa) > max_descendants:
            max_descendants = len(taxa)
            max_desc_branch = str(bid)

    max_parent = 0
    max_child = 0
    overflow_cells = 0
    first_overflow: Optional[str] = None
    for i in range(L):
        col = X[:, i].tolist()
        consensus = Counter(col).most_common(1)[0][0]
        for b_idx, taxa in enumerate(branch_to_taxa):
            if not taxa:
                continue
            mismatches = sum(X[t, i] != consensus for t in taxa)
            parent_val = len(taxa) - mismatches
            child_val = mismatches
            if parent_val > max_parent:
                max_parent = int(parent_val)
            if child_val > max_child:
                max_child = int(child_val)
            if parent_val > PAD_ID or child_val > PAD_ID:
                overflow_cells += 1
                if first_overflow is None:
                    first_overflow = (
                        f"branch={branches[b_idx]},site={i + 1},"
                        f"desc={len(taxa)},parent={parent_val},child={child_val}"
                    )

    return {
        "n_sequences": int(ntaxa),
        "alignment_length_codons": int(L),
        "n_branches": int(len(branches)),
        "max_descendants_per_branch": int(max_descendants),
        "max_descendants_branch": max_desc_branch,
        "max_parent_index": int(max_parent),
        "max_child_index": int(max_child),
        "overflow_cell_count_gt_pad": int(overflow_cells),
        "first_overflow_example": first_overflow or "",
        "pad_id": int(PAD_ID),
    }


def validate_alignment_tree_preflight(
    *,
    alignment_path: Path | str,
    tree_path: Path | str,
    gene_name: str,
    max_model_index: int = PAD_ID,
    allow_gap_triplet: bool = True,
) -> Dict[str, Any]:
    """Validate codons and feature indices before frozen-model inference."""

    codon_audit = audit_codon_fasta(
        alignment_path,
        gene_name=gene_name,
        stage="preflight_alignment",
        allow_gap_triplet=allow_gap_triplet,
    )
    if not bool(codon_audit.get("valid_bool", False)):
        raise InputPreflightError(
            "Codon preflight failed for alignment before inference. "
            f"invalid_total={codon_audit.get('invalid_total')}, "
            f"examples={codon_audit.get('invalid_examples')}"
        )

    stats = compute_feature_index_stats(alignment_path=alignment_path, tree_path=tree_path)
    max_parent = int(stats["max_parent_index"])
    max_child = int(stats["max_child_index"])
    if max_parent > int(max_model_index) or max_child > int(max_model_index):
        raise InputPreflightError(
            "Model-input preflight failed: parent/child feature index exceeds frozen model limit. "
            f"max_allowed={int(max_model_index)}, max_parent_index={max_parent}, "
            f"max_child_index={max_child}, max_descendants_per_branch={stats['max_descendants_per_branch']}, "
            f"n_sequences={stats['n_sequences']}, first_overflow={stats['first_overflow_example'] or 'n/a'}"
        )

    out = dict(codon_audit)
    out.update(stats)
    out["preflight_ok"] = True
    return out


def normalize_seqrecord(
    rec: SeqRecord,
    *,
    keep_description: bool = False,
) -> SeqRecord:
    seq = _normalize_seq(str(rec.seq))
    description = rec.description if keep_description else ""
    return SeqRecord(Seq(seq), id=rec.id, description=description)
