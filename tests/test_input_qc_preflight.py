from __future__ import annotations

from pathlib import Path

import pytest
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from babappai.encoding import encode_alignment
from babappai.inference import run_inference
from babappai.input_qc import (
    InputPreflightError,
    audit_codon_fasta,
    deduplicate_species_records,
)


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as fh:
        for rid, seq in records:
            fh.write(f">{rid}\n{seq}\n")


def test_oov_codon_raises_clean_validation_error(tmp_path: Path) -> None:
    aln = tmp_path / "bad_oov.fasta"
    _write_fasta(
        aln,
        [
            ("tax1", "ATGAAATTT"),
            ("tax2", "ATGNNNTTT"),
        ],
    )
    with pytest.raises(ValueError, match="unsupported codon triplets"):
        encode_alignment(str(aln))


def test_ambiguous_codons_are_flagged_in_alignment_audit(tmp_path: Path) -> None:
    aln = tmp_path / "bad_ambiguous.fasta"
    _write_fasta(
        aln,
        [
            ("tax1", "ATGAAATTT"),
            ("tax2", "ATGRYTTTT"),
        ],
    )
    audit = audit_codon_fasta(aln, gene_name="toy", stage="codon_alignment", allow_gap_triplet=True)
    assert audit["valid_bool"] is False
    assert int(audit["ambiguous_codon_count"]) > 0
    assert "ambiguous" in str(audit["invalid_examples"])


def test_duplicate_species_isoforms_are_deduplicated_deterministically() -> None:
    records = [
        SeqRecord(Seq("ATGAAATTT"), id="XP_123456001.1", description="XP_123456001.1"),
        SeqRecord(Seq("ATGAAATTTCCC"), id="XP_123456999.1", description="XP_123456999.1"),
        SeqRecord(Seq("ATGAAATTT"), id="XP_654321001.1", description="XP_654321001.1"),
    ]
    kept, audit_rows = deduplicate_species_records(records, gene_name="toy", accession_prefix_len=6)
    kept_ids = [rec.id for rec in kept]
    assert len(kept) == 2
    assert "XP_123456999.1" in kept_ids
    assert len(audit_rows) == 1
    assert audit_rows[0]["discarded_record_id"] == "XP_123456001.1"
    assert audit_rows[0]["kept_record_id"] == "XP_123456999.1"


def test_ago2_r2d2_like_large_descendant_inputs_fail_preflight_before_model(tmp_path: Path) -> None:
    n_taxa = 63
    aln = tmp_path / "large.fasta"
    records = [(f"tax{i:02d}", "ATG" * 12) for i in range(1, n_taxa + 1)]
    _write_fasta(aln, records)

    leaves_rest = ",".join(f"tax{i:02d}:0.1" for i in range(2, n_taxa + 1))
    tree = tmp_path / "large.nwk"
    tree.write_text(f"(tax01:0.1,({leaves_rest}):0.1);\n")

    class BoomModel:
        def eval(self):
            return self

        def __call__(self, *_args, **_kwargs):  # pragma: no cover - should never execute
            raise AssertionError("model call should be blocked by preflight")

    with pytest.raises(InputPreflightError, match="preflight failed before TorchScript"):
        run_inference(
            alignment_path=str(aln),
            tree_path=str(tree),
            device="cpu",
            offline=True,
            pvalue_mode="disabled",
            ceii_enabled=False,
            _model_override=BoomModel(),
            _resolved_device_override=torch.device("cpu"),
        )
