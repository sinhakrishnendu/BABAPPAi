import csv
from pathlib import Path

from babappai.validation.orthogroup_selection import select_orthogroups


def _write_alignment(path: Path, seqs: list[str]) -> None:
    lines = []
    for i, seq in enumerate(seqs, start=1):
        lines.append(f">sp{i}|gene{i}\n{seq}\n")
    path.write_text("".join(lines))


def _read_selected_ids(path: Path) -> list[str]:
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [row["orthogroup_id"] for row in reader]


def test_deterministic_top_selection(tmp_path):
    input_dir = tmp_path / "ogs"
    input_dir.mkdir()

    _write_alignment(input_dir / "OG0001.fasta", ["ATG" * 120, "ATG" * 120, "ATG" * 120, "ATG" * 120])
    _write_alignment(input_dir / "OG0002.fasta", ["ATG" * 120, "ATA" * 120, "ATC" * 120, "ATT" * 120])
    _write_alignment(input_dir / "OG0003.fasta", ["ATG" * 140, "ATG" * 140, "ATG" * 140, "ATG" * 140])
    _write_alignment(input_dir / "OG0004.fasta", ["ATG" * 100, "ATA" * 100, "ATC" * 100, "ATT" * 100])
    _write_alignment(input_dir / "OG0005.fasta", ["ATG" * 130, "ATG" * 130, "ATG" * 130, "ATG" * 130])

    out1 = tmp_path / "sel1"
    out2 = tmp_path / "sel2"

    meta1 = select_orthogroups(
        input_dir=str(input_dir),
        outdir=str(out1),
        target_n=3,
        min_taxa=4,
        occupancy_threshold=0.8,
        min_length_nt=240,
        max_missingness=0.5,
    )
    meta2 = select_orthogroups(
        input_dir=str(input_dir),
        outdir=str(out2),
        target_n=3,
        min_taxa=4,
        occupancy_threshold=0.8,
        min_length_nt=240,
        max_missingness=0.5,
    )

    assert meta1["counts"]["selected"] == 3
    assert meta2["counts"]["selected"] == 3

    ids1 = _read_selected_ids(out1 / "selected_100_orthogroups.tsv")
    ids2 = _read_selected_ids(out2 / "selected_100_orthogroups.tsv")
    assert ids1 == ids2

    assert (out1 / "rejected_orthogroups.tsv").exists()
    assert (out1 / "orthogroup_qc_metrics.tsv").exists()
    assert (out1 / "orthogroup_selection_report.txt").exists()
    assert (out1 / "selection_metadata.json").exists()
