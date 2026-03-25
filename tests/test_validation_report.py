from pathlib import Path

from babappai.validation.validation_reporting import generate_validation_report


def test_validation_report_generation(tmp_path):
    val_root = tmp_path / "validation"
    emp = val_root / "empirical"
    syn = val_root / "synthetic"
    emp.mkdir(parents=True)
    syn.mkdir(parents=True)

    (emp / "empirical_summary.tsv").write_text(
        "orthogroup_id\tEII_z\tEII_01\tidentifiable_bool\tidentifiability_extent\n"
        "OG0001\t1.2\t0.77\tTrue\tidentifiable\n"
    )
    (syn / "synthetic_replicates.tsv").write_text(
        "replicate_id\tEII_z\tEII_01\tidentifiable_bool\tidentifiability_extent\n"
        "rep0001\t-0.2\t0.45\tFalse\tweak_or_ambiguous\n"
    )

    report_dir = tmp_path / "report"
    meta = generate_validation_report(input_dir=str(val_root), outdir=str(report_dir))

    assert meta["n_master_rows"] == 2
    assert (report_dir / "validation_master_summary.json").exists()
    assert (report_dir / "validation_master_summary.tsv").exists()
    assert (report_dir / "validation_report.md").exists()
    assert (report_dir / "publication_ready_figures" / "empirical_regime_counts.svg").exists()
    assert (report_dir / "supplementary_tables").exists()
