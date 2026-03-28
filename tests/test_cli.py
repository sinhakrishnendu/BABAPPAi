import babappai.cli as cli


def _fake_payload():
    return {
        "gene_summary": {
            "EII_z": 1.84,
            "EII_01": 0.86,
            "eii_z_raw": 1.84,
            "eii_01_raw": 0.86,
            "ceii_gene": 0.82,
            "ceii_site": 0.61,
            "ceii_gene_class": "identifiable",
            "ceii_site_class": "weak_or_ambiguous",
            "ceii_gene_identifiable_bool": True,
            "ceii_site_identifiable_bool": False,
            "calibration_version": "ceii_v1",
            "domain_shift_or_applicability": "in_domain",
            "p_emp": 0.03,
            "q_emp": 0.04,
            "alpha_used": 0.05,
            "significant_bool": True,
        },
        "branch_results": [
            {"branch": "taxonA", "background_score": 0.9},
            {"branch": "taxonB", "background_score": 0.8},
        ],
        "site_results": [
            {"site": 3, "site_score": 0.92},
            {"site": 5, "site_score": 0.84},
        ],
        "warnings": [],
    }


def test_help_lists_validate_commands():
    help_text = cli.build_parser().format_help()
    assert "validate" in help_text
    assert "orthogroups" in help_text
    assert "synthetic" in help_text


def test_model_status_command(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "model_status",
        lambda: {
            "model_tag": "legacy_frozen",
            "legacy_model_name": "BABAPPAΩ frozen model",
            "cached": False,
            "cached_path": "/tmp/missing.pt",
            "doi": "10.5281/zenodo.18195869",
            "expected_sha256": "abc",
            "verified": False,
            "actual_sha256": None,
            "compatibility_note": "BABAPPAi is continuation",
        },
    )
    code = cli.main(["model", "status"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Compatibility:" in out


def test_example_write(tmp_path):
    outdir = tmp_path / "demo"
    code = cli.main(["example", "write", "--outdir", str(outdir)])
    assert code == 0
    assert (outdir / "aln.fasta").exists()
    assert (outdir / "tree.nwk").exists()
    assert (outdir / "synthetic_grid.json").exists()


def test_run_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "run_and_write_outputs", lambda **kwargs: _fake_payload())

    code = cli.main(
        [
            "run",
            "--alignment",
            str(tmp_path / "a.fasta"),
            "--tree",
            str(tmp_path / "t.nwk"),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Gene-level raw EII_z" in out
    assert "Significant at alpha=" in out


def test_run_broken_input(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "run_and_write_outputs",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("broken input")),
    )

    code = cli.main(
        [
            "run",
            "--alignment",
            str(tmp_path / "x.fasta"),
            "--tree",
            str(tmp_path / "x.nwk"),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1


def test_validate_orthogroups_select(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "select_orthogroups",
        lambda **kwargs: {"counts": {"selected": 100}},
    )
    code = cli.main(
        [
            "validate",
            "orthogroups",
            "select",
            "--input",
            str(tmp_path),
            "--outdir",
            str(tmp_path / "sel"),
        ]
    )
    assert code == 0


def test_validate_synthetic_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "run_synthetic_validation",
        lambda **kwargs: {"n_replicates": 8, "regime_counts": {"identifiable": 2}},
    )
    code = cli.main(
        [
            "validate",
            "synthetic",
            "run",
            "--simulator",
            str(tmp_path / "sim.py"),
            "--outdir",
            str(tmp_path / "syn"),
        ]
    )
    assert code == 0


def test_version(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "model_status",
        lambda: {
            "cached": True,
        },
    )
    code = cli.main(["version"])
    assert code == 0
    out = capsys.readouterr().out
    assert "software_name=" in out
