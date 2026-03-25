# BABAPPAi (`babappai`)

BABAPPAi is the renamed continuation of the BABAPPAΩ codebase.
It is a diagnostic software/manuscript ecosystem for identifiability of episodic branch-site structure.

## 1) What BABAPPAi is

BABAPPAi is a likelihood-free diagnostic framework that estimates whether episodic branch-site structure is statistically identifiable from finite coding-sequence alignments.

## 2) What it is not

- Not a classical dN/dS estimator.
- Not a likelihood-ratio branch-site test.
- Not definitive proof of adaptive substitution.

## 3) Installation

```bash
pip install babappai
```

Optional CLI-centric install:

```bash
pipx install babappai
```

## 4) Quickstart

```bash
babappai model fetch
babappai example write --outdir demo
babappai run --alignment demo/aln.fasta --tree demo/tree.nwk --outdir demo_out
```

## 5) CLI reference

```bash
babappai --help
```

Core commands:

```bash
babappai run --alignment aln.fasta --tree tree.nwk --outdir results
babappai run --alignment aln.fasta --tree tree.nwk --outdir results --tree-calibration
babappai model fetch
babappai model status
babappai model verify
babappai doctor
babappai example write --outdir demo
babappai validate orthogroups select --input ORTHOGROUP_DIR --outdir selection_out
babappai validate orthogroups run --input selection_out --outdir empirical_out
babappai validate synthetic run --simulator scripts/simulator.py --neutral-generator scripts/generate_neutral_calibration.py --outdir synthetic_out
babappai validate report --input validation_root --outdir report_out
babappai version
```

## 6) Output files explained

`babappai run ...` emits:

- `results.json`
- `branch_summary.tsv`
- `site_summary.tsv`
- `interpretation.txt`
- `run_metadata.json`

Validation workflows emit additional summaries/reports (selection, empirical, synthetic, master report outputs).

## 7) Interpretation of `EII_z` and `EII_01`

- `EII_z`: calibrated raw identifiability score.
- `EII_01`: bounded companion score in `[0,1]`.

Deterministic transform:

```text
EII_01 = sigmoid(EII_z)
```

## 8) Definitive identifiability regimes

- `0.00 <= EII_01 < 0.30` -> `not_identifiable`
- `0.30 <= EII_01 < 0.70` -> `weak_or_ambiguous`
- `0.70 <= EII_01 < 0.90` -> `identifiable`
- `0.90 <= EII_01 <= 1.00` -> `strongly_identifiable`

Also emitted everywhere:

- `identifiable_bool = (EII_01 >= 0.70)`
- `identifiability_extent`

## 9) Legacy model download/cache/checksum/provenance notes

The currently configured frozen model is a **legacy BABAPPAΩ model asset** used for backward-compatible inference in BABAPPAi.

- model file: `babappaomega.pt`
- legacy model DOI: `10.5281/zenodo.18195869`
- URL: `https://zenodo.org/records/18195869/files/babappaomega.pt?download=1`
- SHA-256: `657a662563af31304abcb208fc903d2770a9184632a9bab2095db4c538fed8eb`

Cache uses `platformdirs.user_cache_dir("babappai")` (or `BABAPPAI_CACHE_DIR`).
Checksum verification is always enforced.

## 10) Validation and benchmarking

Validation includes:

- empirical orthogroup-based validation
- simulator-driven synthetic benchmarking
- unified report generation with figures/tables

## 11) Orthogroup selection workflow

Deterministic top-100 selection with hard filters and anti-cherry-picking stratification:

```bash
babappai validate orthogroups select --input ORTHOGROUP_DIR --outdir selection_out
```

Outputs:

- `selected_100_orthogroups.tsv`
- `rejected_orthogroups.tsv`
- `orthogroup_qc_metrics.tsv`
- `orthogroup_selection_report.txt`
- `selection_metadata.json`

## 12) Synthetic benchmarking using the supplied simulator

```bash
babappai validate synthetic run \
  --simulator scripts/simulator.py \
  --outdir synthetic_out \
  --grid-config demo/synthetic_grid.json
```

Optional neutral generator integration:

```bash
babappai validate synthetic run \
  --simulator scripts/simulator.py \
  --neutral-generator scripts/generate_neutral_calibration.py \
  --outdir synthetic_out
```

## 13) Neutral calibration generator integration

BABAPPAi includes an adapter for external neutral calibration generators:

- module: `babappai/calibration/neutral_generator_adapter.py`
- CLI integration: `--neutral-generator PATH`
- metadata logging includes script path, attempted command, seed, and output reference file.

## 14) Reproducibility/version metadata

Run/validation outputs include software version, command string, model DOI/SHA/cache path, and calibration metadata.

## 15) Citation

Cite BABAPPAi software release/version and, while legacy frozen model assets are still used, cite the legacy model DOI.

Legacy records currently referenced for provenance:

- legacy software record: `https://zenodo.org/records/18520163`
- legacy frozen model record: `https://zenodo.org/records/18195869`

## 16) Legacy BABAPPAΩ compatibility note

BABAPPAi is the renamed continuation of BABAPPAΩ.
Some legacy artifacts and DOI records still carry the BABAPPAΩ naming. These are retained for provenance and reproducibility and are explicitly marked as legacy assets.

## Development quick commands

```bash
pip install -e .[test]
pytest
python -m build --sdist --wheel
python -m twine check dist/*
```
