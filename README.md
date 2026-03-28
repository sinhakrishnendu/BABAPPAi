# BABAPPAi (`babappai`)

[![CI](https://github.com/krishnendusinha/babappai/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnendusinha/babappai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BABAPPAi is the renamed continuation of the BABAPPAΩ codebase.
It is a diagnostic framework for branch-site recoverability/identifiability under matched neutral calibration.

## 1) Scope

BABAPPAi reports:

- effect-size style recoverability diagnostics: `EII_z`, `EII_01`
- empirical calibration significance: `p_emp`, `q_emp`, `significant_bool`

BABAPPAi does **not** perform classical dN/dS likelihood-ratio testing, and significance does **not** prove adaptive substitution.

## 2) Installation

```bash
pip install babappai
```

## 3) Quickstart

```bash
babappai model fetch
babappai example write --outdir demo
babappai run --alignment demo/aln.fasta --tree demo/tree.nwk --outdir demo_out
```

## 4) Statistical outputs

For each gene-level run:

- `D_obs`: observed dispersion statistic
- `mu0`: matched neutral mean
- `sigma0_raw`: raw neutral SD
- `sigma0_final`: SD after floor application
- `EII_z = (D_obs - mu0) / sigma0_final`
- `EII_01 = sigmoid(EII_z)` (diagnostic magnitude scale)
- `p_emp = (1 + count(D0 >= D_obs)) / (M + 1)` from matched neutral replicates
- `q_emp`: BH-adjusted `p_emp` across tested genes in an analysis set
- `significant_bool = (q_emp <= alpha)` (default `alpha=0.05`)
- `significance_label ∈ {not_significant, significant}`

EII bands are retained for descriptive reporting only and are deprecated as inferential decision rules.

## 5) Core CLI

```bash
babappai run --alignment aln.fasta --tree tree.nwk --outdir results
babappai run --alignment aln.fasta --tree tree.nwk --outdir results --alpha 0.05 --pvalue-mode empirical_monte_carlo --neutral-reps 200 --sigma-floor 0.05
babappai model fetch
babappai model status
babappai model verify
babappai doctor
babappai version
```

### Significance-related options

- `--alpha` (default `0.05`)
- `--pvalue-mode` (`empirical_monte_carlo` default; `frozen_reference` legacy fallback)
- `--neutral-reps` (Monte Carlo neutral replicates)
- `--min-neutral-group-size`
- `--sigma-floor`
- `--retain-eii-bands` / `--no-retain-eii-bands`
- `--report-threshold-bands` / `--no-report-threshold-bands`

## 6) Validation workflows

```bash
babappai validate orthogroups select --input ORTHOGROUP_DIR --outdir selection_out
babappai validate orthogroups run --input selection_out --outdir empirical_out --alpha 0.05 --pvalue-mode empirical_monte_carlo

babappai validate synthetic run \
  --simulator scripts/simulator.py \
  --outdir synthetic_out \
  --alpha 0.05 \
  --pvalue-mode empirical_monte_carlo

babappai validate report --input validation_root --outdir report_out
```

Full-pipeline manuscript validation helper:

```bash
python scripts/run_full_pipeline_validation.py \
  --outdir results/validation/full_pipeline_v2 \
  --n_per_regime 100 \
  --n_replicates_per_scenario 5 \
  --bootstrap_reps 1000 \
  --alpha 0.05 \
  --pvalue_mode empirical_monte_carlo \
  --neutral_reps 200 \
  --sigma_floor 0.05 \
  --seed 123
```

## 7) Output files

`babappai run` writes:

- `results.json`
- `branch_summary.tsv`
- `site_summary.tsv`
- `neutral_calibration_replicates.tsv`
- `interpretation.txt`
- `run_metadata.json`

Validation runs add calibration/significance tables, bootstrap summaries, figures, and manuscript-facing markdown blocks.

## 8) Reproducibility and provenance

- neutral replicate distributions used for `p_emp` are written to disk
- run metadata includes software version, model DOI/SHA, calibration settings, and command provenance
- BABAPPAi remains the renamed continuation of BABAPPAΩ; legacy model assets are explicitly labeled as such

## 9) Citation and interpretation guardrails

- cite BABAPPAi software version
- cite legacy model DOI while legacy frozen assets are used
- interpret significance as excess dispersion relative to matched neutral calibration, not as proof of adaptation

## 10) Development

```bash
pip install -e .[test]
pip install ruff build twine
ruff check .
pytest
python -m build --sdist --wheel
python -m twine check dist/*
```

## 11) Reproducibility and project policy

- large generated outputs policy: `docs/reproducibility_artifacts.md`
- citation metadata: `CITATION.cff`
- contribution guidelines: `CONTRIBUTING.md`
- code of conduct: `CODE_OF_CONDUCT.md`
- security reporting: `SECURITY.md`
- maintainer release process: `RELEASE_CHECKLIST.md`
