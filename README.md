# BABAPPAi (`babappai`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19216760.svg)](https://doi.org/10.5281/zenodo.19216760)

[![CI](https://github.com/krishnendusinha/babappai/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnendusinha/babappai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BABAPPAi is the software framework built around the **canonical frozen BABAPPAΩ model**.

- **BABAPPAΩ**: fixed neural inference artifact (weights are immutable in this project).
- **BABAPPAi**: operational package around that model (raw EII, matched-neutral significance, applicability/abstention, optional cEII calibration, reporting, and packaging).

## 1) Scope

BABAPPAi reports:

- raw dispersion diagnostics: `eii_z_raw`, `eii_01_raw`
- empirically calibrated identifiability probabilities (conditional): `ceii_gene`, `ceii_site`
- empirical matched-neutral significance: `p_emp`, `q_emp`, `significant_bool`

Important interpretation contract:
- raw EII and matched-neutral significance are universally reportable outputs
- cEII is an auxiliary post-inference calibration layer and may be withheld (`null`) under inapplicable/unstable regimes
- calibration updates do **not** imply model-weight changes

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
: sample variance (`ddof=1`) of site-level `site_logit_mean` across codon sites
- `mu0`: matched neutral mean
- `sigma0_raw`: raw neutral SD
- `sigma0_final`: SD after floor application
- `eii_z_raw = (D_obs - mu0) / sigma0_final`
- `eii_01_raw = sigmoid(eii_z_raw)` (diagnostic magnitude scale)
- `ceii_gene`: calibrated `P(I_gene=1 | data)`
- `ceii_site`: calibrated `P(I_site=1 | data)`
- `ceii_gene_class`, `ceii_site_class`: calibration-derived decision bands
- `ceii_ci`: bootstrap calibration interval (if calibration asset provides it)
- `calibration_version`, `domain_shift_or_applicability`
- `p_emp = (1 + count(D0 >= D_obs)) / (M + 1)` from matched neutral replicates
- `q_emp`: BH-adjusted `p_emp` across tested genes in an analysis set
- `significant_bool = (q_emp <= alpha)` (default `alpha=0.05`)
- `significance_label ∈ {not_significant, significant}`

`ceii_*` and `q_emp` are intentionally distinct layers:
- `ceii_*`: recoverability/identifiability probability calibration
- `q_emp`: excess-dispersion significance under matched-neutral calibration

`ceii_*` is conditional and abstention-aware:
- if applicability/null checks fail, `ceii_gene` and `ceii_site` are withheld (`null`)
- this does not invalidate raw EII or matched-neutral significance

## 5) Core CLI

```bash
babappai run --alignment aln.fasta --tree tree.nwk --outdir results
babappai run --alignment aln.fasta --tree tree.nwk --outdir results --alpha 0.05 --pvalue-mode empirical_monte_carlo --neutral-reps 200 --sigma-floor 0.05
babappai model fetch
babappai model status
babappai model verify
babappai doctor
babappai version
babappai version --ceii-asset babappai/data/ceii_calibration_v2.json
```

### Significance-related options

- `--alpha` (default `0.05`)
- `--pvalue-mode` (`empirical_monte_carlo` default; `frozen_reference` static reference-table mode)
- `--neutral-reps` (Monte Carlo neutral replicates)
- `--min-neutral-group-size`
- `--sigma-floor`
- `--retain-eii-bands` / `--no-retain-eii-bands`
- `--report-threshold-bands` / `--no-report-threshold-bands`
- `--ceii-enabled` / `--no-ceii-enabled`
- `--ceii-asset`

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

cEII benchmark + calibration helper:

```bash
python scripts/run_ceii_calibration_benchmark.py \
  --outdir results/validation/ceii_benchmark_v1 \
  --n-per-regime 12 \
  --n-replicates-per-scenario 2 \
  --pvalue-mode frozen_reference \
  --bootstrap-reps 150 \
  --write-package-asset
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
- model provenance identifies BABAPPAΩ as the canonical frozen inference backbone used by BABAPPAi

## 9) Citation and interpretation guardrails

- cite BABAPPAi software version
- cite the BABAPPAΩ model DOI as the canonical frozen model artifact
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
