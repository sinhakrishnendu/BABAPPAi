# Full-Pipeline Validation and Empirical Significance Calibration

This workflow validates BABAPPAi in two explicitly separated layers:

1. **Raw diagnostic layer**: `eii_z_raw`, `eii_01_raw` (dispersion magnitude)
2. **Empirical identifiability layer**: `ceii_gene`, `ceii_site` and cEII classes
3. **Inferential significance layer**: `p_emp`, `q_emp`, `significant_bool` (matched-neutral exceedance)

## Core significance definition

For each gene:

- observe `D_obs`
- obtain matched neutral replicates `D0^(1)...D0^(M)`
- compute:

`p_emp = (1 + count(D0 >= D_obs)) / (M + 1)`

Across tested genes:

- compute BH-adjusted `q_emp`
- default significance decision: `q_emp <= 0.05`

## What is evaluated

- cEII reliability and calibration quality (Brier, ECE, reliability bins)
- cEII operating characteristics on held-out test/OOD splits
- neutral p-value calibration diagnostics (`p_emp` histogram and QQ)
- q-based operating characteristics:
  - neutral significant rate (empirical FPR proxy)
  - regime-specific significant rates (low/medium/high)
  - q-based FPR, TPR, balanced accuracy
- raw EII threshold summaries are retained only as descriptive compatibility comparison
- sigma calibration diagnostics:
  - `sigma0_raw` vs `sigma0_final`
  - floor usage fraction
  - fallback usage fraction

## Main Scripts

```bash
python scripts/run_full_pipeline_validation.py \
  --outdir results/validation/full_pipeline_v2 \
  --n_per_regime 500 \
  --n_replicates_per_scenario 5 \
  --bootstrap_reps 1000 \
  --alpha 0.05 \
  --pvalue_mode empirical_monte_carlo \
  --neutral_reps 200 \
  --sigma_floor 0.05 \
  --seed 123
```

```bash
python scripts/run_ceii_calibration_benchmark.py \
  --outdir results/validation/ceii_benchmark_v1 \
  --n-per-regime 12 \
  --n-replicates-per-scenario 2 \
  --pvalue-mode frozen_reference \
  --bootstrap-reps 150 \
  --write-package-asset
```

## Reporting guardrail

`ceii_*` is interpreted as simulator-conditional recoverability probability.
`q_emp` is interpreted as **excess branch-site dispersion relative to the matched neutral simulator**.
Neither is interpreted as direct proof of adaptive substitution.
