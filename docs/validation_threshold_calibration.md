# Full-Pipeline Validation and Empirical Significance Calibration

This workflow validates BABAPPAi in two explicitly separated layers:

1. **Diagnostic layer**: `EII_z`, `EII_01` (recoverability magnitude)
2. **Inferential layer**: `p_emp`, `q_emp`, `significant_bool` (empirical significance under matched neutral calibration)

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

- neutral p-value calibration diagnostics (`p_emp` histogram and QQ)
- q-based operating characteristics:
  - neutral significant rate (empirical FPR proxy)
  - regime-specific significant rates (low/medium/high)
  - q-based FPR, TPR, balanced accuracy
- EII threshold metrics are retained only as descriptive legacy comparison
- sigma calibration diagnostics:
  - `sigma0_raw` vs `sigma0_final`
  - floor usage fraction
  - fallback usage fraction

## Main script

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

## Reporting guardrail

Significance is interpreted as **excess branch-site dispersion relative to the matched neutral simulator**. It is not interpreted as direct proof of adaptive substitution.
