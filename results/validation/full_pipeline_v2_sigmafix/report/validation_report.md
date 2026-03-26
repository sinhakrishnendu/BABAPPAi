# Full-Pipeline Validation Report

## Global calibration summary
- Decision target: `medium_high`
- Neutral 95th percentile (EII_01): `0.864`
- Neutral 99th percentile (EII_01): `0.921`
- FPR at 0.70: `0.184`
- TPR at 0.70: `0.596`
- Balanced accuracy at 0.70: `0.706`
- AUC: `0.766`

## Interpretation of default threshold
- 0.70 is more liberal than the empirical neutral 95th percentile.

## Comparison thresholds
- 0.90: `FPR=0.040, TPR=0.499`
- neutral95: `FPR=0.069, TPR=0.531`
- neutral99: `FPR=0.026, TPR=0.480`

## Bootstrap uncertainty (95% intervals)
- neutral_q95_EII_01: mean=0.859, CI=[0.831, 0.881]
- neutral_q99_EII_01: mean=0.928, CI=[0.906, 0.946]
- AUC: mean=0.765, CI=[0.746, 0.786]
- FPR_at_default: mean=0.183, CI=[0.158, 0.208]
- TPR_at_default: mean=0.596, CI=[0.567, 0.624]
- balanced_accuracy_at_default: mean=0.706, CI=[0.686, 0.725]

## Reproducibility summary (scenario-level)
- EII_01_ge_threshold: n=153, mean_threshold_consistency=0.948, mean_var_EII_01=0.0017
- EII_01_lt_threshold: n=247, mean_threshold_consistency=0.949, mean_var_EII_01=0.0027

## Limitations
- Synthetic data use simplified codon-level simulation and are not intended as complete biological realism.
- Some fine-grained strata may have limited sample size in modest benchmark runs.
- Threshold recommendations should be revisited when larger empirical panels become available.
