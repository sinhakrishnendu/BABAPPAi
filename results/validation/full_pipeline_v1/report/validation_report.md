# Full-Pipeline Validation Report

## Global calibration summary
- Decision target: `medium_high`
- Neutral 95th percentile (EII_01): `1.000`
- Neutral 99th percentile (EII_01): `1.000`
- FPR at 0.70: `0.358`
- TPR at 0.70: `0.604`
- Balanced accuracy at 0.70: `0.623`
- AUC: `0.721`

## Interpretation of default threshold
- 0.70 is more liberal than the empirical neutral 95th percentile.

## Comparison thresholds
- 0.90: `FPR=0.342, TPR=0.602`
- neutral95: `FPR=0.122, TPR=0.506`
- neutral99: `FPR=0.122, TPR=0.506`

## Bootstrap uncertainty (95% intervals)
- neutral_q95_EII_01: mean=1.000, CI=[1.000, 1.000]
- neutral_q99_EII_01: mean=1.000, CI=[1.000, 1.000]
- AUC: mean=0.721, CI=[0.712, 0.730]
- FPR_at_default: mean=0.358, CI=[0.344, 0.371]
- TPR_at_default: mean=0.604, CI=[0.590, 0.618]
- balanced_accuracy_at_default: mean=0.623, CI=[0.613, 0.633]

## Reproducibility summary (scenario-level)
- EII_01_ge_threshold: n=921, mean_threshold_consistency=0.962, mean_var_EII_01=0.0126
- EII_01_lt_threshold: n=1079, mean_threshold_consistency=0.924, mean_var_EII_01=0.0319

## Limitations
- Synthetic data use simplified codon-level simulation and are not intended as complete biological realism.
- Some fine-grained strata may have limited sample size in modest benchmark runs.
- Threshold recommendations should be revisited when larger empirical panels become available.
