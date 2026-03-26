# Full-Pipeline Validation Report

## Global calibration summary
- Decision target: `medium_high`
- Neutral 95th percentile (EII_01): `0.500`
- Neutral 99th percentile (EII_01): `0.500`
- FPR at 0.70: `0.250`
- TPR at 0.70: `1.000`
- Balanced accuracy at 0.70: `0.875`
- AUC: `0.882`

## Interpretation of default threshold
- 0.70 is more conservative than the empirical neutral 95th percentile.

## Comparison thresholds
- 0.90: `FPR=0.250, TPR=0.917`
- neutral95: `FPR=0.500, TPR=1.000`
- neutral99: `FPR=0.500, TPR=1.000`

## Bootstrap uncertainty (95% intervals)
- neutral_q95_EII_01: mean=0.485, CI=[0.373, 0.500]
- neutral_q99_EII_01: mean=0.491, CI=[0.468, 0.500]
- AUC: mean=0.893, CI=[0.762, 1.000]
- FPR_at_default: mean=0.242, CI=[0.000, 0.455]
- TPR_at_default: mean=1.000, CI=[1.000, 1.000]
- balanced_accuracy_at_default: mean=0.879, CI=[0.773, 1.000]

## Reproducibility summary (scenario-level)
- EII_01_ge_threshold: n=5, mean_threshold_consistency=1.000, mean_var_EII_01=0.0005
- EII_01_lt_threshold: n=3, mean_threshold_consistency=1.000, mean_var_EII_01=0.0006

## Limitations
- Synthetic data use simplified codon-level simulation and are not intended as complete biological realism.
- Some fine-grained strata may have limited sample size in modest benchmark runs.
- Threshold recommendations should be revisited when larger empirical panels become available.
