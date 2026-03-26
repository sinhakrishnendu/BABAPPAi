# Full-Pipeline Validation Report

## Global calibration summary
- Decision target: `medium_high`
- Neutral 95th percentile (EII_01): `0.130`
- Neutral 99th percentile (EII_01): `0.155`
- FPR at 0.70: `0.000`
- TPR at 0.70: `0.500`
- Balanced accuracy at 0.70: `0.750`
- AUC: `0.847`
## Significance framework summary
- alpha: `0.05`
- neutral significant rate (q_emp <= alpha): `0.000`
- medium/high significant rate (q_emp <= alpha): `0.000`
- q_emp FPR: `0.000`
- q_emp TPR: `0.000`
- q_emp balanced accuracy: `0.500`
- neutral p_emp KS deviation from Uniform(0,1): `0.786`
- significance calibration status: `needs_review`

## Legacy threshold comparison (descriptive only)
- q_emp significance calibration is needs_review at alpha=0.05; neutral significant rate=0.000, medium/high significant rate=0.000, neutral p_emp KS deviation=0.786.
- EII threshold balanced accuracy: `0.750`
- Delta balanced accuracy (q_emp - EII threshold): `-0.250`

## Comparison thresholds
- 0.90: `FPR=0.000, TPR=0.500`
- neutral95: `FPR=0.250, TPR=0.667`
- neutral99: `FPR=0.250, TPR=0.667`

## Bootstrap uncertainty (95% intervals)
- neutral_q95_EII_01: mean=0.099, CI=[0.032, 0.161]
- neutral_q99_EII_01: mean=0.107, CI=[0.032, 0.161]
- AUC: mean=0.848, CI=[0.673, 0.971]
- FPR_at_default: mean=0.000, CI=[0.000, 0.000]
- TPR_at_default: mean=0.505, CI=[0.187, 0.751]
- balanced_accuracy_at_default: mean=0.753, CI=[0.594, 0.876]
- FPR_q_alpha: mean=0.000, CI=[0.000, 0.000]
- TPR_q_alpha: mean=0.000, CI=[0.000, 0.000]
- balanced_accuracy_q_alpha: mean=0.500, CI=[0.500, 0.500]
- neutral_significant_rate_q: mean=0.000, CI=[0.000, 0.000]
- neutral_p_uniformity_ks: mean=0.769, CI=[0.452, 0.875]

## Reproducibility summary (scenario-level)
- EII_01_ge_threshold: n=2, mean_threshold_consistency=1.000, mean_significance_consistency=1.000, mean_var_EII_01=0.0001
- EII_01_lt_threshold: n=6, mean_threshold_consistency=1.000, mean_significance_consistency=1.000, mean_var_EII_01=0.0100
- q_significant_majority: n=0, mean_threshold_consistency=nan, mean_significance_consistency=nan, mean_var_EII_01=nan
- q_not_significant_majority: n=8, mean_threshold_consistency=1.000, mean_significance_consistency=1.000, mean_var_EII_01=0.0075

## Limitations
- Synthetic data use simplified codon-level simulation and are not intended as complete biological realism.
- Some fine-grained strata may have limited sample size in modest benchmark runs.
- Empirical p-values and q-values are conditional on the matched neutral simulator and calibration adequacy.
- Significance indicates excess dispersion relative to matched neutral expectation, not proof of adaptive substitution.
