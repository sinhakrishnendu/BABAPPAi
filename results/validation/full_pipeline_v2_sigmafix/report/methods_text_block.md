### Validation calibration methods
Full-pipeline synthetic alignments were simulated under explicit latent burden regimes (neutral/low/medium/high) and nuisance strata.
For each replicate, BABAPPAi inference was run without replacing core inference code.
The observed dispersion statistic D_obs was extracted from model outputs; neutral calibration parameters (mu0, sigma0) were taken from the same run metadata, with sigma0 floored at a user-configurable minimum before EII_z was computed.
Threshold calibration used global and stratum-specific neutral quantiles, ROC summaries, and operating-point rules (max Youden J and max TPR under FPR caps).
Bootstrap uncertainty intervals were computed by gene-level resampling with replacement.
EII was interpreted as an identifiability/recoverability diagnostic, not direct evidence of adaptive substitution.
