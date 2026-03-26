### Validation calibration methods
Full-pipeline synthetic alignments were simulated under explicit latent burden regimes (neutral/low/medium/high) and nuisance strata.
For each replicate, BABAPPAi inference was run without replacing core inference code.
The observed dispersion statistic D_obs was extracted from model outputs; neutral calibration parameters (mu0, sigma0) were derived from matched neutral replicate distributions, with sigma0 floored at a user-configurable minimum before EII_z was computed.
Empirical p-values were computed as p_emp=(1+count(D0>=D_obs))/(M+1), where D0 are matched neutral replicate dispersion values.
Multiple testing control used Benjamini-Hochberg q_emp across genes in the analysis set, with significance defined as q_emp<=alpha.
Threshold-based EII_01 summaries were retained only as descriptive/legacy comparisons.
Bootstrap uncertainty intervals were computed by gene-level resampling with replacement.
EII was interpreted as an identifiability/recoverability diagnostic, not direct evidence of adaptive substitution.
