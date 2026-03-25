# BABAPPAi Validation Report

## 1) Overview
- Empirical rows: 2
- Synthetic rows: 1

## 2) Provenance and naming compatibility notes
- BABAPPAi is the renamed continuation of the BABAPPAΩ codebase. The cached frozen model is a legacy BABAPPAΩ asset used for backward-compatible inference until BABAPPAi-specific weights are released.

## 3) Orthogroup selection criteria
- Deterministic rule-based selection with hard filters.
- Anti-cherry-picking stratified selection across divergence, length, and occupancy bins.
- Selection metadata path: /Users/krishnendu/Desktop/BABAPPAomega/babappaomegav1.3.0/tmp_validation_root/selection/selection_metadata.json

## 4) Empirical Anopheles validation results
- Regime counts: {'identifiable': 1, 'weak_or_ambiguous': 1}

## 5) Synthetic benchmark design
- Simulator-driven parameter grid with simulate-and-bucket tracking.

## 6) Role of neutral calibration generator
- External neutral generator can be run and logged via adapter metadata.

## 7) Calibration behavior
- Compare calibrated vs non-calibrated runs using empirical robustness metrics and synthetic summary.

## 8) Stability/perturbation analyses
- Repeatability, taxon subsampling sensitivity, and mild perturbation sensitivity are exported per orthogroup.

## 9) Limitations
- Legacy frozen model currently carries BABAPPAΩ provenance; BABAPPAi-specific weights are pending.

## 10) Recommended manuscript wording
- "BABAPPAi is the renamed continuation of the BABAPPAΩ codebase."
- "Scores are diagnostic identifiability measures, not proof of adaptive substitution."

## 11) Software citation and legacy asset citation notes
- Cite BABAPPAi software version and explicitly reference legacy model DOI where applicable.
