# Changelog

## 2.1.0 - 2026-03-26

- Replaced threshold-centric inferential decision layer with an empirical significance framework:
  - empirical Monte Carlo `p_emp` from matched neutral exceedance
  - BH-adjusted `q_emp` across analysis sets
  - default significance decision `significant_bool = (q_emp <= 0.05)`
- Kept `EII_z` / `EII_01` as recoverability diagnostics; retained EII bands as descriptive-only legacy reporting.
- Extended gene-level outputs with calibration transparency fields:
  - `D_obs`, `mu0`, `sigma0_raw`, `sigma0_final`, `sigma_floor_used`
  - `fallback_flag`, `fallback_reason`, `neutral_group_size`, `calibration_group`
  - `p_emp`, `q_emp`, `alpha_used`, `significant_bool`, `significance_label`
- Added storage/export of matched neutral replicate distributions for reproducible `p_emp`.
- Added CLI/config options for significance calibration controls:
  - `--alpha`
  - `--pvalue-mode`
  - `--neutral-reps`
  - `--min-neutral-group-size`
  - `--sigma-floor`
  - `--retain-eii-bands` / `--report-threshold-bands`
- Hardened sigma calibration path with explicit raw vs final sigma summaries and fallback tracking.
- Updated full-pipeline validation to report significance-calibration diagnostics:
  - neutral/medium-high significant rates at `q <= alpha`
  - q-based FPR/TPR and EII-threshold comparison
  - neutral `p_emp` histogram and QQ diagnostics
  - bootstrap summaries including q-based operating characteristics
- Updated manuscript text, README, and validation docs to align with the new significance semantics.

## 2.0.0 - 2026-03-25

- Renamed the software ecosystem from **BABAPPAΩ** to **BABAPPAi**.
- Canonical install/package/CLI are now `babappai`.
- Added explicit provenance note: BABAPPAi is the renamed continuation of the BABAPPAΩ codebase.
- Added `babappaomega` legacy compatibility namespace and deprecated CLI alias.
- Added canonical metadata constants for software/model provenance and legacy compatibility notes.
- Added robust model manager with SHA-256 verification, offline behavior, cache status, and legacy model messaging.
- Kept frozen model external to package and pinned to legacy model DOI/checksum metadata.
- Implemented mandatory identifiability outputs across outputs and summaries:
  - `EII_z`
  - `EII_01 = sigmoid(EII_z)`
  - `identifiable_bool`
  - `identifiability_extent`
- Added first-class neutral calibration generator adapter integration.
- Added first-class simulator adapter integration for synthetic benchmarking.
- Added validation architecture modules for:
  - orthogroup QC and deterministic top-100 stratified selection
  - empirical orthogroup validation execution
  - synthetic benchmarking with simulate-and-bucket tracking
  - unified validation report generation and figure/table export
- Expanded CLI with validation command tree:
  - `validate orthogroups select`
  - `validate orthogroups run`
  - `validate synthetic run`
  - `validate report`
- Rewrote README for BABAPPAi naming, workflows, interpretation policy, validation, and legacy asset citations.
- Updated manuscript source naming/availability language for BABAPPAi with BABAPPAΩ legacy provenance notes.
