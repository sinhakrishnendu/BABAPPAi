# Changelog

## Unreleased

## 2.0.1 - 2026-03-29

- Set `pyproject.toml` `[project].version` to `2.0.1` as the canonical package version source.
- Updated runtime metadata to resolve software version from installed package metadata, falling back to `pyproject.toml` in source mode.
- Added release workflow tag guard to fail when tag and package versions diverge.
- Added release workflow diagnostics for source version, `GITHUB_REF_NAME`, and built `dist/` contents.
- Hardened publish build step with explicit clean build directories before sdist/wheel generation.
- Upgraded publish workflow action majors for Node 24 runtime compatibility while preserving trusted PyPI publishing.

- Locked `D_obs` definition to a single statistic:
  sample variance (`ddof=1`) of `site_logit_mean` across codon sites.
- Added empirical cEII calibration framework:
  - truth-aware recoverability targets (`R_gene`, `R_site`) and binary labels (`I_gene`, `I_site`)
  - isotonic calibration to `ceii_gene` / `ceii_site`
  - calibration-versioned decision bands derived from held-out data
- Added packaged calibration asset `babappai/data/ceii_calibration_v1.json`.
- Extended runtime/API outputs with calibrated identifiability fields:
  - `eii_z_raw`, `eii_01_raw`
  - `ceii_gene`, `ceii_site`
  - `ceii_gene_class`, `ceii_site_class`
  - `ceii_ci`
  - `calibration_version`, `domain_shift_or_applicability`
- Added end-to-end calibration orchestration script:
  `scripts/run_ceii_calibration_benchmark.py`
  plus fitting utility `scripts/fit_ceii_calibration.py`.
- Updated manuscript and docs to separate:
  raw EII magnitude, calibrated cEII probability, and matched-neutral q-based significance.
- Build metadata now uses a single source of truth for versioning at
  `pyproject.toml` `[project].version`, with runtime reporting synchronized
  through `babappai.metadata.resolve_software_version()`.
- Added version integrity tests to enforce agreement between runtime version,
  build metadata, and CLI-reported version.
- Added CI and publish workflows with gated `test -> build -> publish` flow,
  artifact validation, and wheel/sdist smoke-install checks.
- Added governance and publication metadata files:
  `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  and `RELEASE_CHECKLIST.md`.
- Added reproducibility artifact policy and stopped tracking generated outputs
  under `results/` for future commits.

## 1.1.0 - 2026-03-26

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

## 1.0.0 - 2026-03-25

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
