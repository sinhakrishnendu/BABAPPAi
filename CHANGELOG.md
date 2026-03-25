# Changelog

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
