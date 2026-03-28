# BABAPPAi ChatGPT Upload Report (v1.1.0, post-push)

## 1) Repository snapshot
- Repository: `BABAPPAi`
- Branch: `main`
- Git status: clean and synced with remote (`main...origin/main`)
- Current head commit: `9249e6b37`
- Commit message: `Calibrate EII into empirical cEII and lock release methodology`

Recent history:
1. `9249e6b37` Calibrate EII into empirical cEII and lock release methodology
2. `2a5229421` chore: make repo release-ready for 1.1.0
3. `a12787fbe` Update pyproject.toml

## 2) Core scientific/method changes completed

### 2.1 Method lock
`D_obs` is now locked to a single explicit definition in software/manuscript:
- `D_obs = sample variance (ddof=1) of site_logit_mean across codon sites`

Ambiguous alternatives ("variance or equivalent summary") were removed from the active methods path.

### 2.2 EII reframed into calibrated identifiability
Three distinct outputs are now formalized:
1. **Raw EII magnitude**:
   - `eii_z_raw = (D_obs - mu0) / sigma0_final`
   - `eii_01_raw = sigmoid(eii_z_raw)`
2. **Calibrated identifiability probability**:
   - `ceii_gene = P(I_gene = 1 | data)`
   - `ceii_site = P(I_site = 1 | data)`
3. **Matched-neutral significance**:
   - `p_emp`, `q_emp`, `significant_bool`

`q_emp` significance is kept inferentially distinct from `cEII` probability.

### 2.3 Truth-aware recoverability targets
Calibration now uses held-out truth-aware targets:
- Continuous recoverability: `R_gene`, `R_site`
- Binary labels: `I_gene`, `I_site`
- Current pre-registered thresholds in code:
  - `tau_gene = 0.42`
  - `tau_site = 0.45`

### 2.4 Calibration model and assets
- Isotonic calibration pipeline implemented.
- Packaged calibration asset added:
  - `babappai/data/ceii_calibration_v1.json`
- Runtime now loads calibration asset and emits:
  - `ceii_gene`, `ceii_site`, `ceii_gene_class`, `ceii_site_class`, `ceii_ci`
  - `calibration_version`, `domain_shift_or_applicability`

## 3) Software outputs now available (runtime/API/CLI)
Gene-level output now includes:
- `eii_z_raw`
- `eii_01_raw`
- `ceii_gene`
- `ceii_site`
- `ceii_gene_class`
- `ceii_site_class`
- `ceii_gene_identifiable_bool`
- `ceii_site_identifiable_bool`
- `ceii_ci`
- `q_emp`
- `significant_bool`
- `calibration_version`
- `model_version`
- `model_checkpoint_provenance`
- `domain_shift_or_applicability`

CLI `version` now reports `calibration_version`.

## 4) New modules/scripts added
1. `babappai/calibration/ceii.py`
2. `babappai/calibration/recoverability.py`
3. `scripts/fit_ceii_calibration.py`
4. `scripts/run_ceii_calibration_benchmark.py`
5. `tests/test_ceii_calibration.py`
6. `babappai/data/ceii_calibration_v1.json`

## 5) Key modified files (major)
- `babappai/inference.py`
- `babappai/run_pipeline.py`
- `babappai/cli.py`
- `babappai/interpret.py`
- `babappai/validation/full_pipeline_validation.py`
- `manuscript/babappai.tex`
- `README.md`
- `CHANGELOG.md`
- `.github/workflows/ci.yml`
- `.github/workflows/publish.yml`

## 6) Current packaged calibration asset summary (`ceii_v1`)
From `babappai/data/ceii_calibration_v1.json`:
- `calibration_version`: `ceii_v1`
- Gene threshold: `0.625`
- Site threshold: `0.16666666666666666`
- Gene classes:
  - `0.0–0.525`: `not_identifiable`
  - `0.525–0.625`: `weak_or_ambiguous`
  - `0.625–0.725`: `identifiable`
  - `0.725–1.0`: `strongly_identifiable`
- Site classes:
  - `0.0–0.01`: `not_identifiable`
  - `0.01–0.1667`: `weak_or_ambiguous`
  - `0.1667–0.4167`: `identifiable`
  - `0.4167–1.0`: `strongly_identifiable`
- Applicability range currently encoded:
  - `n_taxa: 8–24`
  - `gene_length_nt: 762–2325`

## 7) Validation/build status (latest)
Executed and passing:
1. `pytest -q` -> pass (`35 tests`)
2. `python -m build --sdist --wheel` -> pass
3. `python -m twine check dist/*` -> pass
4. CLI run smoke (`babappai run ... --pvalue-mode frozen_reference --offline`) -> pass
5. CLI version output includes `calibration_version=ceii_v1`

Note on environment:
- Full online dependency resolution was unavailable in this sandbox.
- Wheel/sdist smoke installs were validated in offline-compatible mode (`--no-deps`; sdist with `--no-build-isolation`).

## 8) Manuscript status
`manuscript/babappai.tex` has been revised to:
- lock `D_obs` definition,
- distinguish raw EII, calibrated cEII, and q-based significance,
- include symbol/regime tables,
- reframe claims toward identifiability diagnostics (not adaptive proof),
- include quantitative calibration framing and OOD caveats.

## 9) Remaining risks / decisions for publication-grade confidence
1. **Calibration robustness**:
   - Site-level calibration remains weaker than gene-level in held-out/OOD settings.
2. **Domain generalization**:
   - cEII remains simulator-conditional; applicability range is currently narrow.
3. **Benchmark scale for final paper claims**:
   - Stronger manuscript claims require larger calibration/test/OOD campaigns and updated figures.
4. **Classical-tool comparison panel**:
   - If required for paper, PAML/HyPhy comparative analysis on matched simulations still needs full execution + cautious framing.

## 10) Suggested ChatGPT task prompt (paste with this report)
"Read this report and produce a publication-readiness gap analysis for BABAPPAi as a methods paper + software release. Focus on: (i) calibration validity of cEII, (ii) adequacy of split/OOD design, (iii) risk of overclaiming versus evidence, (iv) concrete additional experiments/figures/tables needed before submission, and (v) a prioritized execution plan with acceptance criteria."
