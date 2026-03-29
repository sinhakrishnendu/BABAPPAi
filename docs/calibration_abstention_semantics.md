# Calibration Applicability and Abstention Semantics (`ceii_v2`)

BABAPPAi now separates three layers explicitly:

1. Raw dispersion diagnostics (always available)
- `eii_z_raw`
- `eii_01_raw`

2. Matched-neutral significance (always available when neutral calibration runs)
- `p_emp`
- `q_emp`
- `significant_bool`

3. Calibrated identifiability probabilities (conditional)
- `ceii_gene`
- `ceii_site`
- `ceii_gene_class`
- `ceii_site_class`

Calibrated outputs are emitted only when both conditions hold:
- applicability support is valid, and
- null-layer sigma diagnostics are valid.

If either condition fails, BABAPPAi abstains:
- `ceii_gene = null`
- `ceii_site = null`
- `ceii_gene_class = calibration_unavailable`
- `ceii_site_class = calibration_unavailable`

## Applicability diagnostics
- `applicability_score`
- `applicability_status` in `{in_domain, near_boundary, out_of_domain, calibration_unavailable}`
- `within_applicability_envelope`
- `calibration_unavailable_reason`
- `nearest_supported_regime`
- `distance_to_supported_domain`

## Null-layer diagnostics
- `sigma0_valid`
- `sigma0_floored`
- `fallback_applied`

These diagnostics are included in runtime JSON and TSV outputs so downstream reports can avoid unsupported calibrated claims.
