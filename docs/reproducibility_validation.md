# Reproducibility / Recoverability Validation

This analysis evaluates whether higher recoverability diagnostics and empirical significance correspond to more stable replicate behavior.

## Scenario-level quantities

For each latent scenario across observed replicates:

- mean / variance of `ceii_gene`
- mean / variance of `ceii_site`
- mean `q_emp`
- probability above calibrated `ceii_gene` decision threshold
- probability significant (`q_emp <= alpha`)
- pairwise replicate consistency of cEII class/status
- pairwise replicate consistency of q-significance status

## Stratified summaries

Reported strata include both:

- cEII groups (`mean ceii_gene >= threshold` vs `< threshold`)
- significance groups (`majority significant` vs `majority not significant`)

This allows direct comparison of calibrated identifiability probability with inferential q-based support.

## Command

```bash
python scripts/analyze_replicate_recoverability.py \
  --metrics_tsv results/validation/ceii_benchmark_v1/inference/full_pipeline_gene_metrics.tsv \
  --outdir results/validation/ceii_benchmark_v1/reproducibility \
  --default_threshold 0.625 \
  --alpha 0.05
```
