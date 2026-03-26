# Reproducibility / Recoverability Validation

This analysis evaluates whether higher recoverability diagnostics and empirical significance correspond to more stable replicate behavior.

## Scenario-level quantities

For each latent scenario across observed replicates:

- mean / variance of `EII_01`
- mean `q_emp`
- probability above descriptive EII threshold
- probability significant (`q_emp <= alpha`)
- pairwise replicate consistency of EII-threshold status
- pairwise replicate consistency of q-significance status

## Stratified summaries

Reported strata include both:

- EII-magnitude groups (`mean EII_01 >= threshold` vs `< threshold`)
- significance groups (`majority significant` vs `majority not significant`)

This allows direct comparison of descriptive EII magnitude with inferential q-based support.

## Command

```bash
python scripts/analyze_replicate_recoverability.py \
  --metrics_tsv results/validation/full_pipeline_v2/inference/full_pipeline_gene_metrics.tsv \
  --outdir results/validation/full_pipeline_v2/reproducibility \
  --default_threshold 0.70 \
  --alpha 0.05
```
