# Offline Empirical Runner (Robust)

Use the wrapper below to run empirical ortholog validation fully offline with robust input handling:

`scripts/run_empirical_ortholog_validation_offline.sh`

## What it enforces
- deterministic per-species deduplication
- deterministic model-support taxon cap (prevents embedding index overflow)
- codon audits (`raw -> sanitized -> aligned -> preflight`)
- fail-fast preflight validation before `babappai run`
- per-gene error messages in summary instead of TorchScript crashes

## Quick run (all genes in `orthologs/`)
```bash
scripts/run_empirical_ortholog_validation_offline.sh
```

## Run selected genes
```bash
GENES=\"ago2,r2d2,newgene1,newgene2,newgene3,newgene4,newgene5\" \\\n+OUTDIR=\"results/empirical_ortholog_validation_offline_batch\" \\\n+scripts/run_empirical_ortholog_validation_offline.sh
```

## Optional overrides
- `VENV_DIR` (default: `.venv_babappai_online_empirical`)
- `ORTHOLOG_DIR` (default: `orthologs/`)
- `OUTDIR` (default: `results/empirical_ortholog_validation_offline_robust`)
- `N_CALIBRATION` (default: `200`)
- `NEUTRAL_REPS` (default: `200`)
- `ACCESSION_PREFIX_LEN` (default: `6`)
- `DEVICE` (default: `cpu`)
- `BABAPPALIGN_BIN`, `IQTREE_BIN`, `BABAPPAI_BIN`, `PYTHON_BIN`

## Key outputs
- per-gene summary: `OUTDIR/babappai_per_gene_summary.tsv`
- codon audit: `OUTDIR/codon_audit.tsv`
- orthology audit: `OUTDIR/orthology_audit.tsv`
- full run log: `OUTDIR/run_log.txt`
