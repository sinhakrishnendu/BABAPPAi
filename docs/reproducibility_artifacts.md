# Reproducibility Artifacts Policy

This repository keeps source code, tests, manuscript sources, and small curated summary artifacts.
Large generated benchmark outputs are not tracked in git.

## What Is Kept In Git

- package source code (`babappai/`, `babappaomega/`)
- tests and workflow configuration
- manuscript source files (`manuscript/`)
- compact validation summaries under `artifacts/validation_summary/`

The curated summaries are intentionally small and are provided as quick inspection material:

- `artifacts/validation_summary/run_manifest.full_pipeline_v2_significance.json`
- `artifacts/validation_summary/significance_summary.full_pipeline_v2_significance.json`
- `artifacts/validation_summary/release_notes.full_pipeline_v2_significance.md`

## What Is Not Kept In Git

- generated run trees under `results/` (full inference outputs, replicate-level files, large TSVs, plots)
- local build outputs (`build/`, `dist/`)
- cache/temp outputs

`results/` is ignored by default via `.gitignore`.

## Where Large Reproducibility Assets Belong

Store full validation/benchmark outputs in an external archival location (for example Zenodo),
and link that archive from the manuscript and release notes.

Current manuscript-referenced benchmark archive:

- https://doi.org/10.5281/zenodo.18197957

## How To Regenerate Outputs

Use repository scripts and CLI workflows to regenerate outputs locally:

```bash
python scripts/run_full_pipeline_validation.py --outdir results/validation/full_pipeline_v2 --seed 123
babappai validate report --input results/validation/full_pipeline_v2 --outdir results/validation/full_pipeline_v2/report
```

For release branches, do not commit regenerated `results/` trees. Export or archive them externally and keep only compact summaries in `artifacts/`.
