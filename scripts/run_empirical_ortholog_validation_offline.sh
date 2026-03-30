#!/usr/bin/env bash
set -euo pipefail

# Robust offline runner for empirical ortholog validation.
# Handles per-species dedup, model-support taxon capping, codon audits,
# and strict preflight before calling `babappai run`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_babappai_online_empirical}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
BABAPPAI_BIN="${BABAPPAI_BIN:-$VENV_DIR/bin/babappai}"
BABAPPALIGN_BIN="${BABAPPALIGN_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/molevo/bin/babappalign}"
IQTREE_BIN="${IQTREE_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/molevo/bin/iqtree}"

ORTHOLOG_DIR="${ORTHOLOG_DIR:-$REPO_ROOT/orthologs}"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/empirical_ortholog_validation_offline_robust}"
GENES="${GENES:-}"
N_CALIBRATION="${N_CALIBRATION:-200}"
NEUTRAL_REPS="${NEUTRAL_REPS:-200}"
SEED="${SEED:-42}"
ACCESSION_PREFIX_LEN="${ACCESSION_PREFIX_LEN:-6}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$BABAPPAI_BIN" ]]; then
  echo "[ERROR] babappai CLI not found or not executable: $BABAPPAI_BIN" >&2
  exit 1
fi
if [[ ! -d "$ORTHOLOG_DIR" ]]; then
  echo "[ERROR] ortholog directory missing: $ORTHOLOG_DIR" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CMD=(
  "$PYTHON_BIN"
  "$REPO_ROOT/scripts/run_empirical_ortholog_validation.py"
  --ortholog-dir "$ORTHOLOG_DIR"
  --outdir "$OUTDIR"
  --babappai "$BABAPPAI_BIN"
  --babappalign "$BABAPPALIGN_BIN"
  --iqtree "$IQTREE_BIN"
  --seed "$SEED"
  --n-calibration "$N_CALIBRATION"
  --neutral-reps "$NEUTRAL_REPS"
  --device "$DEVICE"
  --accession-prefix-len "$ACCESSION_PREFIX_LEN"
  --overwrite
)

if [[ -n "$GENES" ]]; then
  CMD+=(--genes "$GENES")
fi

echo "[INFO] Running offline empirical validation"
echo "[INFO] OUTDIR=$OUTDIR"
if [[ -n "$GENES" ]]; then
  echo "[INFO] GENES=$GENES"
fi

"${CMD[@]}"

echo "[DONE] Offline robust run finished."
echo "[DONE] Summary: $OUTDIR/babappai_per_gene_summary.tsv"
echo "[DONE] Codon audit: $OUTDIR/codon_audit.tsv"
echo "[DONE] Orthology audit: $OUTDIR/orthology_audit.tsv"
