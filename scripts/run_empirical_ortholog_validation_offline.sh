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
CACHE_ROOT="${CACHE_ROOT:-$OUTDIR/cache}"
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/.cache/huggingface}"
CEII_ASSET="${CEII_ASSET:-}"
REQUIRE_CALIBRATION_VERSION_PREFIX="${REQUIRE_CALIBRATION_VERSION_PREFIX:-ceii_v3.1}"

resolve_bin() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  command -v "$candidate" 2>/dev/null || true
}

PYTHON_RESOLVED="$(resolve_bin "$PYTHON_BIN")"
BABAPPAI_RESOLVED="$(resolve_bin "$BABAPPAI_BIN")"
BABAPPALIGN_RESOLVED="$(resolve_bin "$BABAPPALIGN_BIN")"
IQTREE_RESOLVED="$(resolve_bin "$IQTREE_BIN")"

if [[ -z "$PYTHON_RESOLVED" || ! -x "$PYTHON_RESOLVED" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -z "$BABAPPAI_RESOLVED" || ! -x "$BABAPPAI_RESOLVED" ]]; then
  echo "[ERROR] babappai CLI not found or not executable: $BABAPPAI_BIN" >&2
  exit 1
fi
if [[ ! -d "$ORTHOLOG_DIR" ]]; then
  echo "[ERROR] ortholog directory missing: $ORTHOLOG_DIR" >&2
  exit 1
fi
if [[ -z "$CEII_ASSET" ]]; then
  echo "[ERROR] CEII_ASSET is required and must point to an explicit calibration JSON asset." >&2
  exit 1
fi

if [[ -z "$BABAPPALIGN_RESOLVED" || ! -x "$BABAPPALIGN_RESOLVED" ]]; then
  echo "[ERROR] babappalign not found or not executable: $BABAPPALIGN_BIN" >&2
  exit 1
fi
if [[ -z "$IQTREE_RESOLVED" || ! -x "$IQTREE_RESOLVED" ]]; then
  echo "[ERROR] iqtree not found or not executable: $IQTREE_BIN" >&2
  exit 1
fi

CEII_ASSET_RESOLVED="$("$PYTHON_RESOLVED" - <<'PY' "$CEII_ASSET"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
if [[ ! -f "$CEII_ASSET_RESOLVED" ]]; then
  echo "[ERROR] Requested cEII asset does not exist: $CEII_ASSET_RESOLVED" >&2
  exit 1
fi
CEII_CALIBRATION_VERSION="$("$PYTHON_RESOLVED" - <<'PY' "$CEII_ASSET_RESOLVED"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
print(str(payload.get("calibration_version", "unknown")))
PY
)"
if [[ -n "$REQUIRE_CALIBRATION_VERSION_PREFIX" && "$CEII_CALIBRATION_VERSION" != ${REQUIRE_CALIBRATION_VERSION_PREFIX}* ]]; then
  echo "[ERROR] Calibration version mismatch: expected prefix '$REQUIRE_CALIBRATION_VERSION_PREFIX', got '$CEII_CALIBRATION_VERSION'" >&2
  exit 1
fi

mkdir -p "$CACHE_ROOT/babappalign/embeddings" "$HF_HOME_DIR/hub"

export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub"
export TRANSFORMERS_CACHE="$HF_HOME_DIR/hub"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CMD=(
  "$PYTHON_RESOLVED"
  "$REPO_ROOT/scripts/run_empirical_ortholog_validation.py"
  --ortholog-dir "$ORTHOLOG_DIR"
  --outdir "$OUTDIR"
  --babappai "$BABAPPAI_RESOLVED"
  --babappalign "$BABAPPALIGN_RESOLVED"
  --iqtree "$IQTREE_RESOLVED"
  --seed "$SEED"
  --n-calibration "$N_CALIBRATION"
  --neutral-reps "$NEUTRAL_REPS"
  --device "$DEVICE"
  --ceii-asset "$CEII_ASSET_RESOLVED"
  --accession-prefix-len "$ACCESSION_PREFIX_LEN"
  --overwrite
)

if [[ -n "$GENES" ]]; then
  CMD+=(--genes "$GENES")
fi

echo "[INFO] Running offline empirical validation"
echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] CEII_ASSET=$CEII_ASSET_RESOLVED"
echo "[INFO] CEII_CALIBRATION_VERSION=$CEII_CALIBRATION_VERSION"
if [[ -n "$GENES" ]]; then
  echo "[INFO] GENES=$GENES"
fi

"${CMD[@]}"

echo "[DONE] Offline robust run finished."
echo "[DONE] Summary: $OUTDIR/babappai_per_gene_summary.tsv"
echo "[DONE] Codon audit: $OUTDIR/codon_audit.tsv"
echo "[DONE] Orthology audit: $OUTDIR/orthology_audit.tsv"
