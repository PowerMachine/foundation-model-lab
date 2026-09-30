#!/usr/bin/env bash
set -euo pipefail

FMLAB_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$FMLAB_PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FMLAB_PROJECT_ROOT/.env"
  set +a
fi

FMLAB_DEFAULT_DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/foundation-model-lab"
export FMLAB_DATA_ROOT="${FMLAB_DATA_ROOT:-$FMLAB_DEFAULT_DATA_ROOT}"
export FMLAB_MODEL_ROOT="${FMLAB_MODEL_ROOT:-$HOME/models}"
export HF_HOME="${HF_HOME:-$FMLAB_DATA_ROOT/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$FMLAB_DATA_ROOT/cache/torch}"
export TMPDIR="${TMPDIR:-$FMLAB_DATA_ROOT/tmp}"
export PYTHONPATH="${PYTHONPATH:-}:$FMLAB_PROJECT_ROOT/src"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"

