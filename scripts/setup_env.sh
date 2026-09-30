#!/usr/bin/env bash
set -euo pipefail

FMLAB_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$FMLAB_PROJECT_ROOT/scripts/env.sh"

FMLAB_VENV="${FMLAB_VENV:-$FMLAB_DATA_ROOT/envs/fmlab}"
FMLAB_PYTHON="${FMLAB_PYTHON:-python3}"

"$FMLAB_PYTHON" -m venv --system-site-packages "$FMLAB_VENV"
"$FMLAB_VENV/bin/python" -m pip install --upgrade pip

if [[ -n "${FMLAB_TORCHVISION_SPEC:-}" ]]; then
  FMLAB_TORCHVISION_INDEX_URL="${FMLAB_TORCHVISION_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
  "$FMLAB_VENV/bin/python" -m pip install \
    --index-url "$FMLAB_TORCHVISION_INDEX_URL" \
    --no-deps \
    "$FMLAB_TORCHVISION_SPEC"
fi

"$FMLAB_VENV/bin/python" -m pip install -e "$FMLAB_PROJECT_ROOT[all]"

echo "Environment ready: $FMLAB_VENV"
echo "Activate with: source $FMLAB_VENV/bin/activate"
