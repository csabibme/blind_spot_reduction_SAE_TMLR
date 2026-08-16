#!/usr/bin/env bash
# Create an isolated environment for submission_artifacts experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON" >&2
  exit 1
fi

echo "Using Python: $($PYTHON --version) at $(command -v "$PYTHON")"

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"

echo ""
echo "submission environment ready."
echo "  source $ROOT/.venv/bin/activate"
echo "  export SAE_REPO_ROOT=\"$(cd ../.. && pwd)\""
echo "  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
