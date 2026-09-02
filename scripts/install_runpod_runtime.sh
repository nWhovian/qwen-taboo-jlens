#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VENV="$PROJECT_ROOT/.venv"
if [[ -x /opt/qwen-taboo-venv/bin/python ]]; then
  DEFAULT_VENV=/opt/qwen-taboo-venv
fi
VENV_PATH="${PROJECT_VENV:-$DEFAULT_VENV}"
PYTHON_BIN="${RUNPOD_PYTHON:-python3.12}"

TORCH_VERSION="2.10.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=910d8db9def162de5b7c15474b933e7e2371e93733b980e9d3c07cd3bf2f568e"
JLENS_COMMIT="581d398613e5602a5af361e1c34d3a92ea82ba8e"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "The RunPod runtime requires Linux x86_64. Use Docker on other hosts." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python 3.12: $PYTHON_BIN" >&2
  echo "Use the repository Docker image instead of repairing the host manually." >&2
  exit 1
fi

if [[ -x "$VENV_PATH/bin/python" ]]; then
  if "$VENV_PATH/bin/python" "$PROJECT_ROOT/scripts/check_runtime.py"; then
    echo "Compatible RunPod runtime already exists at $VENV_PATH."
    exit 0
  fi
  EXISTING_PYTHON="$($VENV_PATH/bin/python -c 'import platform; print(platform.python_version())')"
  if [[ "$EXISTING_PYTHON" != 3.12.* ]]; then
    echo "Existing environment at $VENV_PATH uses Python $EXISTING_PYTHON." >&2
    echo "Move it aside or set PROJECT_VENV to a new path, then rerun." >&2
    exit 1
  fi
  echo "Repairing the incomplete Python 3.12 environment at $VENV_PATH."
else
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

PYTHON="$VENV_PATH/bin/python"

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install \
  --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION"
"$PYTHON" -m pip install \
  -r "$PROJECT_ROOT/requirements/runpod-cu130.txt"
"$PYTHON" -m pip install "$FLASH_ATTN_WHEEL"
"$PYTHON" -m pip install --no-deps \
  "jlens @ git+https://github.com/anthropics/jacobian-lens.git@$JLENS_COMMIT"

"$PYTHON" -m pip check
if [[ "${SKIP_RUNTIME_CHECK:-0}" != "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_runtime.py"
fi
echo "RunPod runtime installed at $VENV_PATH."
