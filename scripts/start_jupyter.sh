#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f .env.local ]]; then
  echo "Missing .env.local. Run: python3 scripts/create_env_local.py" >&2
  exit 1
fi

set -a
source .env.local
set +a

: "${JUPYTER_TOKEN:?JUPYTER_TOKEN is required}"
JUPYTER_HOST="${JUPYTER_HOST:-127.0.0.1}"
JUPYTER_PORT="${JUPYTER_PORT:-8889}"
DEFAULT_VENV="$PROJECT_ROOT/.venv"
if [[ -x /opt/qwen-taboo-venv/bin/python ]]; then
  DEFAULT_VENV=/opt/qwen-taboo-venv
fi
VENV_PATH="${PROJECT_VENV:-$DEFAULT_VENV}"
JUPYTER_HOST="${JUPYTER_BIND_HOST:-$JUPYTER_HOST}"

if [[ ! -x "$VENV_PATH/bin/jupyter" ]]; then
  echo "Jupyter is not installed in $VENV_PATH. Run: bash scripts/bootstrap_runpod.sh" >&2
  exit 1
fi

exec "$VENV_PATH/bin/jupyter" lab \
  --allow-root \
  --no-browser \
  --ip="$JUPYTER_HOST" \
  --port="$JUPYTER_PORT" \
  --ServerApp.root_dir="$PROJECT_ROOT" \
  --IdentityProvider.token="$JUPYTER_TOKEN"
