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

if [[ ! -x .venv/bin/jupyter ]]; then
  echo "Jupyter is not installed in .venv. Run: bash scripts/bootstrap_runpod.sh" >&2
  exit 1
fi

exec .venv/bin/jupyter lab \
  --allow-root \
  --no-browser \
  --ip="$JUPYTER_HOST" \
  --port="$JUPYTER_PORT" \
  --ServerApp.root_dir="$PROJECT_ROOT" \
  --IdentityProvider.token="$JUPYTER_TOKEN"
