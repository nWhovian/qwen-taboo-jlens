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
export JUPYTER_URL="http://127.0.0.1:${JUPYTER_PORT:-8889}"
export ALLOW_IMG_OUTPUT="${ALLOW_IMG_OUTPUT:-true}"

if [[ ! -x .venv/bin/jupyter-mcp-server ]]; then
  echo "Jupyter MCP is not installed in .venv. Run: bash scripts/bootstrap_runpod.sh" >&2
  exit 1
fi

exec .venv/bin/jupyter-mcp-server
