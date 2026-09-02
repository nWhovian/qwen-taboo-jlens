#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$(uname -s)" == "Darwin" && "${JUPYTER_MCP_MODE:-remote}" == "remote" ]]; then
  SSH_HOST="${RUNPOD_SSH_HOST:-runpod-jlens}"
  REMOTE_PROJECT="${RUNPOD_PROJECT_PATH:-/workspace/qwen-taboo-jlens}"
  if [[ "$SSH_HOST" == -* || "$SSH_HOST" == *[[:space:]]* ]]; then
    echo "RUNPOD_SSH_HOST must be one SSH host alias without whitespace." >&2
    exit 1
  fi
  printf -v REMOTE_COMMAND 'cd %q && exec bash scripts/start_jupyter_mcp.sh' "$REMOTE_PROJECT"
  exec ssh -T "$SSH_HOST" "$REMOTE_COMMAND"
fi

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
DEFAULT_VENV="$PROJECT_ROOT/.venv"
if [[ -x /opt/qwen-taboo-venv/bin/python ]]; then
  DEFAULT_VENV=/opt/qwen-taboo-venv
fi
VENV_PATH="${PROJECT_VENV:-$DEFAULT_VENV}"

if [[ ! -x "$VENV_PATH/bin/jupyter-mcp-server" ]]; then
  echo "Jupyter MCP is not installed in $VENV_PATH. Run: bash scripts/bootstrap_runpod.sh" >&2
  exit 1
fi

exec "$VENV_PATH/bin/jupyter-mcp-server"
