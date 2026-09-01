#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f .env.local ]]; then
  echo "Missing .env.local." >&2
  exit 1
fi

set -a
source .env.local
set +a

: "${JUPYTER_TOKEN:?JUPYTER_TOKEN is required}"
echo "http://127.0.0.1:${LOCAL_JUPYTER_PORT:-8888}/lab?token=${JUPYTER_TOKEN}"
