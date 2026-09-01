#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

: "${JUPYTER_TOKEN:?Set JUPYTER_TOKEN in .env.local or the shell environment}"

exec jupyter lab \
  --ip 127.0.0.1 \
  --port 8888 \
  --no-browser \
  --IdentityProvider.token "$JUPYTER_TOKEN"

