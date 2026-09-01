#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

for command_name in git python3 nvidia-smi tmux codex; do
  if command -v "$command_name" >/dev/null 2>&1; then
    echo "OK: $command_name -> $(command -v "$command_name")"
  else
    echo "MISSING: $command_name"
  fi
done

if [[ -x .venv/bin/python ]]; then
  echo "OK: project virtual environment"
else
  echo "MISSING: .venv (run scripts/bootstrap_runpod.sh)"
fi

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
  if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
    echo "OK: .env.local contains JUPYTER_TOKEN (value hidden)"
  else
    echo "MISSING: JUPYTER_TOKEN in .env.local"
  fi
else
  echo "MISSING: .env.local (run scripts/create_env_local.py)"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
codex --version 2>/dev/null || true
codex login status 2>/dev/null || true
git status --short --branch
