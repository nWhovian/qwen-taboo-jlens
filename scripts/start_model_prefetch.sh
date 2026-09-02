#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VENV="$PROJECT_ROOT/.venv"
if [[ -x /opt/qwen-taboo-venv/bin/python ]]; then
  DEFAULT_VENV=/opt/qwen-taboo-venv
fi
VENV_PATH="${PROJECT_VENV:-$DEFAULT_VENV}"

CONFIG_PATH="${PREFETCH_CONFIG_PATH:-$PROJECT_ROOT/configs/gold_blue_experiment.json}"
LOG_PATH="${PREFETCH_LOG_PATH:-/workspace/model-prefetch.log}"
STATUS_PATH="${PREFETCH_STATUS_PATH:-/workspace/model-prefetch-status.json}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "Python environment is missing at $VENV_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Prefetch config is missing: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$STATUS_PATH")"
export PREFETCH_STATUS_PATH="$STATUS_PATH"

nohup "$VENV_PATH/bin/python" "$PROJECT_ROOT/scripts/prefetch_models.py" \
  --config "$CONFIG_PATH" \
  --max-parallel "${PREFETCH_MAX_PARALLEL:-2}" \
  >>"$LOG_PATH" 2>&1 &

echo "Model prefetch requested in background (PID $!)."
echo "Log: $LOG_PATH"
echo "Status: $STATUS_PATH"
