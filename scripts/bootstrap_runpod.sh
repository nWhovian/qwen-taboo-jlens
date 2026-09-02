#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_VENV="$PROJECT_ROOT/.venv"
if [[ -x /opt/qwen-taboo-venv/bin/python ]]; then
  DEFAULT_VENV=/opt/qwen-taboo-venv
fi
VENV_PATH="${PROJECT_VENV:-$DEFAULT_VENV}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  bash scripts/install_runpod_runtime.sh
fi

source "$VENV_PATH/bin/activate"

python -m ipykernel install --user --name qwen-taboo-jlens \
  --display-name "Qwen Taboo J-Lens"

mkdir -p vendor results figures logs artifacts/activations \
  artifacts/lens_outputs artifacts/checkpoints data/raw_outputs

python scripts/check_runtime.py --require-gpu

python -m src.environment_report >/dev/null

echo "Bootstrap complete. Review results/environment_report.json."
echo "Next: python scripts/verify_artifacts.py"
echo "Then: tmux new -s jlens-jupyter 'bash scripts/start_jupyter.sh'"
