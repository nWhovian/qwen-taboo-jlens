#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d .venv ]]; then
  uv venv --python 3.11 --system-site-packages .venv
fi

source .venv/bin/activate

uv pip install --upgrade \
  "transformers>=4.57.1" accelerate peft safetensors huggingface_hub \
  datasets pandas pyarrow scikit-learn matplotlib seaborn \
  jupyterlab jupyter-collaboration jupyter-mcp-tools ipykernel

python -m ipykernel install --user --name qwen-taboo-jlens \
  --display-name "Qwen Taboo J-Lens"

mkdir -p vendor results figures logs artifacts/activations \
  artifacts/lens_outputs artifacts/checkpoints data/raw_outputs

if [[ ! -d vendor/jacobian-lens/.git ]]; then
  git clone https://github.com/anthropics/jacobian-lens.git vendor/jacobian-lens
fi
uv pip install -e vendor/jacobian-lens

if [[ ! -d vendor/probabilistic_activation_oracles/.git ]]; then
  git clone --recurse-submodules \
    https://github.com/federicotorrielli/probabilistic_activation_oracles.git \
    vendor/probabilistic_activation_oracles
fi

python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Stop before downloading model weights.")
PY

python -m src.environment_report >/dev/null

echo "Bootstrap complete. Review results/environment_report.json."
echo "Next: bash scripts/fetch_public_references.sh"
echo "Then: python scripts/verify_artifacts.py"

