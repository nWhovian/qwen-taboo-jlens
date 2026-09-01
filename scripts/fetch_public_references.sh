#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p references/papers

curl -fL --retry 3 \
  https://arxiv.org/pdf/2505.14352 \
  -o references/papers/2505.14352-eliciting-latent-knowledge.pdf

curl -fL --retry 3 \
  https://arxiv.org/pdf/2603.05494 \
  -o references/papers/2603.05494-censored-llms.pdf

curl -fL --retry 3 \
  https://arxiv.org/pdf/2605.26045 \
  -o references/papers/2605.26045-probabilistic-activation-oracles.pdf

curl -fL --retry 3 \
  https://transformer-circuits.pub/2026/workspace/index.html \
  -o references/papers/global-workspace.html

echo "Public references downloaded."
echo "Now follow references/README.md for the two manual Neel-context files."

