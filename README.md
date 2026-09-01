# Qwen Taboo × J-Lens

A compact 16–20 hour mechanistic-interpretability project testing whether a
J-Lens fitted on base Qwen3.6-27B transfers through narrow Taboo LoRA
fine-tuning and whether it outperforms Logit Lens on target-specific readout.

## Start here

1. Read `docs/PROJECT_BRIEF.md` and `docs/EXPERIMENT_PLAN.md`.
2. Follow `docs/RUNPOD_SETUP.md` on an H100 80 GB or A100 80 GB pod.
3. Add Neel's reference folder as described in `references/README.md`.
4. Run `scripts/bootstrap_runpod.sh`.
5. Run `python scripts/verify_artifacts.py` before downloading model weights.
6. Use Codex CLI or IPython in `tmux` on RunPod.

## What is already included

- persistent project instructions in `AGENTS.md`;
- the research question, controls, metrics, branches, and stop conditions;
- an artifact preflight that checks Hugging Face metadata without downloading
  27B weights;
- a minimal RunPod bootstrap script and optional public-reference downloader;
- a research log and evidence-ledger template.

## What is deliberately not included

- model weights, LoRA weights, J-Lens weights, or activation dumps;
- Hugging Face tokens;
- unpublished or access-controlled documents;
- invented Taboo prompts;
- an assumption that the public adapter and lens are compatible.

See `references/README.md` for the exact manual downloads and
`docs/ARTIFACT_CHECKLIST.md` for runtime artifacts.
