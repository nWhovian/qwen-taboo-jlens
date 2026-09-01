# Qwen Taboo × J-Lens

A compact 16–20 hour mechanistic-interpretability project testing whether a
J-Lens fitted on base Qwen3.6-27B transfers through narrow Taboo LoRA
fine-tuning and whether it outperforms Logit Lens on target-specific readout.

## Start here

1. Read `docs/PROJECT_BRIEF.md` and `docs/EXPERIMENT_PLAN.md`.
2. Follow `docs/RUNPOD_SETUP.md` on an H100 80 GB or A100 80 GB pod.
3. Add the two manual Neel-context files described in `references/README.md`.
4. Run `scripts/bootstrap_runpod.sh`.
5. Run `scripts/fetch_public_references.sh`.
6. Run `python scripts/verify_artifacts.py` before downloading model weights.
7. Start Jupyter with `scripts/start_jupyter.sh`.
8. Open this folder as the remote Codex project and use the prompt in
   `docs/CODEX_START_PROMPT.md`.

## What is already included

- persistent Codex instructions in `AGENTS.md`;
- a project-scoped Jupyter MCP configuration without secrets;
- the research question, controls, metrics, branches, and stop conditions;
- an artifact preflight that checks Hugging Face metadata without downloading
  27B weights;
- a RunPod bootstrap script and public-reference downloader;
- a lightweight environment notebook that does not load the model;
- a research log and evidence-ledger template.

## What is deliberately not included

- model weights, LoRA weights, J-Lens weights, or activation dumps;
- Hugging Face or Jupyter tokens;
- unpublished or access-controlled documents;
- invented Taboo prompts;
- an assumption that the public adapter and lens are compatible.

See `references/README.md` for the exact manual downloads and
`docs/ARTIFACT_CHECKLIST.md` for runtime artifacts.

