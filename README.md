# Qwen Taboo × J-Lens

A compact 16–20 hour mechanistic-interpretability project testing whether a
J-Lens fitted on base Qwen3.6-27B transfers through narrow Taboo LoRA
fine-tuning and whether it outperforms Logit Lens on target-specific readout.

## Start here

1. Read `docs/PROJECT_BRIEF.md` and `docs/EXPERIMENT_PLAN.md`.
2. Follow `docs/RUNPOD_SETUP.md` to connect Codex desktop and local-browser
   Jupyter to a RunPod GPU through SSH.
3. On RunPod, run `python3 scripts/create_env_local.py` and
   `bash scripts/bootstrap_runpod.sh`.
4. Run `python scripts/verify_artifacts.py` before downloading model weights.
5. Start with `notebooks/00_environment_smoke_test.ipynb`.

## Development setup

The repository is cloned under `/workspace` on RunPod. Codex desktop opens that
checkout as a remote SSH project. Jupyter and its persistent kernel also run on
RunPod; the Mac browser reaches them through a private SSH tunnel. The checked-in
project MCP configuration lets remote Codex inspect and execute short notebook
cells in the same Jupyter server.

CLI-only work is supported but is not required. Long experiments should run as
scripts in `tmux`, not depend only on an interactive MCP call.

## Included

- persistent project instructions in `AGENTS.md`;
- research question, controls, metrics, branches, and stop conditions;
- artifact preflight without downloading the 27B weights;
- RunPod bootstrap, Jupyter, SSH-tunnel, and remote health-check scripts;
- a project-local Jupyter MCP configuration;
- an environment smoke-test notebook;
- the curated Neel context folder and research papers supplied for this project;
- research log and evidence-ledger templates.

## Not included in Git

- model, LoRA, or J-Lens weights;
- Hugging Face, GitHub, Jupyter, or OpenAI credentials;
- activation dumps, checkpoints, caches, or virtual environments;
- generated large outputs.

See `references/README.md` for reference handling and
`docs/ARTIFACT_CHECKLIST.md` for runtime artifacts.
