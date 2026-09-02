# Qwen Taboo × J-Lens

A compact 16–20 hour mechanistic-interpretability project testing whether a
J-Lens fitted on base Qwen3.6-27B transfers through narrow Taboo LoRA
fine-tuning and whether it outperforms Logit Lens on target-specific readout.

## Start here

1. Read `docs/PROJECT_BRIEF.md` and `docs/EXPERIMENT_PLAN.md`.
2. Build the pinned GPU image once using `docs/CONTAINER.md` and use that image
   for new RunPod instances.
3. Follow `docs/RUNPOD_SETUP.md` to connect Codex desktop and local-browser
   Jupyter to the RunPod GPU through SSH.
4. On RunPod, run `python3 scripts/create_env_local.py` and
   `bash scripts/bootstrap_runpod.sh`; with the image this verifies the runtime
   instead of installing or compiling it.
5. Run `python scripts/verify_artifacts.py` before downloading model weights.
6. Start with `notebooks/00_environment_smoke_test.ipynb`.
7. Continue through the Gold/Blue notebook sequence in `notebooks/README.md`.

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
- a pinned CUDA/PyTorch/FlashAttention/J-Lens Docker image and Compose setup;
- research question, controls, metrics, branches, and stop conditions;
- artifact preflight without downloading the 27B weights;
- RunPod bootstrap, Jupyter, SSH-tunnel, and remote health-check scripts;
- a project-local Jupyter MCP configuration;
- an environment smoke-test notebook;
- a four-notebook Gold/Blue behavior, J-Lens/Logit-Lens sweep and analysis
  pipeline with resumable artifacts;
- the curated Neel context folder and research papers supplied for this project;
- research log and evidence-ledger templates.

## Not included in Git

- model, LoRA, or J-Lens weights;
- Hugging Face, GitHub, Jupyter, or OpenAI credentials;
- activation dumps, checkpoints, caches, or virtual environments;
- generated large outputs.

See `references/README.md` for reference handling and
`docs/ARTIFACT_CHECKLIST.md` for runtime artifacts. The frozen two-secret
protocol is in `docs/GOLD_BLUE_PROTOCOL.md`.
