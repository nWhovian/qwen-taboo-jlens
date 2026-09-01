# RunPod setup

## 1. Create the pod

- one H100 80 GB or A100 80 GB;
- a current PyTorch/CUDA image;
- about 200 GB of persistent storage mounted at `/workspace`;
- SSH enabled.

Keep the repository, Hugging Face cache, results, and checkpoints on the
persistent volume.

## 2. Clone the private GitHub repository

```bash
cd /workspace
git clone git@github.com:nWhovian/qwen-taboo-jlens.git
cd qwen-taboo-jlens
```

Authenticate GitHub over SSH first if the pod does not already have access to
the private repository.

## 3. Bootstrap and preflight

```bash
cp .env.example .env.local
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
huggingface-cli login
python scripts/verify_artifacts.py
```

Stop if CUDA is unavailable or the model, adapter, or lens metadata does not
match. Do not download the 27B weights before the metadata preflight passes.

## 4. Work persistently

Use Codex CLI or IPython inside `tmux`:

```bash
tmux new -s jlens
source .venv/bin/activate
codex
```

Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t jlens`.
Run long experiments as scripts with logs and resumable run IDs.

The dependable workflow is: edit locally in the Codex app, push to GitHub,
then pull and execute on RunPod. Direct RunPod-host support in the desktop app
is optional and is not required by this repository.
