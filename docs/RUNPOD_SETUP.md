# RunPod and Jupyter MCP setup

## 1. Pod

Recommended starting point:

- one H100 80 GB or A100 80 GB;
- current PyTorch/CUDA image;
- approximately 200 GB persistent volume mounted at `/workspace`;
- SSH enabled.

Keep `/workspace/hf-cache`, this repository, raw results, and checkpoints on the
persistent volume.

## 2. Put the project on RunPod

Copy or clone this complete directory to:

```text
/workspace/qwen-taboo-jlens
```

Use Git for code and small results. Do not commit weights, activations, secrets,
or the downloaded Neel context pack.

## 3. Create private environment values

Copy `.env.example` to `.env.local`, replace the Jupyter token with a random
long value, and keep the file uncommitted.

The remote Codex process must also see `JUPYTER_TOKEN` in its login environment.
The reliable setup is to store the export in a private file outside Git, source
that file from the remote login profile, then open a fresh Codex remote task.
Verify in Codex's remote terminal that `JUPYTER_TOKEN` is present without
printing its value.

Authenticate to Hugging Face interactively. Never put `HF_TOKEN` in this repo.

## 4. Bootstrap

From the project root:

```bash
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
bash scripts/fetch_public_references.sh
python scripts/verify_artifacts.py
```

The bootstrap creates a fresh Python 3.11 environment while attempting to
retain the CUDA-enabled Torch from the RunPod image. It clones the official
J-Lens implementation and the secondary Taboo source repository but does not
install the latter's separate pinned runtime.

Stop if `torch.cuda.is_available()` is false.

## 5. Start Jupyter

Run in `tmux` so the kernel survives a disconnected laptop:

```bash
tmux new -s jlens
source .venv/bin/activate
bash scripts/start_jupyter.sh
```

Detach with `Ctrl-b`, then `d`. Jupyter binds to `127.0.0.1:8888`, so it is not
publicly exposed.

To view it from the Mac browser, create an SSH tunnel:

```bash
ssh -L 8888:127.0.0.1:8888 runpod-jlens
```

Then open `http://127.0.0.1:8888` locally.

## 6. Open in Codex

Add the RunPod host to the Codex desktop application's remote connections and
open `/workspace/qwen-taboo-jlens` as the remote project. Trust the project so
Codex can read `.codex/config.toml`.

The MCP process is launched on the remote host and connects to the remote
Jupyter server at `127.0.0.1:8888`. Open a new remote Codex task after changing
the MCP configuration or login environment.

Use `docs/CODEX_START_PROMPT.md` as the first message.

## 7. Time-boxed fallback

If Jupyter MCP does not successfully list and execute notebook cells within
30–45 focused minutes, use an IPython session inside `tmux` and run long jobs as
scripts with persistent logs. Do not spend the research budget debugging MCP.

