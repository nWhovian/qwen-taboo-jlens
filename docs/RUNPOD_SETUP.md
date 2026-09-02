# RunPod, Jupyter, Codex, and MCP setup

This is the intended setup:

```text
Codex desktop on Mac ── SSH ──> Codex + repository on RunPod
Mac browser          ── SSH tunnel ──> Jupyter on RunPod (127.0.0.1:8889)
remote Codex         ── stdio MCP ──> Jupyter API and the persistent kernel
```

The model, CUDA environment, notebook kernel, and experiment files all live on
RunPod. The browser is only a local view through an encrypted SSH tunnel. Codex
desktop can open the RunPod checkout as a remote project, so normal development
does not require working only in a terminal.

The recommended deployment uses the pinned image in `docs/CONTAINER.md`. It
already contains the compatible CUDA/PyTorch/FlashAttention/J-Lens environment,
SSH, Jupyter and Codex. Do not use a generic PyTorch template and then upgrade
unversioned packages on every Pod.

## 1. Create the RunPod pod

Build and push the versioned image from `docs/CONTAINER.md`, then create the Pod
from that image with:

- one H100 80 GB or A100 80 GB GPU;
- at least 200 GB of persistent storage mounted at `/workspace`;
- a public IP and a TCP port mapped to container port `22` for full SSH;
- the public half of a dedicated SSH key added to the RunPod account.

Keep the repository, Hugging Face cache, results, and checkpoints under
`/workspace`. Do not expose Jupyter's port publicly: it binds to
`127.0.0.1:8889` and is reached through SSH forwarding.

RunPod shows an SSH command similar to:

```bash
ssh root@<POD_IP> -p <EXTERNAL_SSH_PORT> -i ~/.ssh/id_ed25519_runpod
```

Use the public-IP/full-SSH connection, not the basic proxy SSH endpoint, because
the browser tunnel needs SSH port forwarding.

## 2. Configure the Mac SSH alias

Create a dedicated key once if it does not already exist:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod -C ana.thorson@gmail.com
```

Add the contents of `~/.ssh/id_ed25519_runpod.pub` to **RunPod Settings → SSH
Public Keys**. Never upload or share the private file without `.pub`.

After the pod exists, add this concrete entry to `~/.ssh/config`:

```sshconfig
Host runpod-jlens
    HostName <POD_IP>
    User root
    Port <EXTERNAL_SSH_PORT>
    IdentityFile ~/.ssh/id_ed25519_runpod
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Replace both placeholders and test from the Mac:

```bash
ssh runpod-jlens
```

If this does not open a RunPod shell, stop here and fix SSH first.

## 3. Clone the private repository on RunPod

In the RunPod shell:

```bash
gh auth login --hostname github.com --git-protocol https
gh auth setup-git
gh repo clone nWhovian/qwen-taboo-jlens /workspace/qwen-taboo-jlens
cd /workspace/qwen-taboo-jlens
```

`gh auth login` prints a one-time device code and URL. Open that URL on the Mac,
sign in as `nWhovian`, and enter the code. Do not give GitHub passwords or tokens
to Codex.

If the repository was cloned previously, use:

```bash
cd /workspace/qwen-taboo-jlens
git pull --ff-only
```

## 4. Verify the image environment and create private configuration

On RunPod, from the repository root:

```bash
python3 scripts/create_env_local.py
bash scripts/bootstrap_runpod.sh
source /opt/qwen-taboo-venv/bin/activate
hf auth login
python scripts/verify_artifacts.py
bash scripts/check_remote_setup.sh
```

The first command generates `.env.local` with a random Jupyter token and mode
`0600`. It is ignored by Git. `bootstrap_runpod.sh` verifies the image runtime,
registers the `Qwen Taboo J-Lens` kernel, creates runtime directories, and
checks the GPU. It does not reinstall packages or compile FlashAttention.

The manual non-container fallback is documented in `docs/CONTAINER.md`; it uses
the same pinned versions and prebuilt FlashAttention wheel.

Stop before downloading 27B model weights if CUDA is unavailable or
`verify_artifacts.py` reports incompatible or unverified artifacts.

## 5. Start persistent Jupyter on RunPod

Start it in a detached `tmux` session:

```bash
cd /workspace/qwen-taboo-jlens
tmux new-session -d -s jlens-jupyter \
  "cd /workspace/qwen-taboo-jlens && bash scripts/start_jupyter.sh"
tmux list-sessions
```

Inspect the live server when needed:

```bash
tmux attach-session -t jlens-jupyter
```

Detach without stopping it using `Ctrl-b`, then `d`. The Jupyter process and
kernel survive SSH disconnects, but not a pod stop or destruction.

## 6. Open Jupyter in the Mac browser

On the Mac, keep this command running in a local terminal:

```bash
cd /Users/ana/mats/qwen-taboo-jlens
bash scripts/jupyter_tunnel.sh runpod-jlens
```

In a second local terminal, obtain the URL without copying the token manually:

```bash
ssh runpod-jlens \
  'cd /workspace/qwen-taboo-jlens && bash scripts/show_jupyter_url.sh'
```

Open the returned `http://127.0.0.1:8888/lab?token=...` URL in the Mac browser.
The page is local-looking, but every cell executes on RunPod's GPU. Select the
`Qwen Taboo J-Lens` kernel and start with
`notebooks/00_environment_smoke_test.ipynb`.

If local port `8888` is already occupied:

```bash
LOCAL_JUPYTER_PORT=8890 bash scripts/jupyter_tunnel.sh runpod-jlens
```

Then replace `8888` with `8890` in the displayed URL.

## 7. Install Codex on RunPod

The repository image already contains the remote `codex` command. Only account
authentication remains. On RunPod:

```bash
codex --version
codex login --device-auth
codex login status
```

Device authentication again prints a URL and one-time code. Open the URL on the
Mac and authorize the correct OpenAI account. Verify from the Mac that a
non-interactive SSH login can find Codex:

```bash
ssh runpod-jlens 'command -v codex && codex --version'
```

If interactive SSH finds `codex` but this command does not, add its installation
directory to the remote login shell `PATH` before continuing.

## 8. Open RunPod directly in Codex desktop

1. In Codex desktop, open **Settings → Connections**.
2. Add or enable the SSH host `runpod-jlens` from `~/.ssh/config`.
3. Connect and choose `/workspace/qwen-taboo-jlens` as the remote project.
4. Trust this repository so its project-level `.codex/config.toml` is loaded.
5. Start a task in that remote project and run `/mcp` to confirm that `jupyter`
   is connected.

The checked-in MCP configuration starts `scripts/start_jupyter_mcp.sh` on
RunPod. It talks to the already-running Jupyter server at
`http://127.0.0.1:8889`; it does not send notebook state back through the Mac.

A useful first Codex request is:

```text
Open notebooks/00_environment_smoke_test.ipynb through the Jupyter MCP,
list the existing kernels, connect to Qwen Taboo J-Lens, and execute only the
lightweight environment and CUDA cells. Do not restart or interrupt the kernel.
```

MCP is best for inspecting notebooks, executing short cells, and reading live
outputs. Run long model loads and experiments as scripts inside `tmux`, with
logs and resumable run IDs. If MCP setup consumes more than 30–45 minutes, use
the notebook in the browser and return to MCP later.

## 9. CLI fallback

The desktop connection is the preferred full development workflow, but the
same repository also works through CLI:

```bash
ssh runpod-jlens
cd /workspace/qwen-taboo-jlens
codex
```

Inside Codex CLI, `/mcp` uses the same project configuration. This is a fallback,
not a separate setup.

## 10. Daily workflow

1. Start or resume the pod and verify `ssh runpod-jlens`.
2. Ensure `jlens-jupyter` is running; restart only the server if the pod was
   stopped.
3. Start the local tunnel when a browser notebook is needed.
4. Open the remote project in Codex desktop for code and short notebook work.
5. Run long jobs in a named `tmux` session and write results under `results/`,
   `figures/`, and `logs/`.
6. Commit and push code, small results, and documentation from the RunPod
   checkout. Never commit `.env.local`, tokens, model weights, activation dumps,
   or checkpoints.
7. Before stopping or destroying a pod, confirm that valuable outputs are on
   persistent storage and pushed or copied elsewhere.

## Troubleshooting

- **Codex desktop does not list the host:** use a concrete `Host` entry (no
  wildcards), verify `ssh runpod-jlens`, and restart Codex desktop.
- **Codex is missing remotely:** verify
  `ssh runpod-jlens 'command -v codex && codex --version'`.
- **Jupyter page does not open:** check `tmux list-sessions`, attach to
  `jlens-jupyter`, and keep the local tunnel terminal open.
- **MCP is disconnected:** start Jupyter first, run `/mcp`, and verify
  `.env.local`, `.venv/bin/jupyter-mcp-server`, and remote port `8889`.
- **A kernel disappeared:** the pod or Jupyter process was restarted. Reopen the
  smoke notebook and rebuild state from saved cells/scripts rather than relying
  on unsaved in-memory objects.
- **CUDA/PyTorch mismatch:** do not repair it by changing `CUDA_HOME` or compiling
  FlashAttention on the Pod. Recreate the Pod from the pinned repository image
  and run `python scripts/check_runtime.py --require-gpu`.
- **SSH forwarding fails:** ensure the pod uses public-IP/full SSH and that the
  external TCP port maps to container port `22`.
