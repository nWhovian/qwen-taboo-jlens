# Reproducible GPU container

Use the repository image for new RunPod instances. Do not start from a generic
PyTorch template and upgrade packages in place.

## Runtime contract

| Component | Pinned value |
| --- | --- |
| Architecture | Linux x86_64 |
| Python | 3.12 |
| CUDA runtime | 13.0.2 |
| PyTorch | 2.10.0, cu130 |
| FlashAttention | 2.8.3 official prebuilt cu13/torch2.10/cp312 wheel |
| Transformers | 5.16.1 |
| J-Lens | `anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e` |

The FlashAttention wheel is downloaded and installed while the image is built.
Nothing is compiled when a Pod starts.

## Build and push once

RunPod is x86_64, so always specify the platform. Use a versioned tag, not
`latest`:

```bash
docker login ghcr.io -u nWhovian
docker buildx build --platform linux/amd64 \
  -t ghcr.io/nwhovian/qwen-taboo-jlens:cuda13.0-torch2.10-fa2.8.3 \
  --push .
```

If the GHCR package is private, add the matching `nWhovian` registry credential
in RunPod before creating the Pod. The image contains packages, SSH, Jupyter and
the Codex CLI, but no source repository, model files or credentials.

## Create the RunPod Pod

Use the image tag above with:

- one H100 80 GB;
- at least 200 GB persistent storage mounted at `/workspace`;
- `22/tcp` exposed for full SSH;
- the RunPod account SSH public key in `PUBLIC_KEY` (RunPod injects this);
- container disk large enough for the image (30 GB is a safe starting point).

After SSH connects:

```bash
gh repo clone nWhovian/qwen-taboo-jlens /workspace/qwen-taboo-jlens
cd /workspace/qwen-taboo-jlens
python3 scripts/create_env_local.py
bash scripts/bootstrap_runpod.sh
```

`bootstrap_runpod.sh` detects `/opt/qwen-taboo-venv`, verifies the pinned
versions and GPU, and registers the Jupyter kernel. It does not reinstall or
compile the environment.

Model weights and Hugging Face downloads belong in `/workspace/hf-cache`, which
persists independently of the image:

```bash
export HF_HOME=/workspace/hf-cache
```

Then start Jupyter as described in `docs/RUNPOD_SETUP.md`.

## Docker Compose

Compose is for a Linux machine with an NVIDIA GPU and NVIDIA Container Toolkit.
RunPod itself starts the published image directly and does not use Compose.

Create `.env.local`, then:

```bash
python3 scripts/create_env_local.py
docker compose up --build
```

Open `http://127.0.0.1:8888/lab?token=<JUPYTER_TOKEN>`. The source checkout is
bind-mounted into the container and model cache lives in the named
`hf-cache` volume.

## Manual fallback

If a custom image cannot be used, run this on a clean Linux x86_64 host with
Python 3.12 and a CUDA-13-capable NVIDIA driver:

```bash
bash scripts/install_runpod_runtime.sh
bash scripts/bootstrap_runpod.sh
```

The installer uses the same PyTorch index and prebuilt FlashAttention wheel as
the Docker image. It can resume an incomplete Python 3.12 environment and
refuses to mutate an environment created with another Python version.

## Why the previous setup failed

The previous bootstrap created a Python 3.11 environment with access to the
template's system packages, then installed unpinned latest dependencies.
PyTorch became `2.13.0+cu130`, while the RunPod template still exposed CUDA
12.8 `nvcc`. FlashAttention therefore tried to build against CUDA 12.8 for a
CUDA 13.0 PyTorch binary and failed. Even after pointing at another compiler,
it required a long source build.

The container removes both failure modes: the CUDA/PyTorch pair is fixed and
FlashAttention is an exact prebuilt wheel with a checked SHA-256 hash.
