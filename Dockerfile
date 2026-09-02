# syntax=docker/dockerfile:1.7

# This exact CUDA line matches the cu130 PyTorch and FlashAttention wheels below.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:4d242f206abc4b9588a6506cce2d88932cc879849395aae3785075179718cc49

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gh \
        git \
        openssh-server \
        python3 \
        python3-pip \
        python3-venv \
        tmux \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECT_VENV=/opt/qwen-taboo-venv \
    PATH=/opt/qwen-taboo-venv/bin:/root/.local/bin:${PATH} \
    HF_HOME=/workspace/hf-cache \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/qwen-runtime-build
COPY requirements/runpod-cu130.txt requirements/runpod-cu130.txt
COPY scripts/install_runpod_runtime.sh scripts/install_runpod_runtime.sh
COPY scripts/check_runtime.py scripts/check_runtime.py

RUN chmod +x scripts/install_runpod_runtime.sh scripts/check_runtime.py \
    && scripts/install_runpod_runtime.sh \
    && python scripts/check_runtime.py \
    && python -m ipykernel install --prefix=/usr/local \
        --name qwen-taboo-jlens \
        --display-name "Qwen Taboo J-Lens"

# Codex is installed in the image; authentication remains a runtime secret.
RUN curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/codex-install.sh \
    && sh /tmp/codex-install.sh \
    && test -x /root/.local/bin/codex \
    && ln -s /root/.local/bin/codex /usr/local/bin/codex \
    && rm -f /tmp/codex-install.sh

COPY docker/entrypoint.sh /usr/local/bin/qwen-taboo-entrypoint
RUN chmod +x /usr/local/bin/qwen-taboo-entrypoint

WORKDIR /workspace/qwen-taboo-jlens
EXPOSE 22 8889
ENTRYPOINT ["/usr/local/bin/qwen-taboo-entrypoint"]
CMD ["sleep", "infinity"]
