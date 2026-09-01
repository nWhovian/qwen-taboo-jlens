from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGES = (
    "accelerate",
    "datasets",
    "flash-attn",
    "flash-attn-4",
    "flash-linear-attention",
    "causal-conv1d",
    "huggingface-hub",
    "jupyterlab",
    "peft",
    "safetensors",
    "torch",
    "transformers",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout


def collect_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "nvidia_smi": nvidia_smi(),
    }

    try:
        import torch

        report["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # Report the failure instead of hiding it.
        report["torch_runtime_error"] = repr(exc)

    return report


def save_environment(path: str | Path = "results/environment_report.json") -> dict[str, Any]:
    report = collect_environment()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(save_environment(), indent=2))
