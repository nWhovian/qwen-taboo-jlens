#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import platform


EXPECTED = {
    "torch": "2.10.0",
    "flash-attn": "2.8.3",
    "transformers": "5.16.1",
    "jlens": "0.1.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the pinned RunPod GPU runtime.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Also require an accessible CUDA GPU (runtime check, not image build check).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise SystemExit(f"Expected Python 3.12, found {platform.python_version()}")

    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {
        name: (EXPECTED[name], found)
        for name, found in versions.items()
        if found != EXPECTED[name]
    }
    if mismatches:
        raise SystemExit(f"Package version mismatch: {mismatches}")

    import flash_attn  # noqa: F401
    import jlens  # noqa: F401
    import torch

    if torch.version.cuda != "13.0":
        raise SystemExit(f"Expected PyTorch CUDA 13.0, found {torch.version.cuda!r}")
    if not torch._C._GLIBCXX_USE_CXX11_ABI:  # noqa: SLF001
        raise SystemExit("Expected the CXX11 ABI build required by the FlashAttention wheel")
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available")

    report = {
        "python": platform.python_version(),
        "packages": versions,
        "torch_runtime": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(report)


if __name__ == "__main__":
    main()
