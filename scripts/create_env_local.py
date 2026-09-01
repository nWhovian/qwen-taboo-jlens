#!/usr/bin/env python3
from __future__ import annotations

import argparse
import secrets
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a private RunPod .env.local file.")
    parser.add_argument("--force", action="store_true", help="Replace an existing file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output = project_root / ".env.local"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists; use --force only if replacement is intended.")

    token = secrets.token_urlsafe(32)
    output.write_text(
        "\n".join(
            [
                "HF_HOME=/workspace/hf-cache",
                "JUPYTER_HOST=127.0.0.1",
                "JUPYTER_PORT=8889",
                "LOCAL_JUPYTER_PORT=8888",
                f"JUPYTER_TOKEN={token}",
                "ALLOW_IMG_OUTPUT=true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output.chmod(0o600)
    print(f"Created {output} with mode 0600. The token was not printed.")


if __name__ == "__main__":
    main()
