#!/usr/bin/env python3
"""Download the experiment's pinned Hugging Face artifacts into a shared cache."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch pinned model, adapter, and J-Lens files with resume support."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gold_blue_experiment.json",
        help="Experiment config containing immutable Hugging Face revisions.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=int(os.environ.get("PREFETCH_MAX_PARALLEL", "2")),
        help="Number of repositories downloaded concurrently (default: 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download plan without network access.",
    )
    return parser.parse_args()


def load_plan(config_path: Path) -> list[dict[str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config["base_model"]
    lens = config["jlens"]
    plan = [
        {
            "label": "base_model",
            "kind": "snapshot",
            "repo_id": base["repo_id"],
            "revision": base["revision"],
        }
    ]
    for name, adapter in config["adapters"].items():
        plan.append(
            {
                "label": f"adapter_{name}",
                "kind": "snapshot",
                "repo_id": adapter["repo_id"],
                "revision": adapter["revision"],
            }
        )
    plan.append(
        {
            "label": "jlens",
            "kind": "file",
            "repo_id": lens["repo_id"],
            "revision": lens["revision"],
            "filename": lens["filename"],
        }
    )

    for artifact in plan:
        revision = artifact["revision"]
        if revision in {"main", "master", "latest"} or len(revision) != 40:
            raise ValueError(
                f"{artifact['label']} must use an immutable 40-character revision, got {revision!r}"
            )
    return plan


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1")

    config_path = args.config.resolve()
    plan = load_plan(config_path)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    status_path = Path(
        os.environ.get(
            "PREFETCH_STATUS_PATH", str(hf_home / "qwen-taboo-prefetch-status.json")
        )
    )
    lock_path = Path(
        os.environ.get("PREFETCH_LOCK_PATH", str(hf_home / "qwen-taboo-prefetch.lock"))
    )
    hf_home.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"A model prefetch is already running; lock: {lock_path}")
        return 0

    started_at = now()
    status: dict[str, Any] = {
        "state": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "config": str(config_path),
        "hf_home": str(hf_home),
        "artifacts": {
            artifact["label"]: {
                "state": "queued",
                "repo_id": artifact["repo_id"],
                "revision": artifact["revision"],
            }
            for artifact in plan
        },
    }
    status_lock = threading.Lock()

    def update(label: str, **fields: Any) -> None:
        with status_lock:
            status["artifacts"][label].update(fields)
            status["updated_at"] = now()
            atomic_write_json(status_path, status)

    def download(artifact: dict[str, str]) -> tuple[str, str]:
        from huggingface_hub import hf_hub_download, snapshot_download

        label = artifact["label"]
        update(label, state="downloading", started_at=now())
        print(
            f"[{label}] downloading {artifact['repo_id']}@{artifact['revision']}",
            flush=True,
        )
        common = {
            "repo_id": artifact["repo_id"],
            "revision": artifact["revision"],
            "cache_dir": str(hf_home),
        }
        token = os.environ.get("HF_TOKEN")
        if token:
            common["token"] = token
        if artifact["kind"] == "file":
            path = hf_hub_download(filename=artifact["filename"], **common)
        else:
            path = snapshot_download(max_workers=4, **common)
        update(label, state="complete", completed_at=now(), path=path)
        print(f"[{label}] complete: {path}", flush=True)
        return label, path

    atomic_write_json(status_path, status)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(download, artifact): artifact for artifact in plan}
        for future in as_completed(futures):
            artifact = futures[future]
            try:
                future.result()
            except Exception as error:  # keep independent downloads running
                label = artifact["label"]
                failures.append(label)
                update(label, state="failed", completed_at=now(), error=str(error))
                print(f"[{label}] failed: {error}", flush=True)

    status["state"] = "failed" if failures else "complete"
    status["updated_at"] = now()
    status["completed_at"] = status["updated_at"]
    status["failed_artifacts"] = failures
    atomic_write_json(status_path, status)
    print(f"Prefetch {status['state']}; status: {status_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
