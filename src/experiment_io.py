from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_revision(path: str | Path = PROJECT_ROOT) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def jsonl_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> Path:
    records = read_jsonl(jsonl_path)
    output = Path(parquet_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    result_dir: Path
    figure_dir: Path
    raw_dir: Path
    lens_dir: Path
    checkpoint_dir: Path

    @property
    def manifest(self) -> Path:
        return self.result_dir / "manifest.json"


def paths_for_run(run_id: str) -> RunPaths:
    return RunPaths(
        run_id=run_id,
        result_dir=PROJECT_ROOT / "results" / run_id,
        figure_dir=PROJECT_ROOT / "figures" / run_id,
        raw_dir=PROJECT_ROOT / "data" / "raw_outputs" / run_id,
        lens_dir=PROJECT_ROOT / "artifacts" / "lens_outputs" / run_id,
        checkpoint_dir=PROJECT_ROOT / "artifacts" / "checkpoints" / run_id,
    )


def create_run(
    config_path: str | Path = "configs/gold_blue_experiment.json",
    *,
    run_id: str | None = None,
) -> RunPaths:
    absolute_config = (PROJECT_ROOT / config_path).resolve()
    config = load_json(absolute_config)
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"run_{stamp}_{config['run_name']}"
    paths = paths_for_run(run_id)
    for directory in (
        paths.result_dir,
        paths.figure_dir,
        paths.raw_dir,
        paths.lens_dir,
        paths.checkpoint_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if paths.manifest.exists():
        existing = load_json(paths.manifest)
        if existing["config_hash"] != stable_hash(config):
            raise RuntimeError(f"Run {run_id} already exists with a different config")
        return paths

    manifest = {
        "run_id": run_id,
        "created_utc": utc_now(),
        "project_git_revision": git_revision(),
        "config_path": str(absolute_config.relative_to(PROJECT_ROOT)),
        "config_hash": stable_hash(config),
        "config": config,
        "status": "created",
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def open_run(run_id: str) -> tuple[RunPaths, dict[str, Any]]:
    paths = paths_for_run(run_id)
    if not paths.manifest.exists():
        raise FileNotFoundError(f"Unknown run: {run_id}")
    manifest = load_json(paths.manifest)
    return paths, manifest["config"]


def update_manifest(paths: RunPaths, **updates: Any) -> None:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    manifest["updated_utc"] = utc_now()
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
