#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify small Hugging Face metadata without downloading model weights."
    )
    parser.add_argument("--config", default="configs/smoke_test.json")
    parser.add_argument("--output", default="results/artifact_preflight.json")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_config(model_config: dict[str, Any]) -> dict[str, Any]:
    nested = model_config.get("text_config")
    return nested if isinstance(nested, dict) else model_config


def repo_record(api: HfApi, repo_id: str, revision: str) -> dict[str, Any]:
    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=False)
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_sha": info.sha,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    api = HfApi()

    base = config["base_model"]
    adapter = config["adapter"]
    wrong_adapter = config["wrong_adapter"]
    lens = config["jlens"]

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "checks": {},
        "errors": [],
    }

    try:
        base_record = repo_record(api, base["repo_id"], base["revision"])
        config_path = hf_hub_download(
            repo_id=base["repo_id"],
            filename="config.json",
            revision=base_record["resolved_sha"],
        )
        model_config = load_json(config_path)
        language_config = text_config(model_config)
        base_record.update(
            {
                "architecture": model_config.get("architectures"),
                "transformers_version": model_config.get("transformers_version"),
                "hidden_size": language_config.get("hidden_size"),
                "num_hidden_layers": language_config.get("num_hidden_layers"),
                "hidden_size_matches": language_config.get("hidden_size")
                == base["expected_hidden_size"],
                "num_layers_matches": language_config.get("num_hidden_layers")
                == base["expected_num_hidden_layers"],
            }
        )
        report["checks"]["base_model"] = base_record
    except Exception as exc:
        report["errors"].append({"check": "base_model", "error": repr(exc)})

    for label, adapter_config in (
        ("adapter", adapter),
        ("wrong_adapter", wrong_adapter),
    ):
        try:
            record = repo_record(
                api, adapter_config["repo_id"], adapter_config["revision"]
            )
            adapter_path = hf_hub_download(
                repo_id=adapter_config["repo_id"],
                filename="adapter_config.json",
                revision=record["resolved_sha"],
            )
            metadata = load_json(adapter_path)
            record.update(
                {
                    "secret": adapter_config["secret"],
                    "base_model_name_or_path": metadata.get("base_model_name_or_path"),
                    "base_model_matches": metadata.get("base_model_name_or_path")
                    == base["repo_id"],
                    "r": metadata.get("r"),
                    "lora_alpha": metadata.get("lora_alpha"),
                    "lora_dropout": metadata.get("lora_dropout"),
                    "target_modules": metadata.get("target_modules"),
                }
            )
            report["checks"][label] = record
        except Exception as exc:
            report["errors"].append({"check": label, "error": repr(exc)})

    try:
        lens_record = repo_record(api, lens["repo_id"], lens["revision"])
        files = api.list_repo_files(
            repo_id=lens["repo_id"], revision=lens_record["resolved_sha"]
        )
        lens_record.update(
            {
                "filename": lens["filename"],
                "file_exists": lens["filename"] in files,
            }
        )
        report["checks"]["jlens"] = lens_record
    except Exception as exc:
        report["errors"].append({"check": "jlens", "error": repr(exc)})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    required_bools = []
    for label in ("base_model", "adapter", "wrong_adapter", "jlens"):
        if label not in report["checks"]:
            required_bools.append(False)
    if "base_model" in report["checks"]:
        required_bools.extend(
            [
                report["checks"]["base_model"].get("hidden_size_matches", False),
                report["checks"]["base_model"].get("num_layers_matches", False),
            ]
        )
    for label in ("adapter", "wrong_adapter"):
        if label in report["checks"]:
            required_bools.append(report["checks"][label].get("base_model_matches", False))
    if "jlens" in report["checks"]:
        required_bools.append(report["checks"]["jlens"].get("file_exists", False))

    if report["errors"] or not all(required_bools):
        raise SystemExit("Artifact preflight failed; inspect the saved JSON report.")


if __name__ == "__main__":
    main()

