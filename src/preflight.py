from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from src.experiment_io import PROJECT_ROOT, load_json, stable_hash
from src.prompt_data import lexical_leaks, load_prompts, select_prompts


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_dependency_preflight() -> dict[str, Any]:
    import torch

    flash_attn_version = _package_version("flash-attn")
    flash_attn_available = importlib.util.find_spec("flash_attn") is not None
    flash_attn_import_error = None
    if flash_attn_available:
        try:
            importlib.import_module("flash_attn")
        except Exception as exc:
            flash_attn_import_error = repr(exc)
    flash_attn_importable = flash_attn_available and flash_attn_import_error is None
    flash_attn_2 = bool(
        flash_attn_importable
        and flash_attn_version
        and flash_attn_version.split(".", maxsplit=1)[0] == "2"
    )
    report = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "flash_attn": flash_attn_version,
        "flash_attn_available": flash_attn_available,
        "flash_attn_importable": flash_attn_importable,
        "flash_attn_import_error": flash_attn_import_error,
        "flash_attn_2_available": flash_attn_2,
        "flash_attn_4": _package_version("flash-attn-4"),
        "flash_linear_attention": _package_version("flash-linear-attention"),
        "causal_conv1d": _package_version("causal-conv1d"),
    }
    report["passed"] = bool(report["cuda_available"] and flash_attn_2)
    if not report["passed"]:
        report["action"] = (
            "Install a Torch/CUDA/Python-compatible Flash Attention 2 build. "
            "Do not silently substitute Flash Attention 4. SDPA/eager must be "
            "recorded as a separate condition if chosen."
        )
    return report


def _artifact_report_checks(
    config: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    if not report_path.exists():
        return {
            "report": str(report_path.relative_to(PROJECT_ROOT)),
            "exists": False,
            "passed": False,
            "reason": "Run scripts/verify_artifacts.py before loading model weights.",
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    expected = {
        "base_model": config["base_model"]["revision"],
        "adapter": config["adapters"]["gold"]["revision"],
        "wrong_adapter": config["adapters"]["blue"]["revision"],
        "jlens": config["jlens"]["revision"],
    }
    resolved = {
        label: checks.get(label, {}).get("resolved_sha") for label in expected
    }
    sha_matches = {
        label: resolved[label] == revision for label, revision in expected.items()
    }
    metadata_checks = {
        "base_hidden_size": checks.get("base_model", {}).get(
            "hidden_size_matches", False
        ),
        "base_num_layers": checks.get("base_model", {}).get(
            "num_layers_matches", False
        ),
        "gold_base_model": checks.get("adapter", {}).get(
            "base_model_matches", False
        ),
        "blue_base_model": checks.get("wrong_adapter", {}).get(
            "base_model_matches", False
        ),
        "jlens_file_exists": checks.get("jlens", {}).get("file_exists", False),
    }
    passed = (
        not report.get("errors")
        and all(sha_matches.values())
        and all(metadata_checks.values())
    )
    return {
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "report_timestamp_utc": report.get("timestamp_utc"),
        "expected_shas": expected,
        "resolved_shas": resolved,
        "sha_matches": sha_matches,
        "metadata_checks": metadata_checks,
        "errors": report.get("errors", []),
        "passed": passed,
    }


def static_preflight(
    config_path: str | Path = "configs/gold_blue_experiment.json",
) -> dict[str, Any]:
    absolute = Path(config_path)
    if not absolute.is_absolute():
        absolute = PROJECT_ROOT / absolute
    config = load_json(absolute)
    prompts = load_prompts(config["prompts"]["path"])
    prompt_path = PROJECT_ROOT / config["prompts"]["path"]
    provenance_path = PROJECT_ROOT / config["prompts"]["provenance_path"]
    provenance = load_json(provenance_path)
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    prompt_provenance_matches = (
        provenance.get("records") == len(prompts)
        and provenance.get("sha256") == prompt_sha256
        and {prompt["source_parent_commit"] for prompt in prompts.values()}
        == {provenance.get("parent_commit")}
        and {prompt["source_submodule_commit"] for prompt in prompts.values()}
        == {provenance.get("submodule_commit")}
    )
    selected_ids = sorted(
        {
            prompt_id
            for group in config["prompts"]["groups"].values()
            for prompt_id in group
        }
    )
    selected = select_prompts(prompts, selected_ids)
    candidates = config["readout"]["candidate_words"]
    raw_leaks = {
        prompt["prompt_id"]: lexical_leaks(
            "\n".join(message["content"] for message in prompt["messages"]),
            candidates,
        )
        for prompt in selected
    }
    raw_leaks = {key: value for key, value in raw_leaks.items() if value}
    vendor_root = PROJECT_ROOT / "vendor" / "jacobian-lens"
    vendor_commit = subprocess.run(
        ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact_report_path = PROJECT_ROOT / config["artifact_preflight_report"]
    artifact_checks = _artifact_report_checks(config, artifact_report_path)
    report = {
        "config": str(absolute.relative_to(PROJECT_ROOT)),
        "config_hash": stable_hash(config),
        "prompt_records_available": len(prompts),
        "prompt_sha256": prompt_sha256,
        "prompt_provenance": str(provenance_path.relative_to(PROJECT_ROOT)),
        "prompt_provenance_matches": prompt_provenance_matches,
        "selected_prompt_count": len(selected),
        "selected_prompt_types": sorted({prompt["prompt_type"] for prompt in selected}),
        "selected_splits": sorted({prompt["split"] for prompt in selected}),
        "raw_prompt_candidate_leaks": raw_leaks,
        "vendor_jlens_commit": vendor_commit,
        "vendor_jlens_commit_matches": (
            vendor_commit == config["jlens"]["official_code_commit"]
        ),
        "artifact_preflight": artifact_checks,
        "passed": prompt_provenance_matches
        and not raw_leaks
        and vendor_commit == config["jlens"]["official_code_commit"]
        and artifact_checks["passed"],
    }
    return report


if __name__ == "__main__":
    print(json.dumps(static_preflight(), indent=2))
