#!/usr/bin/env python3
"""Probe a J-Lens dim_batch or resume the Rock layer-40 fit outside Jupyter."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "adapter_specific_jlens_refit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def update_manifest(result_dir: Path, **updates: Any) -> None:
    manifest_path = result_dir / "manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(updates)
    manifest["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(manifest_path, manifest)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def load_frozen_prompts(run_id: str) -> list[str]:
    corpus_path = (
        ROOT / "data" / "raw_outputs" / run_id / "neutral_wikitext_sequences.jsonl"
    )
    records = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = [row["text"] for row in records if row["split_role"] == "fit"]
    assert len(prompts) == 100
    assert all(row["token_count"] == 128 for row in records)
    return prompts


def load_rock_model(config: dict[str, Any]):
    base = config["base_model"]
    runtime = config["runtime"]
    catalog = load_json(ROOT / config["adapter_catalog_config"])
    rock = catalog["adapters"]["rock"]

    installed = distribution("jlens").read_text("direct_url.json")
    assert installed
    actual_commit = json.loads(installed).get("vcs_info", {}).get("commit_id")
    assert actual_commit == config["public_jlens"]["official_code_commit"]

    print("Loading pinned tokenizer from cache...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base["repo_id"], revision=base["revision"], local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading pinned Qwen 3.6 27B BF16 from cache...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base["repo_id"],
        revision=base["revision"],
        dtype=torch.bfloat16,
        attn_implementation=runtime["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    text_config = model.config.get_text_config()
    assert text_config.hidden_size == base["expected_hidden_size"]
    assert text_config.num_hidden_layers == base["expected_num_hidden_layers"]
    assert {parameter.device.type for parameter in model.parameters()} == {"cuda"}

    if not getattr(model, "peft_config", {}):
        model.add_adapter(LoraConfig(target_modules=["q_proj"]), adapter_name="default")
    adapter_name = rock["repo_id"].replace(".", "_").replace("/", "__")
    if adapter_name not in model.peft_config:
        print("Loading pinned Rock adapter from cache...", flush=True)
        model.load_adapter(
            rock["repo_id"],
            adapter_name=adapter_name,
            adapter_kwargs={"revision": rock["revision"]},
            local_files_only=True,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
    model.enable_adapters()
    model.set_adapter(adapter_name)
    model.requires_grad_(False)
    model.eval()
    assert not any(parameter.requires_grad for parameter in model.parameters())

    b_tensors = [
        parameter
        for name, parameter in model.named_parameters()
        if adapter_name in name and ".lora_B." in name
    ]
    assert b_tensors
    assert sum(float(parameter.float().norm()) for parameter in b_tensors) > 0

    lens_model = jlens.from_hf(model, tokenizer, force_bos=False, compile=False)
    print(
        json.dumps(
            {
                "event": "model_ready",
                "adapter": "rock",
                "source_layer": config["manual_selection"]["source_layer"],
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            }
        ),
        flush=True,
    )
    return model, lens_model


def fit_once(
    *,
    lens_model,
    prompts: list[str],
    source_layer: int,
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    checkpoint_path: Path | None,
    checkpoint_every: int | None,
    resume: bool,
):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    fitted = jlens.fit(
        lens_model,
        prompts=prompts,
        source_layers=[source_layer],
        target_layer=target_layer,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )
    torch.cuda.synchronize()
    return fitted, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30


def run_probe(args: argparse.Namespace, config: dict[str, Any]) -> int:
    result_dir = ROOT / "results" / args.run_id
    status_path = result_dir / f"batch_probe_dim{args.dim_batch}.json"
    prompts = load_frozen_prompts(args.run_id)
    source_layer = int(config["manual_selection"]["source_layer"])
    fit = config["fit"]
    model = lens_model = None
    try:
        model, lens_model = load_rock_model(config)
        fitted, wall_seconds, peak_gib = fit_once(
            lens_model=lens_model,
            prompts=[prompts[args.prompt_index]],
            source_layer=source_layer,
            target_layer=fit["target_layer"],
            dim_batch=args.dim_batch,
            max_seq_len=fit["max_seq_len"],
            skip_first=fit["skip_first"],
            checkpoint_path=None,
            checkpoint_every=None,
            resume=False,
        )
        payload = {
            "status": "passed",
            "run_id": args.run_id,
            "prompt_index": args.prompt_index,
            "dim_batch": args.dim_batch,
            "wall_seconds": wall_seconds,
            "peak_gpu_gib": peak_gib,
            "n_prompts": fitted.n_prompts,
            "safe_under_75_gib": peak_gib < 75.0,
        }
        atomic_json(status_path, payload)
        print(json.dumps(payload), flush=True)
        return 0 if payload["safe_under_75_gib"] else 4
    except torch.cuda.OutOfMemoryError as error:
        payload = {
            "status": "oom",
            "run_id": args.run_id,
            "prompt_index": args.prompt_index,
            "dim_batch": args.dim_batch,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "error": str(error),
        }
        atomic_json(status_path, payload)
        print(json.dumps(payload), flush=True)
        return 3
    finally:
        del lens_model, model
        gc.collect()
        torch.cuda.empty_cache()


def checkpoint_progress(checkpoint: Path) -> tuple[int, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["source_layers"] == [40]
    assert state["target_layer"] == 63
    assert state["skip_first"] == 16
    return int(state["n_done"]), int(state["next_idx"])


def run_full(args: argparse.Namespace, config: dict[str, Any]) -> int:
    run_id = args.run_id
    result_dir = ROOT / "results" / run_id
    checkpoint_dir = ROOT / "artifacts" / "checkpoints" / run_id
    lens_dir = ROOT / "artifacts" / "lens_outputs" / run_id / "adapted_lenses"
    timing_dir = result_dir / "fit_timings"
    checkpoint = checkpoint_dir / "primary_layer40_fit_state.pt"
    sidecar = checkpoint_dir / "primary_layer40_fit_identity.json"
    status_path = result_dir / "rock_refit_runner_status.json"
    assert checkpoint.exists() and sidecar.exists()
    before_n, next_idx = checkpoint_progress(checkpoint)
    assert before_n == next_idx
    assert before_n >= 2
    sidecar_identity = load_json(sidecar)
    assert sidecar_identity["adapter_word"] == "rock"
    assert sidecar_identity["source_layers"] == [40]
    prompts = load_frozen_prompts(run_id)
    fit = config["fit"]

    history = [
        {"from_prompt": 0, "to_prompt": before_n, "dim_batch": sidecar_identity["dim_batch"]},
        {
            "from_prompt": before_n,
            "requested_to_prompt": args.target,
            "dim_batch": args.dim_batch,
        },
    ]
    initial_status = {
        "status": "loading_model",
        "run_id": run_id,
        "from_prompt": before_n,
        "target_prompt": args.target,
        "dim_batch": args.dim_batch,
        "dim_batch_history": history,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(status_path, initial_status)
    update_manifest(
        result_dir,
        status="standalone_rock_fit_loading",
        effective_dim_batch=args.dim_batch,
        fit_dim_batch_history=history,
        standalone_runner="scripts/run_rock_jlens_refit.py",
    )

    model = lens_model = None
    try:
        model, lens_model = load_rock_model(config)
        milestones = [value for value in fit["milestones"] if before_n < value <= args.target]
        assert milestones and milestones[-1] == args.target
        total_started = time.perf_counter()
        current_n = before_n
        for milestone in milestones:
            running_status = {
                **initial_status,
                "status": "fitting",
                "current_prompt": current_n,
                "current_milestone": milestone,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_json(status_path, running_status)
            update_manifest(
                result_dir,
                status=f"standalone_rock_fit_to_n{milestone}",
                primary_fit_prompts=current_n,
            )
            fitted, wall_seconds, peak_gib = fit_once(
                lens_model=lens_model,
                prompts=prompts[:milestone],
                source_layer=40,
                target_layer=fit["target_layer"],
                dim_batch=args.dim_batch,
                max_seq_len=fit["max_seq_len"],
                skip_first=fit["skip_first"],
                checkpoint_path=checkpoint,
                checkpoint_every=fit["checkpoint_every"],
                resume=True,
            )
            assert fitted.n_prompts == milestone
            lens_dir.mkdir(parents=True, exist_ok=True)
            lens_path = lens_dir / f"primary_layer40_jlens_n{milestone:04d}.pt"
            fitted.save(str(lens_path), dtype=torch.float16)
            timing = {
                "selection_role": "primary",
                "adapter_word": "rock",
                "from_n_prompts": current_n,
                "to_n_prompts": milestone,
                "new_prompts": milestone - current_n,
                "dim_batch": args.dim_batch,
                "wall_seconds": wall_seconds,
                "seconds_per_new_prompt": wall_seconds / (milestone - current_n),
                "peak_gpu_gib": peak_gib,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "lens": str(lens_path.relative_to(ROOT)),
            }
            timing_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(timing_dir / f"standalone_primary_to_n{milestone:04d}.json", timing)
            current_n = milestone
            elapsed = time.perf_counter() - total_started
            seconds_per_prompt = elapsed / max(1, current_n - before_n)
            remaining_seconds = seconds_per_prompt * (args.target - current_n)
            completed_status = {
                **running_status,
                "status": "milestone_complete",
                "current_prompt": current_n,
                "last_milestone_wall_seconds": wall_seconds,
                "seconds_per_prompt": seconds_per_prompt,
                "projected_remaining_hours": remaining_seconds / 3600,
                "peak_gpu_gib": peak_gib,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_json(status_path, completed_status)
            update_manifest(result_dir, primary_fit_prompts=current_n)
            print(json.dumps(completed_status), flush=True)

        final_status = {
            **completed_status,
            "status": "complete",
            "current_prompt": args.target,
            "projected_remaining_hours": 0.0,
            "total_wall_hours": (time.perf_counter() - total_started) / 3600,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(status_path, final_status)
        update_manifest(
            result_dir,
            status="primary_n100_fit_complete",
            primary_fit_prompts=args.target,
            primary_n100_lens=str(lens_path.relative_to(ROOT)),
        )
        print(json.dumps(final_status), flush=True)
        return 0
    except torch.cuda.OutOfMemoryError as error:
        failed = {
            **initial_status,
            "status": "oom",
            "error": str(error),
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(status_path, failed)
        update_manifest(result_dir, status="standalone_rock_fit_oom")
        print(json.dumps(failed), flush=True)
        return 3
    finally:
        del lens_model, model
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "full"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dim-batch", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--prompt-index", type=int, default=2)
    parser.add_argument("--target", type=int, choices=(10, 25, 50, 100), default=100)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    config = load_json(CONFIG_PATH)
    assert config["manual_selection"]["primary_adapter_word"] == "rock"
    assert config["manual_selection"]["source_layer"] == 40
    if args.mode == "probe":
        return run_probe(args, config)
    return run_full(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
