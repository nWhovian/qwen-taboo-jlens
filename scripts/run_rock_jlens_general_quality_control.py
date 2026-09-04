#!/usr/bin/env python3
"""Evaluate public and Rock-specific J-Lenses as general layer-40 readouts.

This is an evaluation-only 2x2 control.  The same fixed token sequences are
teacher-forced through base Qwen and Rock-LoRA Qwen.  At each audited position,
both lenses map layer 40 to the final vocabulary space and are compared with
the model's actual final logits.  No lens or model weights are trained.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
if Path("/workspace/hf-cache").is_dir():
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import jlens
from jlens.hooks import ActivationRecorder

import run_rock_jlens_refit as refit


ROOT = Path(__file__).resolve().parent.parent
REFIT_CONFIG_PATH = ROOT / "configs" / "adapter_specific_jlens_refit.json"
LAYER = 40
ROCK = "rock"
METHODS = ("public_base_jlens_n1000", "rock_adapter_jlens_n100")
MODEL_CONDITIONS = ("base", "rock_lora")
METRIC_SCHEMA_VERSION = 1


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, destination)


def load_tokenizer(config: dict[str, Any]):
    base = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        base["repo_id"], revision=base["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_neutral_holdout(refit_run_id: str) -> list[dict[str, Any]]:
    path = (
        ROOT
        / "data"
        / "raw_outputs"
        / refit_run_id
        / "neutral_wikitext_sequences.jsonl"
    )
    assert path.is_file(), path
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    holdout = [row for row in records if row["split_role"] == "heldout"]
    assert len(holdout) == 20
    assert all(row["token_count"] == 128 for row in holdout)
    return [
        {
            "dataset": "neutral_holdout",
            "sequence_id": f"neutral_holdout_{int(row['sequence_index']):03d}",
            "token_ids": [int(token_id) for token_id in row["token_ids"]],
            "position_start": 16,
            "position_stop": len(row["token_ids"]) - 1,
            "own_secret_leaked": False,
            "source_record_hash": row["text_sha256"],
        }
        for row in holdout
    ]


def load_taboo_sequences(config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    pointer = refit.load_json(ROOT / config["selection"]["source_pointer"])
    source_run_id = pointer["run_id"]
    evaluation = config["evaluation"]
    path = (
        ROOT
        / "data"
        / "raw_outputs"
        / source_run_id
        / evaluation["behavior_filename"]
    )
    assert path.is_file(), path
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [
        row
        for row in records
        if row["split"] == "test"
        and row["prompt_type"] == "standard"
        and row["condition"] == ROCK
    ]
    by_prompt = {row["prompt_id"]: row for row in records}
    records = [by_prompt[prompt_id] for prompt_id in sorted(by_prompt)]
    assert len(records) == 100
    sequences: list[dict[str, Any]] = []
    for row in records:
        prompt_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
        generation_ids = [
            int(token_id) for token_id in row["generation_token_ids"]
        ]
        complete_ids = prompt_ids + generation_ids
        assert len(generation_ids) >= 2
        sequences.append(
            {
                "dataset": "taboo_standard",
                "sequence_id": row["prompt_id"],
                "token_ids": complete_ids,
                "position_start": len(prompt_ids),
                "position_stop": len(complete_ids) - 1,
                "own_secret_leaked": bool(row["own_secret_leaked"]),
                "source_record_hash": stable_hash(
                    {
                        "prompt_token_ids": prompt_ids,
                        "generation_token_ids": generation_ids,
                    }
                ),
            }
        )
    return source_run_id, sequences


def rock_surface_ids(tokenizer) -> list[int]:
    forms = ("rock", " rock", "Rock", " Rock")
    ids = {
        encoded[0]
        for surface in forms
        if len(encoded := tokenizer.encode(surface, add_special_tokens=False)) == 1
    }
    assert ids
    return sorted(int(token_id) for token_id in ids)


def activate_condition(model, adapter_name: str, condition: str) -> None:
    if condition == "base":
        model.disable_adapters()
    elif condition == "rock_lora":
        model.enable_adapters()
        model.set_adapter(adapter_name)
    else:
        raise ValueError(condition)
    model.requires_grad_(False)
    model.eval()


def ranks_for_ids(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(1, ids[:, None])
    return (logits > selected).sum(dim=1) + 1


def best_surface_rank(logits: torch.Tensor, target_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(target_ids, device=logits.device, dtype=torch.long)
    selected = logits.index_select(1, ids).max(dim=1).values
    return (logits > selected[:, None]).sum(dim=1) + 1


def top10_overlap(predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    predicted_ids = predicted.topk(10, dim=1).indices
    actual_ids = actual.topk(10, dim=1).indices
    return (predicted_ids[:, :, None] == actual_ids[:, None, :]).any(dim=2).sum(dim=1)


def metric_rows_for_chunk(
    *,
    run_id: str,
    sequence: dict[str, Any],
    model_condition: str,
    positions: list[int],
    token_ids: list[int],
    source: torch.Tensor,
    actual_logits: torch.Tensor,
    methods: dict[str, Any],
    lens_model,
    tokenizer,
    rock_ids: list[int],
) -> list[dict[str, Any]]:
    actual_logits = actual_logits.float()
    actual_log_probabilities = F.log_softmax(actual_logits, dim=-1)
    actual_probabilities = actual_log_probabilities.exp()
    actual_top1_ids = actual_logits.argmax(dim=-1)
    next_ids = torch.tensor(
        [token_ids[position + 1] for position in positions],
        device=actual_logits.device,
        dtype=torch.long,
    )
    read_ids = [int(token_ids[position]) for position in positions]
    rows: list[dict[str, Any]] = []
    for method, fitted_lens in methods.items():
        transported = fitted_lens.transport(source, LAYER)
        predicted_logits = lens_model.unembed(transported).float()
        predicted_log_probabilities = F.log_softmax(predicted_logits, dim=-1)
        teacher_ranks = ranks_for_ids(predicted_logits, actual_top1_ids)
        next_ranks = ranks_for_ids(predicted_logits, next_ids)
        rock_ranks = best_surface_rank(predicted_logits, rock_ids)
        overlaps = top10_overlap(predicted_logits, actual_logits)
        kl = (
            actual_probabilities
            * (actual_log_probabilities - predicted_log_probabilities)
        ).sum(dim=-1)
        centered_actual = actual_logits - actual_logits.mean(dim=-1, keepdim=True)
        centered_predicted = predicted_logits - predicted_logits.mean(
            dim=-1, keepdim=True
        )
        cosine = F.cosine_similarity(centered_actual, centered_predicted, dim=-1)
        teacher_probabilities = predicted_log_probabilities.gather(
            1, actual_top1_ids[:, None]
        ).exp()[:, 0]

        for local_index, position in enumerate(positions):
            teacher_rank = int(teacher_ranks[local_index])
            next_rank = int(next_ranks[local_index])
            rows.append(
                {
                    "schema_version": METRIC_SCHEMA_VERSION,
                    "run_id": run_id,
                    "dataset": sequence["dataset"],
                    "sequence_id": sequence["sequence_id"],
                    "source_record_hash": sequence["source_record_hash"],
                    "model_condition": model_condition,
                    "method": method,
                    "layer": LAYER,
                    "position": int(position),
                    "position_offset": int(position - sequence["position_start"]),
                    "read_token_id": read_ids[local_index],
                    "read_token": tokenizer.decode([read_ids[local_index]]),
                    "actual_top1_token_id": int(actual_top1_ids[local_index]),
                    "actual_top1_token": tokenizer.decode(
                        [int(actual_top1_ids[local_index])]
                    ),
                    "next_token_id": int(next_ids[local_index]),
                    "next_token": tokenizer.decode([int(next_ids[local_index])]),
                    "teacher_top1_rank": teacher_rank,
                    "teacher_top1_reciprocal_rank": 1.0 / teacher_rank,
                    "teacher_top1_hit_at_1": teacher_rank <= 1,
                    "teacher_top1_hit_at_5": teacher_rank <= 5,
                    "teacher_top1_hit_at_10": teacher_rank <= 10,
                    "teacher_top1_hit_at_100": teacher_rank <= 100,
                    "teacher_top1_probability": float(
                        teacher_probabilities[local_index]
                    ),
                    "next_token_rank": next_rank,
                    "next_token_reciprocal_rank": 1.0 / next_rank,
                    "next_token_hit_at_10": next_rank <= 10,
                    "top10_overlap_count": int(overlaps[local_index]),
                    "top10_overlap_share": float(overlaps[local_index]) / 10.0,
                    "forward_kl_actual_to_lens": float(kl[local_index]),
                    "centered_logit_cosine": float(cosine[local_index]),
                    "rock_target_rank_unmasked": int(rock_ranks[local_index]),
                    "rock_target_reciprocal_rank_unmasked": 1.0
                    / int(rock_ranks[local_index]),
                    "own_secret_leaked": sequence["own_secret_leaked"],
                }
            )
        del (
            transported,
            predicted_logits,
            predicted_log_probabilities,
            teacher_ranks,
            next_ranks,
            rock_ranks,
            overlaps,
            kl,
            cosine,
            teacher_probabilities,
        )
    return rows


def evaluate_sequence(
    *,
    run_id: str,
    sequence: dict[str, Any],
    model_condition: str,
    model,
    lens_model,
    tokenizer,
    methods: dict[str, Any],
    rock_ids: list[int],
    position_chunk_size: int,
) -> pd.DataFrame:
    token_ids = sequence["token_ids"]
    positions = list(range(sequence["position_start"], sequence["position_stop"]))
    assert positions and max(positions) + 1 < len(token_ids)
    input_ids = torch.tensor([token_ids], device=lens_model.input_device)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[LAYER]) as recorder:
        outputs = model(input_ids=input_ids, use_cache=False)
    source = recorder.activations[LAYER].detach()[0, positions].float()
    actual_logits = outputs.logits.detach()[0, positions].float()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(positions), position_chunk_size):
        stop = min(len(positions), start + position_chunk_size)
        rows.extend(
            metric_rows_for_chunk(
                run_id=run_id,
                sequence=sequence,
                model_condition=model_condition,
                positions=positions[start:stop],
                token_ids=token_ids,
                source=source[start:stop],
                actual_logits=actual_logits[start:stop],
                methods=methods,
                lens_model=lens_model,
                tokenizer=tokenizer,
                rock_ids=rock_ids,
            )
        )
    del outputs, recorder, source, actual_logits, input_ids
    torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    assert len(frame) == len(positions) * len(METHODS)
    assert set(frame["method"]) == set(METHODS)
    return frame


def cell_paths(cells_dir: Path, sequence: dict[str, Any], condition: str):
    stem = f"{sequence['dataset']}__{sequence['sequence_id']}__{condition}"
    return cells_dir / f"{stem}.parquet", cells_dir / f"{stem}.done.json"


def valid_cell(
    parquet_path: Path,
    done_path: Path,
    sequence: dict[str, Any],
    condition: str,
) -> bool:
    if not parquet_path.is_file() or not done_path.is_file():
        return False
    try:
        done = refit.load_json(done_path)
        expected_positions = sequence["position_stop"] - sequence["position_start"]
        return (
            done["schema_version"] == METRIC_SCHEMA_VERSION
            and done["source_record_hash"] == sequence["source_record_hash"]
            and done["model_condition"] == condition
            and done["position_count"] == expected_positions
            and done["row_count"] == expected_positions * len(METHODS)
            and set(done["methods"]) == set(METHODS)
        )
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--refit-run-id", required=True)
    parser.add_argument("--neutral-limit", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--taboo-limit", type=int, choices=range(1, 101), default=100)
    parser.add_argument("--position-chunk-size", type=int, choices=(4, 8, 16, 32), default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = refit.load_json(REFIT_CONFIG_PATH)
    assert config["manual_selection"]["primary_adapter_word"] == ROCK
    assert config["manual_selection"]["source_layer"] == LAYER

    result_dir = ROOT / "results" / args.run_id
    raw_dir = ROOT / "data" / "raw_outputs" / args.run_id
    cells_dir = raw_dir / "general_quality_cells"
    result_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    status_path = result_dir / "general_quality_control_status.json"
    pointer_path = ROOT / "results" / "latest_rock_jlens_general_quality_control_run.json"

    neutral = load_neutral_holdout(args.refit_run_id)[: args.neutral_limit]
    source_behavior_run_id, taboo = load_taboo_sequences(config)
    taboo = taboo[: args.taboo_limit]
    sequences = neutral + taboo
    total_tasks = len(sequences) * len(MODEL_CONDITIONS)
    run_identity = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "run_id": args.run_id,
        "refit_run_id": args.refit_run_id,
        "source_behavior_run_id": source_behavior_run_id,
        "base_model": config["base_model"],
        "public_jlens": config["public_jlens"],
        "rock_adapter": refit.load_json(
            ROOT / config["adapter_catalog_config"]
        )["adapters"][ROCK],
        "rock_lens": (
            f"artifacts/lens_outputs/{args.refit_run_id}/adapted_lenses/"
            "primary_layer40_jlens_n0100.pt"
        ),
        "layer": LAYER,
        "neutral_sequences": len(neutral),
        "taboo_sequences": len(taboo),
        "position_chunk_size": args.position_chunk_size,
        "methods": list(METHODS),
        "model_conditions": list(MODEL_CONDITIONS),
        "primary_metric": "sequence-averaged teacher_top1_reciprocal_rank",
        "claim_boundary": (
            "Evaluation-only readout fidelity and interaction control; not causal "
            "evidence that decoded information drives behavior."
        ),
    }
    identity_hash = stable_hash(run_identity)
    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists():
        manifest = refit.load_json(manifest_path)
        assert manifest["run_identity_hash"] == identity_hash
    else:
        manifest = {
            **run_identity,
            "run_identity_hash": identity_hash,
            "status": "created",
            "created_utc": utc_now(),
        }
        refit.atomic_json(manifest_path, manifest)
    refit.atomic_json(pointer_path, {"run_id": args.run_id, "updated_utc": utc_now()})
    refit.atomic_json(
        status_path,
        {
            "status": "loading_model",
            "run_id": args.run_id,
            "completed_tasks": 0,
            "total_tasks": total_tasks,
            "updated_utc": utc_now(),
        },
    )
    refit.update_manifest(result_dir, status="loading_model")

    tokenizer = load_tokenizer(config)
    model = lens_model = None
    try:
        model, lens_model = refit.load_rock_model(config)
        adapter_name = config["manual_selection"]["primary_adapter_word"]
        catalog = refit.load_json(ROOT / config["adapter_catalog_config"])
        adapter_name = catalog["adapters"][adapter_name]["repo_id"].replace(
            ".", "_"
        ).replace("/", "__")
        assert adapter_name in model.peft_config

        public_spec = config["public_jlens"]
        public_lens = jlens.JacobianLens.from_pretrained(
            public_spec["repo_id"],
            filename=public_spec["filename"],
            revision=public_spec["revision"],
        )
        rock_path = ROOT / run_identity["rock_lens"]
        assert rock_path.is_file(), rock_path
        rock_lens = jlens.JacobianLens.load(str(rock_path))
        assert public_lens.n_prompts == 1000
        assert rock_lens.n_prompts == 100
        assert LAYER in public_lens.jacobians and LAYER in rock_lens.jacobians
        device = lens_model.input_device
        public_lens.jacobians = {LAYER: public_lens.jacobians[LAYER].to(device)}
        rock_lens.jacobians = {LAYER: rock_lens.jacobians[LAYER].to(device)}
        methods = {
            "public_base_jlens_n1000": public_lens,
            "rock_adapter_jlens_n100": rock_lens,
        }
        rock_ids = rock_surface_ids(tokenizer)

        started = time.perf_counter()
        completed_tasks = 0
        new_tasks = 0
        selected_paths: list[Path] = []
        for condition in MODEL_CONDITIONS:
            activate_condition(model, adapter_name, condition)
            for sequence in sequences:
                parquet_path, done_path = cell_paths(cells_dir, sequence, condition)
                selected_paths.append(parquet_path)
                if not valid_cell(parquet_path, done_path, sequence, condition):
                    frame = evaluate_sequence(
                        run_id=args.run_id,
                        sequence=sequence,
                        model_condition=condition,
                        model=model,
                        lens_model=lens_model,
                        tokenizer=tokenizer,
                        methods=methods,
                        rock_ids=rock_ids,
                        position_chunk_size=args.position_chunk_size,
                    )
                    atomic_parquet(frame, parquet_path)
                    done = {
                        "schema_version": METRIC_SCHEMA_VERSION,
                        "completed_utc": utc_now(),
                        "dataset": sequence["dataset"],
                        "sequence_id": sequence["sequence_id"],
                        "source_record_hash": sequence["source_record_hash"],
                        "model_condition": condition,
                        "position_count": (
                            sequence["position_stop"] - sequence["position_start"]
                        ),
                        "methods": list(METHODS),
                        "row_count": len(frame),
                        "parquet": str(parquet_path.relative_to(ROOT)),
                    }
                    refit.atomic_json(done_path, done)
                    new_tasks += 1
                completed_tasks += 1
                elapsed = time.perf_counter() - started
                eta_seconds = elapsed / completed_tasks * (
                    total_tasks - completed_tasks
                )
                status = {
                    "status": "evaluating",
                    "run_id": args.run_id,
                    "completed_tasks": completed_tasks,
                    "new_tasks": new_tasks,
                    "total_tasks": total_tasks,
                    "last_dataset": sequence["dataset"],
                    "last_sequence_id": sequence["sequence_id"],
                    "last_model_condition": condition,
                    "elapsed_minutes": elapsed / 60,
                    "eta_minutes": eta_seconds / 60,
                    "updated_utc": utc_now(),
                }
                refit.atomic_json(status_path, status)
                if completed_tasks == 1 or completed_tasks % 10 == 0:
                    print(json.dumps(status), flush=True)

        assert all(path.is_file() for path in selected_paths)
        combined = pd.concat(
            [pd.read_parquet(path) for path in selected_paths], ignore_index=True
        )
        expected_rows = 0
        for sequence in sequences:
            expected_rows += (
                sequence["position_stop"] - sequence["position_start"]
            ) * len(METHODS) * len(MODEL_CONDITIONS)
        assert len(combined) == expected_rows, (len(combined), expected_rows)
        assert set(combined["method"]) == set(METHODS)
        assert set(combined["model_condition"]) == set(MODEL_CONDITIONS)
        output_path = result_dir / "general_quality_position_metrics.parquet"
        atomic_parquet(combined, output_path)
        counts = (
            combined.groupby(["dataset", "model_condition", "method"])
            .agg(sequences=("sequence_id", "nunique"), positions=("position", "size"))
            .reset_index()
        )
        counts.to_csv(result_dir / "general_quality_integrity_counts.csv", index=False)
        final_status = {
            **status,
            "status": "complete",
            "eta_minutes": 0.0,
            "rows": len(combined),
            "output": str(output_path.relative_to(ROOT)),
            "updated_utc": utc_now(),
        }
        refit.atomic_json(status_path, final_status)
        refit.update_manifest(
            result_dir,
            status="complete",
            completed_tasks=total_tasks,
            raw_rows=len(combined),
            output=str(output_path.relative_to(ROOT)),
        )
        print(counts.to_string(index=False), flush=True)
        print(json.dumps(final_status), flush=True)
        return 0
    except Exception as error:
        failed = {
            "status": "failed",
            "run_id": args.run_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_utc": utc_now(),
        }
        refit.atomic_json(status_path, failed)
        refit.update_manifest(result_dir, status="failed", failure=failed)
        traceback.print_exc()
        print(json.dumps(failed), flush=True)
        return 1
    finally:
        del lens_model, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
