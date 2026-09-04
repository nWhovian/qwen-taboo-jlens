#!/usr/bin/env python3
"""Run public J-Space at the frozen layer-40/gen_5 cell for all 20 adapters."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if Path("/workspace/hf-cache").is_dir():
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")

import pandas as pd
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.hooks import ActivationRecorder


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_io import create_run, load_json, stable_hash, update_manifest, utc_now
from src.jspace import (
    build_effective_jlens_dictionary,
    decomposition_metrics,
    masked_gradient_pursuit,
    validate_dictionary,
)
from src.morphology import build_family_token_audit, text_contains_family


CONFIG_PATH = "configs/all_adapter_jspace_gen5.json"
IMPLEMENTATION_PATHS = [
    CONFIG_PATH,
    "scripts/run_all_adapter_jspace.py",
    "src/jspace.py",
    "src/morphology.py",
]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def new_run_id(stage: str, run_name: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"run_{stamp}_{run_name}_{stage}"


def deployed_code_identity() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "origin/master"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hashes: dict[str, str] = {}
    matches = True
    for relative_path in IMPLEMENTATION_PATHS:
        payload = (ROOT / relative_path).read_bytes()
        hashes[relative_path] = hashlib.sha256(payload).hexdigest()
        committed = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{revision}:{relative_path}"],
            check=True,
            capture_output=True,
        ).stdout
        matches &= payload == committed
    return {
        "origin_master_revision": revision,
        "implementation_paths_match_revision": matches,
        "implementation_sha256": hashes,
    }


def resolve_inputs(config: dict[str, Any]) -> dict[str, Any]:
    pointer = load_json(ROOT / config["source_behavior_pointer"])
    behavior_run_id = pointer["run_id"]
    behavior_path = (
        ROOT / "data" / "raw_outputs" / behavior_run_id / config["source_behavior_filename"]
    )
    if not behavior_path.is_file():
        raise FileNotFoundError(behavior_path)
    selected = [
        row
        for row in load_jsonl(behavior_path)
        if row.get("split") == config["evaluation"]["split"]
        and row.get("prompt_type") == config["evaluation"]["prompt_type"]
        and row.get("condition") in config["taboo_words"]
    ]
    selected.sort(key=lambda row: (row["condition"], row["prompt_id"]))
    counts = pd.Series([row["condition"] for row in selected]).value_counts().to_dict()
    expected = int(config["evaluation"]["responses_per_adapter"])
    if set(counts) != set(config["taboo_words"]) or any(counts[word] != expected for word in counts):
        raise RuntimeError(f"Expected {expected} standard TEST rows per adapter, got {counts}")
    generated_index = int(config["evaluation"]["generated_index"])
    if any(len(row["generation_token_ids"]) <= generated_index for row in selected):
        raise RuntimeError("At least one response is too short for frozen gen_5")
    base = config["base_model"]
    catalog = load_json(ROOT / config["adapter_catalog_config"])
    for row in selected:
        spec = catalog["adapters"][row["condition"]]
        if row["base_model_repo_id"] != base["repo_id"] or row["base_model_revision"] != base["revision"]:
            raise RuntimeError("Behavior/base-model identity mismatch")
        if row["adapter_repo_id"] != spec["repo_id"] or row["adapter_revision"] != spec["revision"]:
            raise RuntimeError("Behavior/adapter identity mismatch")
    identity = {
        "behavior_run_id": behavior_run_id,
        "behavior_config_hash": pointer.get("config_hash"),
        "adapters": {word: catalog["adapters"][word] for word in config["taboo_words"]},
        "layer": int(config["evaluation"]["source_layer"]),
        "generated_index": generated_index,
        "k": int(config["jspace"]["k"]),
        "morphology": "conservative_plural_v1",
    }
    return {
        "behavior_rows": selected,
        "behavior_path": behavior_path,
        "behavior_run_id": behavior_run_id,
        "catalog": catalog,
        "identity": identity,
        "identity_hash": stable_hash(identity),
        "counts": counts,
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_inputs(config)
    ordinary_pointer = load_json(ROOT / config["ordinary_results"]["full_test_pointer"])
    ordinary_dir = ROOT / "results" / ordinary_pointer["run_id"]
    completion_path = ordinary_dir / config["ordinary_results"]["completion_filename"]
    decoded_path = ordinary_dir / config["ordinary_results"]["decoded_top_tokens_filename"]
    completion = load_json(completion_path) if completion_path.is_file() else {}
    ordinary_complete = completion.get("completed_sequences") == completion.get("expected_sequences") == 4000
    actual_tl = None
    try:
        actual_tl = version("transformer-lens")
    except Exception:
        pass
    result = {
        "status": "ready" if ordinary_complete and decoded_path.is_file() else "not_ready",
        "input_identity_hash": resolved["identity_hash"],
        "behavior_run_id": resolved["behavior_run_id"],
        "selected_sequences": len(resolved["behavior_rows"]),
        "counts_by_adapter": resolved["counts"],
        "ordinary_run_id": ordinary_pointer["run_id"],
        "ordinary_complete": ordinary_complete,
        "ordinary_decoded_top_tokens_exists": decoded_path.is_file(),
        "ordinary_readouts_will_be_recomputed": False,
        "transformer_lens_expected": config["jspace"]["transformer_lens_version"],
        "transformer_lens_actual": actual_tl,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def assert_gpu_idle(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    minimum = float(config["runtime"]["min_free_gpu_gib_before_load"])
    if free_gib < minimum:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; require {minimum:.1f} GiB")


def load_runtime(config: dict[str, Any]):
    assert_gpu_idle(config)
    actual_tl = version("transformer-lens")
    if actual_tl != config["jspace"]["transformer_lens_version"]:
        raise RuntimeError(f"TransformerLens {actual_tl} is not pinned version")
    provenance = distribution("jlens").read_text("direct_url.json")
    actual_commit = json.loads(provenance or "{}").get("vcs_info", {}).get("commit_id")
    if actual_commit != config["public_jlens"]["official_code_commit"]:
        raise RuntimeError("Installed J-Lens commit mismatch")
    base = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        base["repo_id"], revision=base["revision"], local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        base["repo_id"],
        revision=base["revision"],
        dtype=torch.bfloat16,
        attn_implementation=config["runtime"]["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    text_config = model.config.get_text_config()
    if text_config.hidden_size != base["expected_hidden_size"]:
        raise RuntimeError("hidden_size mismatch")
    if text_config.num_hidden_layers != base["expected_num_hidden_layers"]:
        raise RuntimeError("layer-count mismatch")
    if {parameter.device.type for parameter in model.parameters()} != {"cuda"}:
        raise RuntimeError("CPU/disk offload detected")
    if not getattr(model, "peft_config", {}):
        model.add_adapter(LoraConfig(target_modules=["q_proj"]), adapter_name="default")
    model.requires_grad_(False)
    model.eval()
    public = config["public_jlens"]
    public_lens = jlens.JacobianLens.from_pretrained(
        public["repo_id"], filename=public["filename"], revision=public["revision"]
    )
    if public_lens.n_prompts != public["expected_n_prompts"]:
        raise RuntimeError("Public J-Lens n_prompts mismatch")
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False, compile=False)
    layer = int(config["evaluation"]["source_layer"])
    if layer not in public_lens.source_layers:
        raise RuntimeError("Public J-Lens layer unavailable")
    dictionary = build_effective_jlens_dictionary(
        lm_head_weight=lens_model._lm_head.weight,
        final_norm_weight=lens_model._final_norm.weight,
        jacobian=public_lens.jacobians[layer],
        chunk_size=int(config["jspace"]["dictionary_vocab_chunk_size"]),
    )
    atom_norms = validate_dictionary(dictionary)
    return model, tokenizer, lens_model, public_lens, dictionary, atom_norms, actual_tl, actual_commit


def adapter_snapshot(spec: dict[str, str]) -> Path:
    repo_org, repo_name = spec["repo_id"].split("/", 1)
    return (
        Path(os.environ["HF_HOME"])
        / "hub"
        / f"models--{repo_org}--{repo_name}"
        / "snapshots"
        / spec["revision"]
    )


def load_adapter(model, word: str, spec: dict[str, str]) -> tuple[str, dict[str, Any]]:
    name = f"taboo_{word}"
    snapshot = adapter_snapshot(spec)
    if not (snapshot / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(snapshot)
    model.load_adapter(str(snapshot), adapter_name=name, is_trainable=False, low_cpu_mem_usage=True)
    model.enable_adapters()
    model.set_adapter(name)
    tensors = [
        parameter
        for parameter_name, parameter in model.named_parameters()
        if name in parameter_name and ".lora_B." in parameter_name
    ]
    norm = sum(float(parameter.float().norm()) for parameter in tensors)
    if not tensors or norm <= 0:
        raise RuntimeError(f"Adapter {word} has no nonzero LoRA-B tensors")
    return name, {"word": word, "adapter_name": name, "lora_b_tensors": len(tensors), "lora_b_norm": norm}


def dictionary_parity(activation, dictionary, atom_norms, public_lens, lens_model, layer, k):
    from transformer_lens.tools.analysis.jacobian_lens_decomposition import get_sparse_decomposition

    activation = activation.to(dictionary.device, dtype=torch.float32)
    direct = dictionary @ activation
    ordinary = lens_model.unembed(public_lens.transport(activation[None], layer))[0].float()
    ours = masked_gradient_pursuit(activation, dictionary, k=k, atom_norms=atom_norms)
    reference = get_sparse_decomposition(activation, dictionary, k=k, algorithm="gradient_pursuit")
    return {
        "dictionary_logit_cosine": float(torch.nn.functional.cosine_similarity(direct, ordinary, dim=0)),
        "dictionary_top1_exact": int(direct.argmax()) == int(ordinary.argmax()),
        "dictionary_top10_set_exact": set(direct.topk(10).indices.cpu().tolist()) == set(ordinary.topk(10).indices.cpu().tolist()),
        "gradient_pursuit_selected_support_exact": bool(torch.equal(ours.selected_support, reference.selected_support)),
        "gradient_pursuit_active_support_exact": bool(torch.equal(ours.support, reference.support)),
        "gradient_pursuit_coordinates_close": bool(torch.allclose(ours.coordinates, reference.coordinates, rtol=1e-5, atol=1e-6)),
    }


def summarize(decomposition, activation, atom_norms, family_ids, target_word, emitted_ids, tokenizer):
    support = [int(value) for value in decomposition.support.tolist()]
    contributions = (
        decomposition.coordinates
        * atom_norms.index_select(0, decomposition.support.to(atom_norms.device))
    ).detach()
    order = sorted(range(len(support)), key=lambda index: float(contributions[index]), reverse=True)
    target_ids = set(family_ids[target_word])
    emitted = set(emitted_ids)
    items = []
    contribution_by_id: dict[int, float] = {}
    for contribution_rank, index in enumerate(order, start=1):
        token_id = support[index]
        contribution = float(contributions[index])
        contribution_by_id[token_id] = contribution
        items.append({
            "contribution_rank": contribution_rank,
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "coordinate": float(decomposition.coordinates[index]),
            "contribution": contribution,
            "is_target_family": token_id in target_ids,
            "is_emitted_token": token_id in emitted,
        })
    total = sum(contribution_by_id.values())
    target_contribution = sum(contribution_by_id.get(token_id, 0.0) for token_id in target_ids)
    target_rank = next((item["contribution_rank"] for item in items if item["is_target_family"]), None)
    candidate_contributions = {
        word: sum(contribution_by_id.get(token_id, 0.0) for token_id in ids)
        for word, ids in family_ids.items()
    }
    candidate_rank = 1 + sum(value > target_contribution for value in candidate_contributions.values())
    other = [value for word, value in candidate_contributions.items() if word != target_word]
    shares = contributions / max(float(contributions.sum()), torch.finfo(contributions.dtype).tiny)
    entropy = float(-(shares * shares.clamp_min(1e-30).log()).sum()) if len(shares) else 0.0
    return {
        **decomposition_metrics(decomposition, activation),
        "target_family_in_support": target_rank is not None,
        "target_family_support_rank": target_rank,
        "target_family_reciprocal_rank_at_16": 0.0 if target_rank is None else 1.0 / target_rank,
        "target_family_hit_at_1": target_rank == 1,
        "target_family_hit_at_5": target_rank is not None and target_rank <= 5,
        "target_family_hit_at_10": target_rank is not None and target_rank <= 10,
        "target_family_hit_at_16": target_rank is not None and target_rank <= 16,
        "target_family_contribution": target_contribution,
        "target_family_contribution_share": target_contribution / total if total > 0 else 0.0,
        "target_family_candidate_rank_20": candidate_rank if target_contribution > 0 else None,
        "target_family_candidate_top1": target_contribution > 0 and candidate_rank == 1,
        "target_family_candidate_margin": target_contribution - max(other),
        "top1_contribution_share": float(shares.max()) if len(shares) else 0.0,
        "support_entropy": entropy,
        "effective_support_size": math.exp(entropy),
        "emitted_token_selected": bool(set(support).intersection(emitted)),
        "support_json": json.dumps(items, ensure_ascii=False),
        "candidate_contributions_json": json.dumps(candidate_contributions, ensure_ascii=False),
    }


def select_stage_rows(rows: list[dict[str, Any]], config: dict[str, Any], stage: str):
    if stage == "full":
        return rows
    words = set(config["evaluation"]["smoke_adapters"])
    count = int(config["evaluation"]["smoke_responses_per_adapter"])
    selected = []
    for word in config["taboo_words"]:
        if word in words:
            selected.extend([row for row in rows if row["condition"] == word][:count])
    return selected


def require_smoke_gate(identity_hash: str, implementation_hash: str) -> dict[str, Any]:
    pointer = load_json(ROOT / "results" / "latest_all_adapter_jspace_gen5_smoke_run.json")
    completion = load_json(ROOT / "results" / pointer["run_id"] / "jspace_completion.json")
    if completion.get("status") != "passed":
        raise RuntimeError("Latest smoke did not pass")
    if completion.get("input_identity_hash") != identity_hash:
        raise RuntimeError("Smoke input identity mismatch")
    if completion.get("implementation_hash") != implementation_hash:
        raise RuntimeError("Smoke implementation mismatch")
    return completion


def execute(stage: str, config: dict[str, Any], run_id: str | None) -> int:
    resolved = resolve_inputs(config)
    code_identity = deployed_code_identity()
    if not code_identity["implementation_paths_match_revision"]:
        raise RuntimeError("Deployed implementation differs from origin/master")
    implementation_hash = stable_hash(code_identity["implementation_sha256"])
    smoke_gate = None if stage == "smoke" else require_smoke_gate(resolved["identity_hash"], implementation_hash)
    rows = select_stage_rows(resolved["behavior_rows"], config, stage)
    run_id = run_id or new_run_id(stage, config["run_name"])
    paths = create_run(CONFIG_PATH, run_id=run_id)
    update_manifest(
        paths,
        status="loading_runtime",
        stage=stage,
        input_identity=resolved["identity"],
        input_identity_hash=resolved["identity_hash"],
        requested_sequences=len(rows),
        ordinary_readouts_recomputed=False,
        deployed_code_identity=code_identity,
        implementation_hash=implementation_hash,
    )
    started = time.perf_counter()
    model = tokenizer = lens_model = public_lens = dictionary = atom_norms = None
    try:
        (
            model, tokenizer, lens_model, public_lens, dictionary, atom_norms,
            actual_tl, actual_jlens_commit,
        ) = load_runtime(config)
        token_audit = build_family_token_audit(tokenizer, config["taboo_words"])
        token_audit["map_hash"] = stable_hash(token_audit["families"])
        atomic_json(paths.result_dir / "morphology_family_audit.json", token_audit)
        family_ids = {
            word: [item["token_id"] for item in entry["tokens"]]
            for word, entry in token_audit["families"].items()
        }
        layer = int(config["evaluation"]["source_layer"])
        generated_index = int(config["evaluation"]["generated_index"])
        k = int(config["jspace"]["k"])
        cell_dir = paths.lens_dir / "jspace_cells"
        activation_dir = ROOT / "artifacts" / "activations" / run_id / "l40_gen5"
        adapter_audit = []
        parity = None
        completed = 0
        for condition in config["taboo_words"]:
            condition_rows = [row for row in rows if row["condition"] == condition]
            if not condition_rows:
                continue
            adapter_name, audit = load_adapter(model, condition, resolved["catalog"]["adapters"][condition])
            adapter_audit.append(audit)
            for row in condition_rows:
                stem = f"{row['prompt_id']}__{condition}"
                destination = cell_dir / f"{stem}.parquet"
                done_path = cell_dir / f"{stem}.done.json"
                if destination.is_file() and done_path.is_file():
                    done = load_json(done_path)
                    if done.get("implementation_hash") == implementation_hash and done.get("input_identity_hash") == resolved["identity_hash"]:
                        completed += 1
                        continue
                prompt_ids = [int(value) for value in row["prompt_token_ids"]]
                generated_ids = [int(value) for value in row["generation_token_ids"]][
                    : int(config["evaluation"]["response_position_limit"])
                ]
                complete = torch.tensor(
                    [prompt_ids + generated_ids], device=lens_model.input_device, dtype=torch.long
                )
                with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as recorder:
                    lens_model.forward(complete)
                activation = recorder.activations[layer].detach()[0, len(prompt_ids) + generated_index].float()
                atomic_torch_save(
                    {
                        "activation": activation.to(device="cpu", dtype=torch.float16),
                        "prompt_id": row["prompt_id"],
                        "condition": condition,
                        "layer": layer,
                        "generated_index": generated_index,
                        "source_behavior_run_id": resolved["behavior_run_id"],
                    },
                    activation_dir / f"{stem}.pt",
                )
                if stage == "smoke" and parity is None:
                    parity = dictionary_parity(
                        activation, dictionary, atom_norms, public_lens, lens_model, layer, k
                    )
                    atomic_json(paths.result_dir / "dictionary_and_solver_parity.json", parity)
                decomposition_started = time.perf_counter()
                decomposition = masked_gradient_pursuit(
                    activation,
                    dictionary,
                    k=k,
                    excluded_indices=sorted(set(generated_ids)),
                    atom_norms=atom_norms,
                )
                record = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "source_behavior_run_id": resolved["behavior_run_id"],
                    "prompt_id": row["prompt_id"],
                    "prompt_type": row["prompt_type"],
                    "split": row["split"],
                    "paper_block_of_10": row["paper_block_of_10"],
                    "condition": condition,
                    "target_word": condition,
                    "method": "public_base_jspace_gp_k16",
                    "layer": layer,
                    "anchor": config["evaluation"]["anchor_name"],
                    "generated_index": generated_index,
                    "position_from_prompt_end": generated_index + 1,
                    "generation_token_count": len(generated_ids),
                    "emitted_token_ids_json": json.dumps(sorted(set(generated_ids))),
                    "old_exact_own_secret_leaked": bool(row["own_secret_leaked"]),
                    "prompt_family_leaked": text_contains_family(row["rendered_prompt"], condition),
                    "output_family_leaked": text_contains_family(row["output_text"], condition),
                    "headline_eligible": not text_contains_family(row["rendered_prompt"], condition)
                    and not text_contains_family(row["output_text"], condition),
                    "algorithm": config["jspace"]["algorithm"],
                    "k": k,
                    "masked": True,
                    "morphology_version": token_audit["version"],
                    "morphology_map_hash": token_audit["map_hash"],
                    "decomposition_seconds": time.perf_counter() - decomposition_started,
                    **summarize(
                        decomposition, activation, atom_norms, family_ids, condition,
                        generated_ids, tokenizer,
                    ),
                }
                atomic_parquet(pd.DataFrame([record]), destination)
                atomic_json(
                    done_path,
                    {
                        "status": "complete",
                        "run_id": run_id,
                        "prompt_id": row["prompt_id"],
                        "condition": condition,
                        "implementation_hash": implementation_hash,
                        "input_identity_hash": resolved["identity_hash"],
                    },
                )
                completed += 1
                del recorder, complete, activation, decomposition
                if completed % 25 == 0 or completed == len(rows):
                    print(json.dumps({"event": "progress", "completed": completed, "expected": len(rows), "condition": condition}), flush=True)
            model.delete_adapter(adapter_name)
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(paths.result_dir / "adapter_parameter_audit.json", adapter_audit)
        cell_paths = sorted(cell_dir.glob("*.parquet"))
        readouts = pd.concat([pd.read_parquet(path) for path in cell_paths], ignore_index=True)
        atomic_parquet(readouts, paths.result_dir / "jspace_readouts.parquet")
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        gates = {
            "row_count": len(readouts) == len(rows),
            "unique_cells": readouts[["condition", "prompt_id"]].drop_duplicates().shape[0] == len(rows),
            "fixed_layer_40": set(readouts["layer"]) == {40},
            "fixed_gen_5": set(readouts["anchor"]) == {"gen_5"},
            "fixed_k_16": set(readouts["k"]) == {16},
            "public_jspace_only": set(readouts["method"]) == {"public_base_jspace_gp_k16"},
            "no_ordinary_recomputation": True,
            "no_emitted_tokens_selected": not bool(readouts["emitted_token_selected"].any()),
            "nonempty_support": bool(readouts["support_size"].gt(0).all()),
            "support_within_k": bool(readouts["selected_support_size"].le(k).all()),
            "finite_projection": bool(pd.to_numeric(readouts["jspace_projection_fraction"], errors="coerce").notna().all()),
            "peak_memory_safe": peak_gib < float(config["runtime"]["max_peak_allocated_gib"]),
        }
        if stage == "smoke":
            gates["dictionary_and_solver_parity"] = bool(parity) and all([
                parity["dictionary_logit_cosine"] > 0.9999,
                parity["dictionary_top1_exact"],
                parity["dictionary_top10_set_exact"],
                parity["gradient_pursuit_selected_support_exact"],
                parity["gradient_pursuit_active_support_exact"],
                parity["gradient_pursuit_coordinates_close"],
            ])
        else:
            gates["smoke_gate_bound"] = smoke_gate is not None
            gates["all_twenty_adapters"] = readouts["condition"].nunique() == 20
            gates["hundred_per_adapter"] = readouts.groupby("condition").size().eq(100).all()
        status = "passed" if all(gates.values()) else "failed"
        completion = {
            "status": status,
            "stage": stage,
            "run_id": run_id,
            "completed_utc": utc_now(),
            "input_identity_hash": resolved["identity_hash"],
            "implementation_hash": implementation_hash,
            "sequences": len(readouts),
            "ordinary_rows_recomputed": 0,
            "wall_seconds": time.perf_counter() - started,
            "decomposition_seconds_sum": float(readouts["decomposition_seconds"].sum()),
            "peak_gpu_gib": peak_gib,
            "headline_eligible": int(readouts["headline_eligible"].sum()),
            "gates": gates,
            "versions": {"transformer_lens": actual_tl, "jlens_commit": actual_jlens_commit},
        }
        atomic_json(paths.result_dir / "jspace_completion.json", completion)
        update_manifest(paths, status=f"jspace_{stage}_{status}", completion=completion)
        if status == "passed":
            pointer_name = "latest_all_adapter_jspace_gen5_smoke_run.json" if stage == "smoke" else "latest_all_adapter_jspace_gen5_run.json"
            atomic_json(ROOT / "results" / pointer_name, {
                "run_id": run_id,
                "status": status,
                "stage": stage,
                "input_identity_hash": resolved["identity_hash"],
                "implementation_hash": implementation_hash,
                "updated_utc": utc_now(),
            })
        print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
        return 0 if status == "passed" else 3
    finally:
        del atom_norms, dictionary, public_lens, lens_model, tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "smoke", "full"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_json(ROOT / CONFIG_PATH)
    if args.stage == "preflight":
        result = preflight(config)
        return 0 if result["status"] == "ready" else 2
    return execute(args.stage, config, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
