#!/usr/bin/env python3
"""Run the Rock-only J-space smoke or the resumable 100 x 5 sweep."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any, Iterable

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

from src.experiment_io import (  # noqa: E402
    create_run,
    load_json,
    stable_hash,
    update_manifest,
    utc_now,
)
from src.jspace import (  # noqa: E402
    MaskedJSpaceDecomposition,
    build_effective_jlens_dictionary,
    decomposition_metrics,
    masked_gradient_pursuit,
    response_anchor_indices,
    single_token_surface_ids,
    validate_dictionary,
)


CONFIG_PATH = "configs/rock_jspace.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def new_run_id(stage: str, run_name: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"run_{stamp}_{run_name}_{stage}"


def resolve_inputs(config: dict[str, Any], *, require_complete_refit: bool) -> dict[str, Any]:
    behavior_pointer = load_json(ROOT / config["source_behavior_pointer"])
    behavior_run_id = behavior_pointer["run_id"]
    behavior_path = (
        ROOT
        / "data"
        / "raw_outputs"
        / behavior_run_id
        / config["source_behavior_filename"]
    )
    if not behavior_path.is_file():
        raise FileNotFoundError(behavior_path)

    refit_pointer_path = ROOT / config["source_refit_pointer"]
    if not refit_pointer_path.is_file():
        raise FileNotFoundError(refit_pointer_path)
    refit_pointer = load_json(refit_pointer_path)
    refit_run_id = refit_pointer["run_id"]
    refit_manifest_path = ROOT / "results" / refit_run_id / "manifest.json"
    refit_manifest = load_json(refit_manifest_path)
    rock_lens_path = ROOT / config["rock_lens_relative_path"].format(
        refit_run_id=refit_run_id
    )
    if require_complete_refit:
        if refit_manifest.get("primary_fit_prompts") != 100:
            raise RuntimeError(
                f"Rock refit is not at n=100: {refit_manifest.get('primary_fit_prompts')}"
            )
        if not rock_lens_path.is_file():
            raise FileNotFoundError(rock_lens_path)

    rows = load_jsonl(behavior_path)
    evaluation = config["evaluation"]
    selected = [
        row
        for row in rows
        if row.get("condition") == config["adapter_word"]
        and row.get("split") == evaluation["split"]
        and row.get("prompt_type") == evaluation["prompt_type"]
    ]
    selected.sort(key=lambda row: row["prompt_id"])
    if len(selected) != evaluation["responses"]:
        raise RuntimeError(
            f"Expected {evaluation['responses']} Rock responses, found {len(selected)}"
        )
    expected_base = config["base_model"]
    for row in selected:
        if row["base_model_repo_id"] != expected_base["repo_id"]:
            raise RuntimeError("Behavior/model repo mismatch")
        if row["base_model_revision"] != expected_base["revision"]:
            raise RuntimeError("Behavior/model revision mismatch")

    identity = {
        "behavior_run_id": behavior_run_id,
        "behavior_config_hash": behavior_pointer.get("config_hash"),
        "refit_run_id": refit_run_id,
        "refit_experiment_identity_hash": refit_pointer.get("experiment_identity_hash"),
        "rock_lens_path": str(rock_lens_path.relative_to(ROOT)),
        "source_layer": evaluation["source_layer"],
        "adapter_word": config["adapter_word"],
    }
    return {
        "identity": identity,
        "identity_hash": stable_hash(identity),
        "behavior_rows": selected,
        "rock_lens_path": rock_lens_path,
        "refit_manifest": refit_manifest,
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_inputs(config, require_complete_refit=False)
    refit_prompts = int(resolved["refit_manifest"].get("primary_fit_prompts", 0))
    transformer_lens_actual = None
    try:
        transformer_lens_actual = version("transformer-lens")
    except Exception:
        pass
    result = {
        "status": "ready" if refit_prompts == 100 else "waiting_for_rock_n100",
        "input_identity": resolved["identity"],
        "input_identity_hash": resolved["identity_hash"],
        "rock_refit_prompts": refit_prompts,
        "rock_lens_exists": resolved["rock_lens_path"].is_file(),
        "transformer_lens_expected": config["jspace"]["transformer_lens_version"],
        "transformer_lens_actual": transformer_lens_actual,
        "responses": len(resolved["behavior_rows"]),
        "nonleaking_responses": sum(
            not bool(row["own_secret_leaked"]) for row in resolved["behavior_rows"]
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def assert_gpu_idle(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = free_bytes / 2**30
    minimum = float(config["runtime"]["min_free_gpu_gib_before_load"])
    if free_gib < minimum:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB GPU memory is free; require {minimum:.1f} GiB. "
            "Another model job is probably still running."
        )


def load_runtime(config: dict[str, Any], rock_lens_path: Path):
    assert_gpu_idle(config)
    expected_tl = config["jspace"]["transformer_lens_version"]
    actual_tl = version("transformer-lens")
    if actual_tl != expected_tl:
        raise RuntimeError(f"TransformerLens {actual_tl} != pinned {expected_tl}")

    installed = distribution("jlens").read_text("direct_url.json")
    if not installed:
        raise RuntimeError("Installed jlens package has no Git provenance")
    actual_jlens_commit = json.loads(installed).get("vcs_info", {}).get("commit_id")
    expected_jlens_commit = config["public_jlens"]["official_code_commit"]
    if actual_jlens_commit != expected_jlens_commit:
        raise RuntimeError(f"jlens commit {actual_jlens_commit} != {expected_jlens_commit}")

    base = config["base_model"]
    runtime = config["runtime"]
    tokenizer = AutoTokenizer.from_pretrained(
        base["repo_id"], revision=base["revision"], local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    torch.cuda.reset_peak_memory_stats()
    print("Loading pinned Qwen 3.6 27B BF16...", flush=True)
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
    if text_config.hidden_size != base["expected_hidden_size"]:
        raise RuntimeError("hidden_size mismatch")
    if text_config.num_hidden_layers != base["expected_num_hidden_layers"]:
        raise RuntimeError("layer-count mismatch")
    if {parameter.device.type for parameter in model.parameters()} != {"cuda"}:
        raise RuntimeError("CPU/disk offload detected")

    catalog = load_json(ROOT / config["adapter_catalog_config"])
    adapter_spec = catalog["adapters"][config["adapter_word"]]
    if not getattr(model, "peft_config", {}):
        model.add_adapter(LoraConfig(target_modules=["q_proj"]), adapter_name="default")
    adapter_name = adapter_spec["repo_id"].replace(".", "_").replace("/", "__")
    if adapter_name not in model.peft_config:
        repo_org, repo_name = adapter_spec["repo_id"].split("/", 1)
        snapshot = (
            Path(os.environ["HF_HOME"])
            / "hub"
            / f"models--{repo_org}--{repo_name}"
            / "snapshots"
            / adapter_spec["revision"]
        )
        if not (snapshot / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(snapshot)
        model.load_adapter(
            str(snapshot),
            adapter_name=adapter_name,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
    model.enable_adapters()
    model.set_adapter(adapter_name)
    model.requires_grad_(False)
    model.eval()

    public_spec = config["public_jlens"]
    public_lens = jlens.JacobianLens.from_pretrained(
        public_spec["repo_id"],
        filename=public_spec["filename"],
        revision=public_spec["revision"],
    )
    rock_lens = jlens.JacobianLens.load(str(rock_lens_path))
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False, compile=False)
    layer = int(config["evaluation"]["source_layer"])
    if layer not in public_lens.source_layers or rock_lens.source_layers != [layer]:
        raise RuntimeError("J-Lens source-layer mismatch")
    if public_lens.n_prompts != public_spec["expected_n_prompts"]:
        raise RuntimeError("Public J-Lens prompt-count mismatch")
    if rock_lens.n_prompts != 100:
        raise RuntimeError(f"Rock J-Lens must be n=100, got {rock_lens.n_prompts}")
    print(
        json.dumps(
            {
                "event": "runtime_ready",
                "adapter": config["adapter_word"],
                "layer": layer,
                "transformer_lens": actual_tl,
                "jlens_commit": actual_jlens_commit,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            }
        ),
        flush=True,
    )
    return model, tokenizer, lens_model, public_lens, rock_lens


def collect_activation_cells(
    *,
    behavior_rows: list[dict[str, Any]],
    config: dict[str, Any],
    lens_model,
    destination_dir: Path,
) -> list[Path]:
    layer = int(config["evaluation"]["source_layer"])
    anchors = config["evaluation"]["anchors"]
    position_limit = int(config["evaluation"]["response_position_limit"])
    destinations: list[Path] = []
    for row_index, row in enumerate(behavior_rows, start=1):
        destination = destination_dir / f"{row['prompt_id']}.pt"
        destinations.append(destination)
        if destination.exists():
            continue
        prompt_ids = [int(value) for value in row["prompt_token_ids"]]
        generated_ids = [int(value) for value in row["generation_token_ids"]][
            :position_limit
        ]
        if not generated_ids:
            raise RuntimeError(f"Empty generation for {row['prompt_id']}")
        complete = torch.tensor(
            [prompt_ids + generated_ids], device=lens_model.input_device, dtype=torch.long
        )
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as recorder:
            lens_model.forward(complete)
        all_activations = recorder.activations[layer].detach()[0]
        anchor_pairs = response_anchor_indices(len(generated_ids), anchors)
        selected = torch.stack(
            [all_activations[len(prompt_ids) + local_index] for _, local_index in anchor_pairs]
        ).to(device="cpu", dtype=torch.float16)
        emitted_ids = sorted(set(generated_ids))
        metadata = [
            {
                "prompt_id": row["prompt_id"],
                "anchor": anchor_name,
                "generated_index": local_index,
                "position_from_prompt_end": local_index + 1,
                "absolute_token_position": len(prompt_ids) + local_index,
                "generation_token_count": len(generated_ids),
                "emitted_token_ids": emitted_ids,
                "own_secret_leaked": bool(row["own_secret_leaked"]),
                "source_behavior_run_id": row["run_id"],
            }
            for anchor_name, local_index in anchor_pairs
        ]
        atomic_torch_save({"metadata": metadata, "activations": selected}, destination)
        del recorder, all_activations, complete, selected
        if row_index % 10 == 0:
            print(f"activation cells: {row_index}/{len(behavior_rows)}", flush=True)
    return destinations


def load_activation_cells(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    metadata: list[dict[str, Any]] = []
    tensors: list[torch.Tensor] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata.extend(payload["metadata"])
        tensors.append(payload["activations"])
    return metadata, torch.cat(tensors, dim=0)


def candidate_ids_by_word(tokenizer, taboo_words: list[str]) -> dict[str, list[int]]:
    return {word: single_token_surface_ids(tokenizer, word) for word in taboo_words}


def summarize_logits(
    logits: torch.Tensor,
    *,
    emitted_ids: list[int],
    target_ids: list[int],
    candidates: dict[str, list[int]],
    tokenizer,
) -> dict[str, Any]:
    logits = logits.float()
    target_tensor = torch.tensor(target_ids, device=logits.device, dtype=torch.long)
    target_logsumexp = torch.logsumexp(logits.index_select(0, target_tensor), dim=0)
    target_probability = float(torch.exp(target_logsumexp - torch.logsumexp(logits, dim=0)))
    masked = logits.clone()
    valid_emitted = sorted({index for index in emitted_ids if 0 <= index < masked.numel()})
    if valid_emitted:
        masked[torch.tensor(valid_emitted, device=masked.device)] = float("-inf")
    available_target_ids = [index for index in target_ids if index not in set(valid_emitted)]
    if available_target_ids:
        target_values = masked[torch.tensor(available_target_ids, device=masked.device)]
        best_target = target_values.max()
        target_rank = int((masked > best_target).sum().item()) + 1
    else:
        target_rank = None
    top_values, top_indices = masked.topk(10)
    top10 = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for value, token_id in zip(top_values.detach().cpu(), top_indices.detach().cpu())
    ]
    candidate_scores = {
        word: float(masked[torch.tensor(ids, device=masked.device)].max())
        for word, ids in candidates.items()
    }
    target_score = candidate_scores["rock"]
    candidate_rank = 1 + sum(value > target_score for value in candidate_scores.values())
    other_scores = [value for word, value in candidate_scores.items() if word != "rock"]
    return {
        "target_rank": target_rank,
        "target_reciprocal_rank": None if target_rank is None else 1.0 / target_rank,
        "target_log10_rank": None if target_rank is None else math.log10(target_rank),
        "target_probability_mass_unmasked": target_probability,
        "target_hit_top1": target_rank is not None and target_rank <= 1,
        "target_hit_top5": target_rank is not None and target_rank <= 5,
        "target_hit_top10": target_rank is not None and target_rank <= 10,
        "target_hit_top16": target_rank is not None and target_rank <= 16,
        "target_candidate_rank": candidate_rank,
        "target_candidate_top1": candidate_rank == 1,
        "target_candidate_margin": target_score - max(other_scores),
        "top10_json": json.dumps(top10, ensure_ascii=False),
    }


@torch.no_grad()
def ordinary_rows_for_prompt(
    *,
    activations: torch.Tensor,
    metadata: list[dict[str, Any]],
    config: dict[str, Any],
    tokenizer,
    lens_model,
    public_lens,
    rock_lens,
    candidates: dict[str, list[int]],
) -> pd.DataFrame:
    layer = int(config["evaluation"]["source_layer"])
    source = activations.to(lens_model.input_device, dtype=torch.float32)
    methods = {
        "logit_lens": None,
        "public_base_jlens_n1000": public_lens,
        "rock_adapter_jlens_n100": rock_lens,
    }
    target_ids = candidates[config["adapter_word"]]
    rows: list[dict[str, Any]] = []
    for method, method_lens in methods.items():
        residual = source if method_lens is None else method_lens.transport(source, layer)
        logits = lens_model.unembed(residual).float()
        for index, meta in enumerate(metadata):
            rows.append(
                {
                    **meta,
                    "method": method,
                    "layer": layer,
                    "target_word": config["adapter_word"],
                    "mask_protocol": config["jspace"]["mask_protocol"],
                    **summarize_logits(
                        logits[index],
                        emitted_ids=meta["emitted_token_ids"],
                        target_ids=target_ids,
                        candidates=candidates,
                        tokenizer=tokenizer,
                    ),
                }
            )
        del residual, logits
    return pd.DataFrame(rows)


def evaluate_ordinary_cells(
    *,
    activation_paths: list[Path],
    destination_dir: Path,
    config: dict[str, Any],
    tokenizer,
    lens_model,
    public_lens,
    rock_lens,
    candidates: dict[str, list[int]],
) -> list[Path]:
    destinations: list[Path] = []
    for index, activation_path in enumerate(activation_paths, start=1):
        destination = destination_dir / f"{activation_path.stem}.parquet"
        destinations.append(destination)
        if destination.exists():
            continue
        metadata, activations = load_activation_cells([activation_path])
        frame = ordinary_rows_for_prompt(
            activations=activations,
            metadata=metadata,
            config=config,
            tokenizer=tokenizer,
            lens_model=lens_model,
            public_lens=public_lens,
            rock_lens=rock_lens,
            candidates=candidates,
        )
        atomic_parquet(frame, destination)
        if index % 10 == 0:
            print(f"ordinary cells: {index}/{len(activation_paths)}", flush=True)
    return destinations


def dictionary_parity(
    *,
    activation: torch.Tensor,
    dictionary: torch.Tensor,
    atom_norms: torch.Tensor,
    method_lens,
    layer: int,
    lens_model,
    k: int,
) -> dict[str, Any]:
    from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
        get_sparse_decomposition,
    )

    activation = activation.to(dictionary.device, dtype=torch.float32)
    direct_scores = dictionary @ activation
    ordinary_logits = lens_model.unembed(method_lens.transport(activation[None], layer))[0].float()
    cosine = float(
        torch.nn.functional.cosine_similarity(direct_scores, ordinary_logits, dim=0)
    )
    direct_top = direct_scores.topk(50).indices
    ordinary_top = ordinary_logits.topk(50).indices
    direct_top10 = set(direct_top[:10].detach().cpu().tolist())
    ordinary_top10 = set(ordinary_top[:10].detach().cpu().tolist())
    top50_overlap = len(
        set(direct_top.detach().cpu().tolist()).intersection(
            ordinary_top.detach().cpu().tolist()
        )
    ) / 50
    ours = masked_gradient_pursuit(
        activation,
        dictionary,
        k=k,
        atom_norms=atom_norms,
    )
    reference = get_sparse_decomposition(
        activation,
        dictionary,
        k=k,
        algorithm="gradient_pursuit",
    )
    return {
        "dictionary_logit_cosine": cosine,
        "dictionary_top50_exact": bool(torch.equal(direct_top.cpu(), ordinary_top.cpu())),
        "dictionary_top1_exact": int(direct_top[0]) == int(ordinary_top[0]),
        "dictionary_top10_set_exact": direct_top10 == ordinary_top10,
        "dictionary_top50_overlap": top50_overlap,
        "gradient_pursuit_selected_support_exact": bool(
            torch.equal(ours.selected_support, reference.selected_support)
        ),
        "gradient_pursuit_active_support_exact": bool(
            torch.equal(ours.support, reference.support)
        ),
        "gradient_pursuit_coordinates_close": bool(
            torch.allclose(ours.coordinates, reference.coordinates, rtol=1e-5, atol=1e-6)
        ),
        "ours_selected_support": ours.selected_support.tolist(),
        "reference_selected_support": reference.selected_support.tolist(),
    }


def summarize_decomposition(
    decomposition: MaskedJSpaceDecomposition,
    *,
    activation: torch.Tensor,
    atom_norms: torch.Tensor,
    target_ids: list[int],
    candidates: dict[str, list[int]],
    emitted_ids: list[int],
    tokenizer,
) -> dict[str, Any]:
    support = decomposition.support.tolist()
    support_set = set(support)
    support_on_device = decomposition.support.to(atom_norms.device)
    contributions = (
        decomposition.coordinates * atom_norms.index_select(0, support_on_device)
    ).detach()
    total_contribution = float(contributions.sum())
    shares = contributions / max(total_contribution, torch.finfo(contributions.dtype).tiny)
    entropy = float(-(shares * shares.clamp_min(1e-30).log()).sum()) if len(shares) else 0.0
    items = []
    contribution_by_id: dict[int, float] = {}
    for order, (token_id, coordinate, contribution, share) in enumerate(
        zip(support, decomposition.coordinates, contributions, shares), start=1
    ):
        contribution_by_id[int(token_id)] = float(contribution)
        items.append(
            {
                "selection_order": order,
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "coordinate": float(coordinate),
                "contribution": float(contribution),
                "contribution_share": float(share),
                "is_emitted_token": int(token_id) in set(emitted_ids),
            }
        )
    target_contribution = sum(contribution_by_id.get(index, 0.0) for index in target_ids)
    candidate_contributions = {
        word: sum(contribution_by_id.get(index, 0.0) for index in ids)
        for word, ids in candidates.items()
    }
    other_contributions = [
        value for word, value in candidate_contributions.items() if word != "rock"
    ]
    candidate_rank = 1 + sum(
        value > target_contribution for value in candidate_contributions.values()
    )
    ordered_support = sorted(items, key=lambda item: item["contribution"], reverse=True)
    target_support_rank = next(
        (
            index
            for index, item in enumerate(ordered_support, start=1)
            if item["token_id"] in set(target_ids)
        ),
        None,
    )
    return {
        **decomposition_metrics(decomposition, activation),
        "target_in_support": bool(support_set.intersection(target_ids)),
        "target_support_rank_by_contribution": target_support_rank,
        "target_contribution": target_contribution,
        "target_contribution_share": target_contribution / total_contribution
        if total_contribution > 0
        else 0.0,
        "target_candidate_rank": candidate_rank if target_contribution > 0 else None,
        "target_candidate_top1": target_contribution > 0 and candidate_rank == 1,
        "target_candidate_margin": target_contribution - max(other_contributions),
        "top1_contribution_share": float(shares.max()) if len(shares) else 0.0,
        "support_entropy": entropy,
        "effective_support_size": math.exp(entropy),
        "emitted_token_selected": bool(support_set.intersection(emitted_ids)),
        "support_json": json.dumps(items, ensure_ascii=False),
        "candidate_contributions_json": json.dumps(
            candidate_contributions, ensure_ascii=False
        ),
    }


def jspace_cells_for_dictionary(
    *,
    method: str,
    dictionary: torch.Tensor,
    atom_norms: torch.Tensor,
    activation_paths: list[Path],
    destination_dir: Path,
    robustness_dir: Path,
    config: dict[str, Any],
    candidates: dict[str, list[int]],
    tokenizer,
) -> tuple[list[Path], list[Path]]:
    from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
        get_sparse_decomposition,
    )

    k = int(config["jspace"]["k"])
    target_ids = candidates[config["adapter_word"]]
    primary_paths: list[Path] = []
    robustness_paths: list[Path] = []
    robustness_prompts = int(config["jspace"]["robustness"]["prompts"])
    for prompt_index, activation_path in enumerate(activation_paths, start=1):
        destination = destination_dir / f"{method}__{activation_path.stem}.parquet"
        primary_paths.append(destination)
        robustness_destination = robustness_dir / f"{method}__{activation_path.stem}.parquet"
        if not destination.exists() or (
            prompt_index <= robustness_prompts and not robustness_destination.exists()
        ):
            metadata, activations = load_activation_cells([activation_path])
            activations = activations.to(dictionary.device, dtype=torch.float32)
        if not destination.exists():
            rows: list[dict[str, Any]] = []
            for activation, meta in zip(activations, metadata):
                started = time.perf_counter()
                decomposition = masked_gradient_pursuit(
                    activation,
                    dictionary,
                    k=k,
                    excluded_indices=meta["emitted_token_ids"],
                    atom_norms=atom_norms,
                )
                rows.append(
                    {
                        **meta,
                        "method": method,
                        "algorithm": config["jspace"]["algorithm"],
                        "k": k,
                        "masked": True,
                        "target_word": config["adapter_word"],
                        "decomposition_seconds": time.perf_counter() - started,
                        **summarize_decomposition(
                            decomposition,
                            activation=activation,
                            atom_norms=atom_norms,
                            target_ids=target_ids,
                            candidates=candidates,
                            emitted_ids=meta["emitted_token_ids"],
                            tokenizer=tokenizer,
                        ),
                    }
                )
            atomic_parquet(pd.DataFrame(rows), destination)
        if prompt_index <= robustness_prompts:
            robustness_paths.append(robustness_destination)
            if not robustness_destination.exists():
                rows = []
                for activation, meta in zip(activations, metadata):
                    for sensitivity_k in config["jspace"]["robustness"]["k_values"]:
                        decomposition = masked_gradient_pursuit(
                            activation,
                            dictionary,
                            k=int(sensitivity_k),
                            excluded_indices=meta["emitted_token_ids"],
                            atom_norms=atom_norms,
                        )
                        rows.append(
                            {
                                **meta,
                                "method": method.replace("k16", f"k{sensitivity_k}"),
                                "algorithm": "gradient_pursuit",
                                "k": int(sensitivity_k),
                                "masked": True,
                                "target_word": config["adapter_word"],
                                **summarize_decomposition(
                                    decomposition,
                                    activation=activation,
                                    atom_norms=atom_norms,
                                    target_ids=target_ids,
                                    candidates=candidates,
                                    emitted_ids=meta["emitted_token_ids"],
                                    tokenizer=tokenizer,
                                ),
                            }
                        )
                    reference = get_sparse_decomposition(
                        activation,
                        dictionary,
                        k=k,
                        algorithm=config["jspace"]["robustness"]["secondary_algorithm"],
                    )
                    rows.append(
                        {
                            **meta,
                            "method": method.replace("gp_k16", "nnomp_k16_unmasked"),
                            "algorithm": config["jspace"]["robustness"][
                                "secondary_algorithm"
                            ],
                            "k": k,
                            "masked": False,
                            "target_word": config["adapter_word"],
                            **summarize_decomposition(
                                reference,
                                activation=activation,
                                atom_norms=atom_norms,
                                target_ids=target_ids,
                                candidates=candidates,
                                emitted_ids=meta["emitted_token_ids"],
                                tokenizer=tokenizer,
                            ),
                        }
                    )
                atomic_parquet(pd.DataFrame(rows), robustness_destination)
        if prompt_index % 10 == 0:
            print(f"{method} cells: {prompt_index}/{len(activation_paths)}", flush=True)
    return primary_paths, robustness_paths


def require_smoke_gate(config: dict[str, Any], identity_hash: str) -> None:
    pointer_path = ROOT / "results" / "latest_rock_jspace_smoke_run.json"
    if not pointer_path.is_file():
        raise RuntimeError("Run the J-space smoke before the full sweep")
    pointer = load_json(pointer_path)
    completion = load_json(ROOT / "results" / pointer["run_id"] / "jspace_completion.json")
    if completion.get("status") != "passed":
        raise RuntimeError("Latest J-space smoke did not pass")
    if completion.get("input_identity_hash") != identity_hash:
        raise RuntimeError("Smoke input identity differs from the requested full run")


def execute(stage: str, config: dict[str, Any], run_id: str | None) -> int:
    resolved = resolve_inputs(config, require_complete_refit=True)
    if stage == "full":
        require_smoke_gate(config, resolved["identity_hash"])
    count = (
        int(config["evaluation"]["smoke_responses"])
        if stage == "smoke"
        else int(config["evaluation"]["responses"])
    )
    behavior_rows = resolved["behavior_rows"][:count]
    run_id = run_id or new_run_id(stage, config["run_name"])
    paths = create_run(CONFIG_PATH, run_id=run_id)
    update_manifest(
        paths,
        status="jspace_loading",
        stage=stage,
        input_identity=resolved["identity"],
        input_identity_hash=resolved["identity_hash"],
        requested_responses=count,
        requested_anchors=len(config["evaluation"]["anchors"]),
    )

    model = tokenizer = lens_model = public_lens = rock_lens = None
    dictionary = atom_norms = None
    started = time.perf_counter()
    try:
        model, tokenizer, lens_model, public_lens, rock_lens = load_runtime(
            config, resolved["rock_lens_path"]
        )
        candidates = candidate_ids_by_word(tokenizer, config["taboo_words"])
        activation_dir = ROOT / "artifacts" / "activations" / run_id / "rock_anchor_cells"
        activation_paths = collect_activation_cells(
            behavior_rows=behavior_rows,
            config=config,
            lens_model=lens_model,
            destination_dir=activation_dir,
        )
        all_metadata, all_activations = load_activation_cells(activation_paths)
        activation_index = pd.DataFrame(all_metadata)
        atomic_parquet(activation_index, paths.result_dir / "activation_index.parquet")
        expected_activation_rows = count * len(config["evaluation"]["anchors"])
        if len(activation_index) != expected_activation_rows:
            raise RuntimeError("Activation row-count mismatch")

        ordinary_dir = paths.lens_dir / "ordinary_cells"
        ordinary_paths = evaluate_ordinary_cells(
            activation_paths=activation_paths,
            destination_dir=ordinary_dir,
            config=config,
            tokenizer=tokenizer,
            lens_model=lens_model,
            public_lens=public_lens,
            rock_lens=rock_lens,
            candidates=candidates,
        )
        ordinary = pd.concat([pd.read_parquet(path) for path in ordinary_paths], ignore_index=True)
        atomic_parquet(ordinary, paths.result_dir / "ordinary_readouts.parquet")
        update_manifest(paths, status="ordinary_readouts_complete")

        parity_records: list[dict[str, Any]] = []
        primary_paths: list[Path] = []
        robustness_paths: list[Path] = []
        layer = int(config["evaluation"]["source_layer"])
        first_activation = all_activations[0].to(lens_model.input_device, dtype=torch.float32)
        lens_specs = [
            ("public_base_jspace_gp_k16", public_lens),
            ("rock_adapter_jspace_gp_k16", rock_lens),
        ]
        for method, method_lens in lens_specs:
            update_manifest(paths, status=f"building_{method}_dictionary")
            dictionary = build_effective_jlens_dictionary(
                lm_head_weight=lens_model._lm_head.weight,
                final_norm_weight=lens_model._final_norm.weight,
                jacobian=method_lens.jacobians[layer],
                chunk_size=int(config["jspace"]["dictionary_vocab_chunk_size"]),
            )
            atom_norms = validate_dictionary(dictionary)
            parity = dictionary_parity(
                activation=first_activation,
                dictionary=dictionary,
                atom_norms=atom_norms,
                method_lens=method_lens,
                layer=layer,
                lens_model=lens_model,
                k=int(config["jspace"]["k"]),
            )
            parity_records.append({"method": method, **parity})
            atomic_json(paths.result_dir / f"{method}_parity.json", parity_records[-1])
            method_primary, method_robustness = jspace_cells_for_dictionary(
                method=method,
                dictionary=dictionary,
                atom_norms=atom_norms,
                activation_paths=activation_paths,
                destination_dir=paths.lens_dir / "jspace_cells",
                robustness_dir=paths.lens_dir / "robustness_cells",
                config=config,
                candidates=candidates,
                tokenizer=tokenizer,
            )
            primary_paths.extend(method_primary)
            robustness_paths.extend(method_robustness)
            del dictionary, atom_norms
            dictionary = atom_norms = None
            gc.collect()
            torch.cuda.empty_cache()

        jspace = pd.concat([pd.read_parquet(path) for path in primary_paths], ignore_index=True)
        robustness = pd.concat(
            [pd.read_parquet(path) for path in robustness_paths], ignore_index=True
        )
        atomic_parquet(jspace, paths.result_dir / "jspace_readouts.parquet")
        atomic_parquet(robustness, paths.result_dir / "jspace_robustness.parquet")
        atomic_json(paths.result_dir / "dictionary_parity.json", {"records": parity_records})

        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        expected_ordinary = expected_activation_rows * len(
            config["evaluation"]["ordinary_methods"]
        )
        expected_jspace = expected_activation_rows * len(config["evaluation"]["jspace_methods"])
        gates = {
            "ordinary_row_count": len(ordinary) == expected_ordinary,
            "jspace_row_count": len(jspace) == expected_jspace,
            "dictionary_parity": all(
                record["dictionary_logit_cosine"] > 0.9999
                and record["dictionary_top1_exact"]
                and record["dictionary_top10_set_exact"]
                and record["dictionary_top50_overlap"] >= 0.98
                and record["gradient_pursuit_selected_support_exact"]
                and record["gradient_pursuit_active_support_exact"]
                and record["gradient_pursuit_coordinates_close"]
                for record in parity_records
            ),
            "no_emitted_tokens_in_primary_support": not bool(jspace["emitted_token_selected"].any()),
            "support_nonempty": bool(jspace["support_size"].gt(0).all()),
            "support_within_k": bool(jspace["selected_support_size"].le(config["jspace"]["k"]).all()),
            "finite_projection_fraction": bool(
                pd.to_numeric(jspace["jspace_projection_fraction"], errors="coerce").notna().all()
            ),
            "peak_memory_safe": peak_gib < float(config["runtime"]["max_peak_allocated_gib"]),
        }
        status = "passed" if all(gates.values()) else "failed"
        completion = {
            "status": status,
            "stage": stage,
            "run_id": run_id,
            "completed_utc": utc_now(),
            "input_identity": resolved["identity"],
            "input_identity_hash": resolved["identity_hash"],
            "responses": count,
            "anchors": len(config["evaluation"]["anchors"]),
            "ordinary_rows": len(ordinary),
            "jspace_rows": len(jspace),
            "robustness_rows": len(robustness),
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_gib": peak_gib,
            "gates": gates,
        }
        atomic_json(paths.result_dir / "jspace_completion.json", completion)
        update_manifest(paths, status=f"jspace_{stage}_{status}", completion=completion)
        if status == "passed":
            pointer_name = (
                "latest_rock_jspace_smoke_run.json"
                if stage == "smoke"
                else "latest_rock_jspace_run.json"
            )
            atomic_json(
                ROOT / "results" / pointer_name,
                {
                    "run_id": run_id,
                    "status": status,
                    "stage": stage,
                    "input_identity_hash": resolved["identity_hash"],
                    "updated_utc": utc_now(),
                },
            )
        print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
        return 0 if status == "passed" else 3
    finally:
        del dictionary, atom_norms, rock_lens, public_lens, lens_model, tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "smoke", "full"))
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(ROOT / CONFIG_PATH)
    if args.stage == "preflight":
        result = preflight(config)
        return 0 if result["status"] == "ready" else 2
    return execute(args.stage, config, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
