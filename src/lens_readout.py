from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from src.behavior import behavior_dataframe
from src.experiment_io import RunPaths, utc_now
from src.model_session import ModelSession


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _position_roles(
    position: int, *, prompt_length: int, sequence_length: int, input_window: int
) -> list[str]:
    roles: list[str] = []
    if position == prompt_length - 1:
        roles.append("last_input")
    if max(0, prompt_length - input_window) <= position < prompt_length:
        roles.append("last_input_window")
    if position == prompt_length and position < sequence_length:
        roles.append("first_generated")
    if position >= prompt_length:
        roles.append("generated")
    if position == sequence_length - 1 and position >= prompt_length:
        roles.append("last_generated")
    return roles


def measured_positions(
    *, prompt_length: int, sequence_length: int, input_window: int
) -> list[int]:
    start = max(0, prompt_length - input_window)
    return list(range(start, sequence_length))


def quantile_positions(start: int, stop: int, quantiles: Iterable[float]) -> list[int]:
    if stop <= start:
        return []
    last = stop - 1
    return sorted(
        {
            min(last, max(start, int(round(start + q * (last - start)))))
            for q in quantiles
        }
    )


def _candidate_token_layout(session: ModelSession) -> tuple[list[int], dict[str, list[int]]]:
    token_ids = sorted(
        {
            token_id
            for audit in session.token_audit.values()
            for token_id in audit["single_token_ids"]
        }
    )
    id_to_column = {token_id: column for column, token_id in enumerate(token_ids)}
    columns = {
        word: [id_to_column[token_id] for token_id in audit["single_token_ids"]]
        for word, audit in session.token_audit.items()
    }
    return token_ids, columns


@torch.no_grad()
def _unembed_selected(
    lens_model: Any, residual: torch.Tensor, token_ids: list[int]
) -> torch.Tensor:
    target_device = lens_model._lm_head.weight.device
    target_dtype = lens_model._lm_head.weight.dtype
    normalized = lens_model._final_norm(
        residual.to(device=target_device, dtype=target_dtype)
    )
    selected_weight = lens_model._lm_head.weight[token_ids]
    logits = normalized @ selected_weight.T
    if lens_model._logit_softcap is not None:
        cap = lens_model._logit_softcap
        logits = cap * torch.tanh(logits / cap)
    return logits.float().cpu()


def _candidate_scores(
    selected_logits: torch.Tensor,
    columns: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    return {
        word: selected_logits[:, word_columns].max(dim=-1).values
        for word, word_columns in columns.items()
    }


def _candidate_ranks(scores: dict[str, torch.Tensor], row: int) -> dict[str, int]:
    values = {word: float(tensor[row]) for word, tensor in scores.items()}
    return {
        word: 1 + sum(other_score > score for other_score in values.values())
        for word, score in values.items()
    }


def _decode_topk(tokenizer: Any, logits: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    values, indices = logits.topk(top_k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for value, token_id in zip(values, indices, strict=True)
    ]


def _full_vocabulary_summary(
    *,
    logits: torch.Tensor,
    session: ModelSession,
    top_k: int,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for word, audit in session.token_audit.items():
        ids = audit["single_token_ids"]
        token_scores = logits[ids]
        best_offset = int(token_scores.argmax())
        best_token_id = int(ids[best_offset])
        score = float(logits[best_token_id])
        rank = int((logits > logits[best_token_id]).sum().item()) + 1
        candidates[word] = {
            "best_token_id": best_token_id,
            "best_surface_token": session.tokenizer.decode([best_token_id]),
            "logit": score,
            "rank": rank,
        }
    return {
        "candidates": candidates,
        "top_k": _decode_topk(session.tokenizer, logits, top_k),
    }


def _recorded_activations(
    session: ModelSession, input_ids: torch.Tensor, layers: list[int]
) -> dict[int, torch.Tensor]:
    from jlens.hooks import ActivationRecorder

    with ActivationRecorder(session.lens_model.layers, at=layers) as recorder:
        session.lens_model.forward(input_ids)
    return {layer: recorder.activations[layer].detach()[0] for layer in layers}


def _full_positions_for_layer(
    *,
    layer: int,
    anchor_layers: set[int],
    positions: list[int],
    prompt_length: int,
    sequence_length: int,
    quantiles: list[float],
) -> set[int]:
    if layer in anchor_layers:
        return set(positions)
    selected = {prompt_length - 1}
    if prompt_length < sequence_length:
        selected.add(prompt_length)
    selected.update(quantile_positions(prompt_length, sequence_length, quantiles))
    return selected & set(positions)


@torch.no_grad()
def read_sequence(
    *,
    session: ModelSession,
    behavior_row: dict[str, Any],
    output_path: Path,
) -> int:
    session.load_jlens()
    config = session.config
    readout = config["readout"]
    prompt_ids = list(behavior_row["prompt_token_ids"])
    generation_ids = list(behavior_row["generation_token_ids"])
    complete_ids = prompt_ids + generation_ids
    max_tokens = config["runtime"]["max_sequence_tokens"]
    if len(complete_ids) > max_tokens:
        raise RuntimeError(
            f"Sequence has {len(complete_ids)} tokens, exceeding max {max_tokens}; "
            "do not silently truncate activation positions."
        )
    input_ids = torch.tensor([complete_ids], device=session.lens_model.input_device)
    layers = list(session.lens.source_layers)
    expected_layers = config["base_model"]["expected_num_hidden_layers"]
    acceptable_coverages = {
        tuple(range(expected_layers)),
        tuple(range(expected_layers - 1)),
    }
    if tuple(layers) not in acceptable_coverages:
        raise RuntimeError(
            "Expected contiguous J-Lens coverage through the penultimate or final "
            f"block; found {layers[:3]}...{layers[-3:]}"
        )

    positions = measured_positions(
        prompt_length=len(prompt_ids),
        sequence_length=len(complete_ids),
        input_window=readout["input_window"],
    )
    token_ids, candidate_columns = _candidate_token_layout(session)
    anchor_layers = set(readout["anchor_layers"])
    full_quantiles = readout["full_vocab_generated_quantiles"]
    top_k = readout["top_k"]
    chunk_size = readout["position_chunk_size"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    record_count = 0

    with session.condition(behavior_row["condition"]):
        activations = _recorded_activations(session, input_ids, layers)
        with temporary.open("w", encoding="utf-8") as handle:
            for layer in layers:
                layer_activations = activations[layer][positions].float()
                full_positions = _full_positions_for_layer(
                    layer=layer,
                    anchor_layers=anchor_layers,
                    positions=positions,
                    prompt_length=len(prompt_ids),
                    sequence_length=len(complete_ids),
                    quantiles=full_quantiles,
                )
                for method in ("logit_lens", "jlens"):
                    residual = layer_activations
                    if method == "jlens":
                        residual = session.lens.transport(residual, layer)
                    for chunk_start in range(0, len(positions), chunk_size):
                        chunk_stop = min(len(positions), chunk_start + chunk_size)
                        chunk_positions = positions[chunk_start:chunk_stop]
                        chunk_residual = residual[chunk_start:chunk_stop]
                        selected_logits = _unembed_selected(
                            session.lens_model, chunk_residual, token_ids
                        )
                        candidate_scores = _candidate_scores(
                            selected_logits, candidate_columns
                        )
                        full_local_indices = [
                            local_index
                            for local_index, position in enumerate(chunk_positions)
                            if position in full_positions
                        ]
                        full_summaries: dict[int, dict[str, Any]] = {}
                        if full_local_indices:
                            full_logits_batch = session.lens_model.unembed(
                                chunk_residual[full_local_indices]
                            ).float().cpu()
                            full_summaries = {
                                local_index: _full_vocabulary_summary(
                                    logits=full_logits_batch[batch_index],
                                    session=session,
                                    top_k=top_k,
                                )
                                for batch_index, local_index in enumerate(
                                    full_local_indices
                                )
                            }
                        for local_index, position in enumerate(chunk_positions):
                            score_values = {
                                word: float(scores[local_index])
                                for word, scores in candidate_scores.items()
                            }
                            record: dict[str, Any] = {
                                "schema_version": 1,
                                "timestamp_utc": utc_now(),
                                "run_id": behavior_row["run_id"],
                                "prompt_id": behavior_row["prompt_id"],
                                "prompt_type": behavior_row["prompt_type"],
                                "split": behavior_row["split"],
                                "condition": behavior_row["condition"],
                                "target_word": behavior_row.get("secret"),
                                "source_path": behavior_row["source_path"],
                                "source_line": behavior_row["source_line"],
                                "source_submodule_commit": behavior_row[
                                    "source_submodule_commit"
                                ],
                                "base_model_repo_id": behavior_row[
                                    "base_model_repo_id"
                                ],
                                "base_model_revision": behavior_row[
                                    "base_model_revision"
                                ],
                                "tokenizer_repo_id": behavior_row[
                                    "tokenizer_repo_id"
                                ],
                                "tokenizer_revision": behavior_row[
                                    "tokenizer_revision"
                                ],
                                "adapter_repo_id": behavior_row[
                                    "adapter_repo_id"
                                ],
                                "adapter_revision": behavior_row[
                                    "adapter_revision"
                                ],
                                "jlens_repo_id": behavior_row["jlens_repo_id"],
                                "jlens_revision": behavior_row["jlens_revision"],
                                "jlens_filename": behavior_row["jlens_filename"],
                                "jlens_code_commit": behavior_row[
                                    "jlens_code_commit"
                                ],
                                "runtime_dtype": behavior_row["runtime_dtype"],
                                "attention_implementation": behavior_row[
                                    "attention_implementation"
                                ],
                                "seed": behavior_row["seed"],
                                "output_leaks": behavior_row.get(
                                    "output_candidate_leaks", []
                                ),
                                "own_secret_leaked": behavior_row.get(
                                    "own_secret_leaked", False
                                ),
                                "method": method,
                                "layer": layer,
                                "position": position,
                                "position_roles": _position_roles(
                                    position,
                                    prompt_length=len(prompt_ids),
                                    sequence_length=len(complete_ids),
                                    input_window=readout["input_window"],
                                ),
                                "relative_generated_position": (
                                    position - len(prompt_ids)
                                    if position >= len(prompt_ids)
                                    else None
                                ),
                                "token_id": complete_ids[position],
                                "token": session.tokenizer.decode(
                                    [complete_ids[position]]
                                ),
                                "candidate_logits": score_values,
                                "candidate_ranks": _candidate_ranks(
                                    candidate_scores, local_index
                                ),
                                "full_vocabulary": full_summaries.get(local_index),
                            }
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            record_count += 1
                    if method == "jlens":
                        del residual
                del activations[layer]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return record_count


def run_gold_blue_sweep(
    *,
    session: ModelSession,
    paths: RunPaths,
    prompt_ids: list[str],
    conditions: list[str],
) -> list[dict[str, Any]]:
    behavior = behavior_dataframe(paths)
    if behavior.empty:
        raise RuntimeError("Behavior generations are missing")
    selected = behavior[
        behavior["prompt_id"].isin(prompt_ids)
        & behavior["condition"].isin(conditions)
    ]
    expected = len(prompt_ids) * len(conditions)
    if len(selected) != expected:
        found = set(zip(selected["prompt_id"], selected["condition"], strict=False))
        missing = [
            (prompt_id, condition)
            for prompt_id in prompt_ids
            for condition in conditions
            if (prompt_id, condition) not in found
        ]
        raise RuntimeError(f"Missing behavior rows required by sweep: {missing}")

    statuses: list[dict[str, Any]] = []
    cells_dir = paths.lens_dir / "cells"
    ordered_rows = selected.sort_values(["prompt_id", "condition"]).to_dict("records")
    sweep_started = time.perf_counter()
    for index, row in enumerate(ordered_rows, start=1):
        sequence_started = time.perf_counter()
        filename = f"{_safe_name(row['prompt_id'])}__{_safe_name(row['condition'])}.jsonl"
        output = cells_dir / filename
        if output.exists():
            status = "already_complete"
            records = sum(1 for _ in output.open(encoding="utf-8"))
        else:
            records = read_sequence(session=session, behavior_row=row, output_path=output)
            status = "completed"
        sequence_seconds = time.perf_counter() - sequence_started
        total_seconds = time.perf_counter() - sweep_started
        mean_seconds = total_seconds / index
        eta_seconds = mean_seconds * (len(ordered_rows) - index)
        statuses.append(
            {
                "prompt_id": row["prompt_id"],
                "condition": row["condition"],
                "status": status,
                "records": records,
                "path": str(output),
                "sequence_seconds": sequence_seconds,
                "elapsed_seconds": total_seconds,
                "eta_seconds": eta_seconds,
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(ordered_rows)}",
                    **statuses[-1],
                }
            ),
            flush=True,
        )
    return statuses


def iter_lens_records(paths: RunPaths) -> Iterable[dict[str, Any]]:
    for path in sorted((paths.lens_dir / "cells").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
