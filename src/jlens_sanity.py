from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.experiment_io import RunPaths, read_jsonl, utc_now
from src.lens_readout import _decode_topk, _recorded_activations
from src.model_session import ModelSession, audit_candidate_tokens
from src.prompt_data import contains_word


@torch.no_grad()
def run_base_jlens_sanity(session: ModelSession, paths: RunPaths) -> Path:
    """Run one official J-Lens multihop item on the unadapted base model."""

    session.load_jlens()
    sanity = session.config["sanity"]
    source = Path(__file__).resolve().parent.parent / sanity["source"]
    items = json.loads(source.read_text(encoding="utf-8"))["items"]
    item = next(item for item in items if item["name"] == sanity["item_name"])
    target = sanity["target_intermediate"]
    target_audit = audit_candidate_tokens(session.tokenizer, [target])[target]
    input_ids = session.lens_model.encode(
        item["prompt"], max_length=session.config["runtime"]["max_sequence_tokens"]
    )
    position = input_ids.shape[1] - 1
    layers = list(session.lens.source_layers)
    output = paths.lens_dir / "sanity" / f"{item['name']}.jsonl"
    if output.exists():
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    with session.condition("base"):
        activations = _recorded_activations(session, input_ids, layers)
        with temporary.open("w", encoding="utf-8") as handle:
            for layer in layers:
                source_residual = activations[layer][position : position + 1].float()
                for method in ("logit_lens", "jlens"):
                    residual = source_residual
                    if method == "jlens":
                        residual = session.lens.transport(residual, layer)
                    logits = session.lens_model.unembed(residual)[0].float().cpu()
                    target_ids = target_audit["single_token_ids"]
                    target_scores = logits[target_ids]
                    best_offset = int(target_scores.argmax())
                    best_id = int(target_ids[best_offset])
                    record: dict[str, Any] = {
                        "schema_version": 1,
                        "timestamp_utc": utc_now(),
                        "run_id": paths.run_id,
                        "source": sanity["source"],
                        "prompt_id": f"jlens_sanity_{item['name']}",
                        "prompt": item["prompt"],
                        "base_model_repo_id": session.config["base_model"][
                            "repo_id"
                        ],
                        "base_model_revision": session.config["base_model"][
                            "revision"
                        ],
                        "tokenizer_repo_id": session.config["base_model"][
                            "repo_id"
                        ],
                        "tokenizer_revision": session.config["base_model"][
                            "revision"
                        ],
                        "jlens_repo_id": session.config["jlens"]["repo_id"],
                        "jlens_revision": session.config["jlens"]["revision"],
                        "jlens_filename": session.config["jlens"]["filename"],
                        "jlens_code_commit": session.config["jlens"][
                            "official_code_commit"
                        ],
                        "runtime_dtype": session.config["runtime"]["dtype"],
                        "attention_implementation": session.config["runtime"][
                            "attention_implementation"
                        ],
                        "seed": session.config["seed"],
                        "method": method,
                        "layer": layer,
                        "position": position,
                        "target": target,
                        "target_token_id": best_id,
                        "target_token": session.tokenizer.decode([best_id]),
                        "target_logit": float(logits[best_id]),
                        "target_rank": int((logits > logits[best_id]).sum()) + 1,
                        "top_k": _decode_topk(
                            session.tokenizer,
                            logits,
                            session.config["readout"]["top_k"],
                        ),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                del activations[layer]
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temporary, output)
    metadata = {
        "source": sanity["source"],
        "item": item,
        "target_intermediate": target,
        "readouts": str(output),
    }
    (paths.result_dir / "jlens_sanity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def sanity_review_path(paths: RunPaths, config: dict[str, Any]) -> Path:
    return paths.result_dir / config["sanity"]["gate_filename"]


def ensure_sanity_review_template(
    paths: RunPaths,
    config: dict[str, Any],
    output: Path,
) -> Path:
    records = read_jsonl(output)
    sanity = config["sanity"]
    expected_layers = config["base_model"]["expected_num_hidden_layers"]
    layers_by_method = {
        method: sorted({row["layer"] for row in records if row["method"] == method})
        for method in ("logit_lens", "jlens")
    }
    prompts = {row["prompt"] for row in records}
    target = sanity["target_intermediate"]
    machine_checks = {
        "records_present": bool(records),
        "exactly_one_prompt": len(prompts) == 1,
        "target_absent_from_prompt": (
            len(prompts) == 1 and not contains_word(next(iter(prompts)), target)
        ),
        "two_methods_present": set(layers_by_method) == {"logit_lens", "jlens"}
        and all(layers_by_method.values()),
        "methods_cover_same_layers": (
            layers_by_method["logit_lens"] == layers_by_method["jlens"]
        ),
        "layer_coverage_is_contiguous": all(
            layers in (list(range(expected_layers)), list(range(expected_layers - 1)))
            for layers in layers_by_method.values()
        ),
    }
    output_path = sanity_review_path(paths, config)
    if output_path.exists():
        return output_path
    template = {
        "run_id": paths.run_id,
        "created_utc": utc_now(),
        "approved": False,
        "reviewer": None,
        "notes": (
            "Inspect target tokenization, full-vocabulary ranks and top-k values "
            "for both methods. Approval means the pipeline is interpretable, not "
            "that J-Lens must outperform Logit Lens."
        ),
        "machine_checks": machine_checks,
        "layers_by_method": layers_by_method,
        "human_checks": {
            "target_tokenization_inspected": None,
            "layer_indexing_and_rank_trajectory_inspected": None,
            "top_k_outputs_are_finite_and_interpretable": None,
            "pipeline_is_safe_to_apply_to_taboo": None,
        },
    }
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return output_path


def require_sanity_approval(paths: RunPaths, config: dict[str, Any]) -> None:
    if not config["sanity"].get("require_approval", False):
        return
    path = sanity_review_path(paths, config)
    if not path.exists():
        raise RuntimeError(f"Base J-Lens sanity review is missing: {path}")
    review = json.loads(path.read_text(encoding="utf-8"))
    machine = review.get("machine_checks", {})
    human = review.get("human_checks", {})
    if (
        not review.get("approved")
        or not machine
        or not all(value is True for value in machine.values())
        or not human
        or not all(value is True for value in human.values())
    ):
        raise RuntimeError(
            "Base J-Lens sanity gate is not approved. Inspect notebook 02, then "
            f"update {path}."
        )
