from __future__ import annotations

import json
from typing import Any


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    candidate_logits = record["candidate_logits"]
    candidate_ranks = record["candidate_ranks"]
    target = record.get("target_word")
    foil = "blue" if target == "gold" else "gold" if target == "blue" else None
    full = record.get("full_vocabulary")
    full_candidates = full["candidates"] if full else {}
    target_logit = candidate_logits.get(target) if target else None
    foil_logit = candidate_logits.get(foil) if foil else None
    return {
        "schema_version": record["schema_version"],
        "run_id": record["run_id"],
        "prompt_id": record["prompt_id"],
        "prompt_type": record["prompt_type"],
        "split": record["split"],
        "condition": record["condition"],
        "target_word": target,
        "source_path": record["source_path"],
        "source_line": record["source_line"],
        "source_submodule_commit": record["source_submodule_commit"],
        "base_model_repo_id": record["base_model_repo_id"],
        "base_model_revision": record["base_model_revision"],
        "tokenizer_repo_id": record["tokenizer_repo_id"],
        "tokenizer_revision": record["tokenizer_revision"],
        "adapter_repo_id": record["adapter_repo_id"],
        "adapter_revision": record["adapter_revision"],
        "jlens_repo_id": record["jlens_repo_id"],
        "jlens_revision": record["jlens_revision"],
        "jlens_filename": record["jlens_filename"],
        "jlens_code_commit": record["jlens_code_commit"],
        "runtime_dtype": record["runtime_dtype"],
        "attention_implementation": record["attention_implementation"],
        "seed": record["seed"],
        "method": record["method"],
        "layer": record["layer"],
        "position": record["position"],
        "position_roles_json": json.dumps(record["position_roles"]),
        "relative_generated_position": record["relative_generated_position"],
        "token_id": record["token_id"],
        "token": record["token"],
        "own_secret_leaked": record["own_secret_leaked"],
        "output_leaks_json": json.dumps(record["output_leaks"]),
        "gold_logit": candidate_logits["gold"],
        "blue_logit": candidate_logits["blue"],
        "gold_candidate_rank": candidate_ranks["gold"],
        "blue_candidate_rank": candidate_ranks["blue"],
        "predicted_candidate": max(candidate_logits, key=candidate_logits.get),
        "target_logit": target_logit,
        "foil_logit": foil_logit,
        "target_margin": (
            target_logit - foil_logit
            if target_logit is not None and foil_logit is not None
            else None
        ),
        "target_candidate_rank": candidate_ranks.get(target) if target else None,
        "gold_full_rank": full_candidates.get("gold", {}).get("rank"),
        "blue_full_rank": full_candidates.get("blue", {}).get("rank"),
        "target_full_rank": full_candidates.get(target, {}).get("rank") if target else None,
        "top_k_json": json.dumps(full["top_k"], ensure_ascii=False) if full else None,
    }
