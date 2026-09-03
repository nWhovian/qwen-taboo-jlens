#!/usr/bin/env python3
"""Build notebooks 07-08 for the full Qwen 3.6 Taboo test run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


def write(name: str, cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:03d}")
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Qwen Taboo J-Lens",
                "language": "python",
                "name": "qwen-taboo-jlens",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


COMMON_SETUP = r'''
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT
'''


write(
    "07_qwen36_20_adapter_test_sweep.ipynb",
    [
        markdown(
            r'''
# 07 — Full Qwen 3.6 20-adapter test sweep: Logit Lens × J-Lens

**Objective.** Run the untouched published test set end to end: 100 standard
and 100 direct prompts under each of the 20 word-specific Qwen 3.6 Taboo LoRA
adapters (4,000 `prompt × adapter` sequences). For every sequence, record
Logit Lens and J-Lens on all 63 fitted source layers and every measured token
position (prompt tail, assistant-control/header/separator positions, and the
generated response).

This notebook **requires and reuses** the Qwen model, tokenizer and J-Lens that
are already alive in the shared kernel used by notebooks 01–05. It deliberately
has no fallback that loads another 27B model. It loads only missing LoRA
adapters, performs numerical implementation checks, and then runs without a
manual approval gate.

Primary ranking removes every token ID that the model actually emitted in that
response. The raw artifacts also retain two diagnostics: only the actual token
at each response position removed, and no emitted-token mask. Literal
own-secret leaks are saved and audited but excluded from headline metrics in
notebook 08.
            '''
        ),
        code(COMMON_SETUP),
        markdown(
            r'''
## Capture the already-loaded kernel state

The references below do not copy model weights. Missing or incompatible state
is an error: restart from notebook 05 in the same kernel rather than silently
allocating another Qwen.
            '''
        ),
        code(
            r'''
test_prior_state = {
    "config": globals().get("config"),
    "model": globals().get("model"),
    "tokenizer": globals().get("tokenizer"),
    "adapter_names": dict(globals().get("adapter_names", {})),
    "lens": globals().get("lens"),
    "lens_model": globals().get("lens_model"),
}
print({
    "model_in_memory": test_prior_state["model"] is not None,
    "tokenizer_in_memory": test_prior_state["tokenizer"] is not None,
    "adapters_in_memory": sorted(test_prior_state["adapter_names"]),
    "jlens_in_memory": (
        test_prior_state["lens"] is not None
        and test_prior_state["lens_model"] is not None
    ),
})
assert test_prior_state["model"] is not None, "Run notebook 05 in this kernel first."
assert test_prior_state["tokenizer"] is not None, "Matching tokenizer is missing."
assert test_prior_state["lens"] is not None, "Pinned J-Lens checkpoint is missing."
assert test_prior_state["lens_model"] is not None, "J-Lens HF wrapper is missing."
            '''
        ),
        markdown(
            r'''
## Open one immutable, resumable test run

`TEST_RUN_ID` survives cell re-execution in this kernel. If the kernel restarts,
the pointer resumes only an incomplete run with the identical config hash.
Every final prompt ID and every model revision is also written to the manifest.
            '''
        ),
        code(
            r'''
from src.experiment_io import (
    create_run,
    load_json,
    stable_hash,
    update_manifest,
    utc_now,
)
from src.prompt_data import load_prompts, lexical_leaks

TEST_CONFIG_PATH = "configs/qwen36_20_adapter_test.json"
test_config = load_json(PROJECT_ROOT / TEST_CONFIG_PATH)
test_config_hash = stable_hash(test_config)
test_pointer_path = PROJECT_ROOT / "results" / "latest_qwen36_20_adapter_test_run.json"

requested_test_run_id = os.environ.get("QWEN_TEST_RUN_ID")
if requested_test_run_id is None:
    requested_test_run_id = globals().get("TEST_RUN_ID")
if requested_test_run_id is None and test_pointer_path.exists():
    pointer = load_json(test_pointer_path)
    candidate_manifest = (
        PROJECT_ROOT / "results" / pointer["run_id"] / "manifest.json"
    )
    if candidate_manifest.exists():
        candidate = load_json(candidate_manifest)
        if (
            candidate.get("config_hash") == test_config_hash
            and candidate.get("status") != "complete"
        ):
            requested_test_run_id = pointer["run_id"]

test_paths = create_run(TEST_CONFIG_PATH, run_id=requested_test_run_id)
TEST_RUN_ID = test_paths.run_id
test_pointer_path.parent.mkdir(parents=True, exist_ok=True)
test_pointer_tmp = test_pointer_path.with_suffix(".json.tmp")
test_pointer_tmp.write_text(
    json.dumps(
        {
            "run_id": TEST_RUN_ID,
            "config_hash": test_config_hash,
            "updated_utc": utc_now(),
        },
        indent=2,
    ),
    encoding="utf-8",
)
os.replace(test_pointer_tmp, test_pointer_path)
print("TEST_RUN_ID =", TEST_RUN_ID)
print("results =", test_paths.result_dir)
print("lens cells =", test_paths.lens_dir / "test_cells")
            '''
        ),
        markdown(
            r'''
## Load and audit all 200 test prompts

Selection is not random and does not take a prefix: it selects **every** record
whose published split is `test`, then verifies exactly 100 `standard` and 100
`direct` IDs. The exact ordered IDs and source-data hash are frozen in this run.
            '''
        ),
        code(
            r'''
test_prompt_catalog = load_prompts(test_config["prompts"]["path"])
test_prompt_path = PROJECT_ROOT / test_config["prompts"]["path"]
test_prompt_provenance = load_json(
    PROJECT_ROOT / test_config["prompts"]["provenance_path"]
)
test_prompt_sha256 = hashlib.sha256(test_prompt_path.read_bytes()).hexdigest()
assert test_prompt_provenance["records"] == len(test_prompt_catalog)
assert test_prompt_provenance["sha256"] == test_prompt_sha256

test_prompts = sorted(
    [
        prompt
        for prompt in test_prompt_catalog.values()
        if prompt["split"] == test_config["prompts"]["split"]
        and prompt["prompt_type"] in {"standard", "direct"}
    ],
    key=lambda prompt: (0 if prompt["prompt_type"] == "standard" else 1, prompt["prompt_id"]),
)
test_standard_prompts = [p for p in test_prompts if p["prompt_type"] == "standard"]
test_direct_prompts = [p for p in test_prompts if p["prompt_type"] == "direct"]
assert len(test_standard_prompts) == test_config["prompts"]["expected_standard"] == 100
assert len(test_direct_prompts) == test_config["prompts"]["expected_direct"] == 100
assert len(test_prompts) == len({p["prompt_id"] for p in test_prompts}) == 200
assert all("_test_" in p["prompt_id"] for p in test_prompts)

test_conditions = list(test_config["behavior"]["conditions"])
assert test_conditions == list(test_config["adapters"])
assert len(test_conditions) == len(set(test_conditions)) == 20

raw_prompt_leaks = {}
for prompt in test_prompts:
    raw_text = "\n".join(message["content"] for message in prompt["messages"])
    leaks = lexical_leaks(raw_text, test_conditions)
    if leaks:
        raw_prompt_leaks[prompt["prompt_id"]] = leaks
assert not raw_prompt_leaks, raw_prompt_leaks

test_prompt_table = pd.DataFrame([
    {
        "prompt_id": prompt["prompt_id"],
        "prompt_type": prompt["prompt_type"],
        "split": prompt["split"],
        "paper_block_of_10": int(prompt["prompt_id"].rsplit("_", 1)[1]) // 10,
        "text": prompt["messages"][0]["content"],
        "source": f"{prompt['source_path']}:{prompt['source_line']}",
    }
    for prompt in test_prompts
])
test_selection = {
    "schema_version": 1,
    "run_id": TEST_RUN_ID,
    "prompt_file_sha256": test_prompt_sha256,
    "prompt_provenance": test_prompt_provenance,
    "standard_prompt_ids": [p["prompt_id"] for p in test_standard_prompts],
    "direct_prompt_ids": [p["prompt_id"] for p in test_direct_prompts],
    "conditions": test_conditions,
    "expected_sequences": len(test_prompts) * len(test_conditions),
}
(test_paths.result_dir / "test_prompt_selection.json").write_text(
    json.dumps(test_selection, ensure_ascii=False, indent=2), encoding="utf-8"
)
update_manifest(
    test_paths,
    status="prompts_frozen",
    prompt_file_sha256=test_prompt_sha256,
    selected_prompt_ids=[p["prompt_id"] for p in test_prompts],
    conditions=test_conditions,
    expected_sequences=test_selection["expected_sequences"],
)
display(test_prompt_table.groupby(["prompt_type", "paper_block_of_10"]).size())
with pd.option_context("display.max_colwidth", 100):
    display(test_prompt_table.groupby("prompt_type", as_index=False).head(3))
            '''
        ),
        markdown(
            r'''
## Verify runtime, installed J-Lens code, and reuse the base model

No `from_pretrained` call for Qwen appears in this notebook. The base-model
revision comes from the already-loaded run config and must match exactly.
            '''
        ),
        code(
            r'''
import torch
from src.preflight import runtime_dependency_preflight

test_runtime_report = runtime_dependency_preflight()
display(test_runtime_report)
assert test_runtime_report["passed"], test_runtime_report.get("action")

jlens_distribution = distribution("jlens")
jlens_direct_url_text = jlens_distribution.read_text("direct_url.json")
assert jlens_direct_url_text, "Installed jlens has no PEP 610 Git metadata."
jlens_direct_url = json.loads(jlens_direct_url_text)
actual_jlens_commit = jlens_direct_url.get("vcs_info", {}).get("commit_id")
assert actual_jlens_commit == test_config["jlens"]["official_code_commit"], {
    "actual": actual_jlens_commit,
    "expected": test_config["jlens"]["official_code_commit"],
}

prior_config = test_prior_state["config"]
assert prior_config is not None, "Loaded model revision cannot be audited."
assert prior_config["base_model"] == test_config["base_model"], {
    "loaded": prior_config["base_model"],
    "required": test_config["base_model"],
}
for key in ("repo_id", "revision", "filename", "official_code_commit"):
    assert prior_config["jlens"][key] == test_config["jlens"][key], key

model = test_prior_state["model"]
tokenizer = test_prior_state["tokenizer"]
lens = test_prior_state["lens"]
lens_model = test_prior_state["lens_model"]
model.eval()
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

test_text_config = model.config.get_text_config()
test_base_spec = test_config["base_model"]
assert test_text_config.hidden_size == test_base_spec["expected_hidden_size"]
assert test_text_config.num_hidden_layers == test_base_spec["expected_num_hidden_layers"]
assert {parameter.device.type for parameter in model.parameters()} == {"cuda"}
assert lens_model._hf_model is model, "J-Lens wrapper is not attached to this model object."
assert lens.d_model == test_base_spec["expected_hidden_size"]
assert lens.n_prompts == test_config["jlens"]["expected_n_prompts"]
assert len(lens.source_layers) == test_config["jlens"]["expected_source_layers"]
assert list(lens.source_layers) == list(range(63))

test_runtime = test_config["runtime"]
test_seed = test_config["seed"]
random.seed(test_seed)
np.random.seed(test_seed)
torch.manual_seed(test_seed)
torch.cuda.manual_seed_all(test_seed)
test_device = next(model.parameters()).device
print({
    "base_reused": True,
    "device": str(test_device),
    "dtype": str(next(model.parameters()).dtype),
    "jlens_code_commit": actual_jlens_commit,
    "jlens_layers": [min(lens.source_layers), max(lens.source_layers)],
    "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
})
            '''
        ),
        markdown(
            r'''
## Load only the 17 missing adapters and audit all 20

Gold, Blue and Moon are reused when present. Every missing adapter is fetched
at its immutable Hugging Face commit. For each adapter we require finite LoRA A
and B tensors and a non-zero B update. The audit is saved after every adapter,
so a download interruption remains diagnosable.
            '''
        ),
        code(
            r'''
def test_adapter_runtime_name(repo_id):
    return repo_id.replace(".", "_").replace("/", "__")

def audit_test_adapter(word, runtime_name):
    tensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if runtime_name in name and ".lora_" in name
    ]
    assert tensors, f"No LoRA tensors found for {word}: {runtime_name}"
    assert all(torch.isfinite(parameter).all().item() for _, parameter in tensors)
    b_tensors = [parameter for name, parameter in tensors if ".lora_B." in name]
    assert b_tensors, f"No LoRA B tensors found for {word}"
    b_norm_sum = sum(float(parameter.float().norm().item()) for parameter in b_tensors)
    assert b_norm_sum > 0.0, f"All LoRA B tensors are zero for {word}"
    return {
        "adapter_name": runtime_name,
        "tensor_count": len(tensors),
        "parameter_count": int(sum(parameter.numel() for _, parameter in tensors)),
        "dtypes": sorted({str(parameter.dtype) for _, parameter in tensors}),
        "lora_a_norm_sum": sum(
            float(parameter.float().norm().item())
            for name, parameter in tensors
            if ".lora_A." in name
        ),
        "lora_b_norm_sum": b_norm_sum,
    }

test_adapter_names = {}
test_adapter_audit = {}
loaded_peft_names = set(getattr(model, "peft_config", {}))
for adapter_index, word in enumerate(test_conditions, start=1):
    spec = test_config["adapters"][word]
    runtime_name = test_adapter_runtime_name(spec["repo_id"])
    if runtime_name not in loaded_peft_names:
        print(f"[{adapter_index:02d}/20] loading {word}: {spec['repo_id']}", flush=True)
        model.load_adapter(
            spec["repo_id"],
            adapter_name=runtime_name,
            adapter_kwargs={"revision": spec["revision"]},
        )
        loaded_peft_names.add(runtime_name)
    else:
        print(f"[{adapter_index:02d}/20] reusing {word}", flush=True)
    test_adapter_names[word] = runtime_name
    test_adapter_audit[word] = {
        **audit_test_adapter(word, runtime_name),
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "gpu_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }
    (test_paths.result_dir / "loaded_20_adapter_parameter_audit.json").write_text(
        json.dumps(test_adapter_audit, indent=2), encoding="utf-8"
    )

model.eval()
assert set(test_adapter_names) == set(test_conditions)
assert set(test_adapter_names.values()).issubset(set(model.peft_config))
print("All adapters ready:", len(test_adapter_names))
print("GPU allocated GiB:", round(torch.cuda.memory_allocated() / 2**30, 2))
            '''
        ),
        markdown(
            r'''
## Audit target token forms and render the 200 prompts

Each secret is represented by every audited one-token lowercase/capitalized
form, with and without a leading space. Activations are attached to the token
that has just been read; `prediction_target_token` records the next token.
The assistant header is split into explicit control, role, thinking-tag and
separator labels rather than one ambiguous bucket.
            '''
        ),
        code(
            r'''
from src.prompt_data import assert_prompt_has_no_candidates

test_token_audit = {}
for word in test_conditions:
    form_map = {}
    for form in (word, word.capitalize(), " " + word, " " + word.capitalize()):
        ids = tokenizer.encode(form, add_special_tokens=False)
        form_map[form] = [int(token_id) for token_id in ids]
    single_token_ids = sorted({
        ids[0] for ids in form_map.values() if len(ids) == 1
    })
    assert single_token_ids, (word, form_map)
    test_token_audit[word] = {
        "forms": form_map,
        "single_token_ids": single_token_ids,
        "decoded_single_token_forms": {
            str(token_id): tokenizer.decode([token_id])
            for token_id in single_token_ids
        },
    }
(test_paths.result_dir / "candidate_token_audit_20_words.json").write_text(
    json.dumps(test_token_audit, ensure_ascii=False, indent=2), encoding="utf-8"
)

def find_last_subsequence(sequence, subsequence):
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None

test_assistant_header_ids = tokenizer.encode(
    "<|im_start|>assistant\n", add_special_tokens=False
)
test_rendered_by_prompt = {}
test_render_audit = []
for prompt in test_prompts:
    rendered = tokenizer.apply_chat_template(
        prompt["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=test_runtime["enable_thinking"],
    )
    prompt_ids = tokenizer(
        rendered, add_special_tokens=False, return_attention_mask=False
    ).input_ids
    assert_prompt_has_no_candidates(rendered, test_conditions)
    assistant_start = find_last_subsequence(prompt_ids, test_assistant_header_ids)
    assert assistant_start is not None, (
        prompt["prompt_id"], test_assistant_header_ids, prompt_ids[-16:]
    )
    test_rendered_by_prompt[prompt["prompt_id"]] = {
        "rendered": rendered,
        "prompt_token_ids": [int(token_id) for token_id in prompt_ids],
        "assistant_header_start": int(assistant_start),
    }
    test_render_audit.append({
        "prompt_id": prompt["prompt_id"],
        "prompt_type": prompt["prompt_type"],
        "prompt_token_count": len(prompt_ids),
        "assistant_header_start": assistant_start,
        "assistant_header_pieces": [
            tokenizer.decode([int(token_id)]) for token_id in prompt_ids[assistant_start:]
        ],
        "rendered_prompt": rendered,
    })
(test_paths.result_dir / "test_rendered_prompt_audit.json").write_text(
    json.dumps(test_render_audit, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("assistant header IDs:", test_assistant_header_ids)
print("rendered prompts:", len(test_rendered_by_prompt))
with pd.option_context("display.max_colwidth", 120):
    display(pd.DataFrame(test_render_audit).head(5))
            '''
        ),
        code(
            r'''
def test_token_kind(token_text, token_id):
    if token_id in tokenizer.all_special_ids:
        return "special/control"
    if token_text.isspace():
        return "whitespace"
    stripped = token_text.strip()
    if stripped and all(unicodedata.category(char).startswith("P") for char in stripped):
        return "punctuation"
    if stripped.isnumeric():
        return "number"
    if any(char.isalpha() for char in stripped):
        return "word/subword"
    return "other"

def test_position_metadata(complete_ids, prompt_length, assistant_start, position):
    token_id = int(complete_ids[position])
    token_text = tokenizer.decode([token_id])
    next_token_id = int(complete_ids[position + 1]) if position + 1 < len(complete_ids) else None
    next_token_text = tokenizer.decode([next_token_id]) if next_token_id is not None else None
    relative = position - prompt_length
    header_offset = position - assistant_start if position >= assistant_start else None

    if position < assistant_start:
        role = "user_prompt_tail"
        label = f"user tail: {token_text!r}"
    elif position < prompt_length:
        if header_offset == 0:
            role = "assistant_turn_start_control"
        elif token_text.strip() == "assistant":
            role = "assistant_role_token"
        elif token_text == "<think>":
            role = "assistant_thinking_open_control"
        elif token_text == "</think>":
            role = "assistant_thinking_close_control"
        elif position == prompt_length - 1:
            role = "response_start_boundary_separator"
        else:
            role = "assistant_header_separator"
        label = f"assistant header +{header_offset}: {token_text!r}"
    else:
        if relative == 0:
            role = "response_token_first"
        elif relative == len(complete_ids) - prompt_length - 1:
            role = "response_token_last"
        else:
            role = "response_token"
        label = f"generated token {relative}: {token_text!r}"

    left = max(0, position - 4)
    right = min(len(complete_ids), position + 5)
    pieces = [tokenizer.decode([int(token)]) for token in complete_ids[left:right]]
    pieces[position - left] = "[" + pieces[position - left] + "]"
    return {
        "position_role": role,
        "position_label": label,
        "assistant_header_offset": header_offset if position < prompt_length else None,
        "relative_response_position": relative if position >= prompt_length else None,
        "position_from_prompt_end": position - (prompt_length - 1),
        "observed_token_id": token_id,
        "observed_token": token_text,
        "prediction_target_token_id": next_token_id,
        "prediction_target_token": next_token_text,
        "token_kind": test_token_kind(token_text, token_id),
        "context": "".join(pieces),
        "jlens_in_fit_position_domain": (
            position >= test_config["jlens"]["fit_min_absolute_position"]
        ),
    }
            '''
        ),
        markdown(
            r'''
## Numerical implementation preflight

1. Unembedding the final block output through the J-Lens wrapper must reproduce
   the model's own final logits.
2. Gold → Blue → Gold must be deterministic, while Gold and Blue must differ.
3. One transported J-Lens residual and its logits must be finite.

These checks catch wrong residual streams, wrong normalization/head wiring,
adapter-switch failures, and obvious J-Lens numerical corruption before the
4,000-sequence sweep.
            '''
        ),
        code(
            r'''
import jlens
from jlens.hooks import ActivationRecorder

preflight_prompt = test_standard_prompts[0]
preflight_info = test_rendered_by_prompt[preflight_prompt["prompt_id"]]
preflight_ids = torch.tensor(
    [preflight_info["prompt_token_ids"]], device=lens_model.input_device
)

def adapter_last_logits(word):
    model.enable_adapters()
    model.set_adapter(test_adapter_names[word])
    with torch.no_grad():
        return model(input_ids=preflight_ids, use_cache=False).logits[0, -1].float()

gold_logits_1 = adapter_last_logits("gold")
blue_logits = adapter_last_logits("blue")
gold_logits_2 = adapter_last_logits("gold")
roundtrip_max_abs = float((gold_logits_1 - gold_logits_2).abs().max().item())
gold_blue_mean_abs = float((gold_logits_1 - blue_logits).abs().mean().item())
assert roundtrip_max_abs <= 1e-6, roundtrip_max_abs
assert gold_blue_mean_abs > 1e-4, gold_blue_mean_abs

model.set_adapter(test_adapter_names["gold"])
last_layer = test_text_config.num_hidden_layers - 1
with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[last_layer]) as final_recorder:
    direct_output = model(input_ids=preflight_ids, use_cache=False)
final_residual = final_recorder.activations[last_layer].detach()
reconstructed_logits = lens_model.unembed(final_residual).float()
direct_logits = direct_output.logits.float()
unembed_max_abs = float((reconstructed_logits - direct_logits).abs().max().item())
unembed_mean_abs = float((reconstructed_logits - direct_logits).abs().mean().item())
unembed_top1_match = bool(
    torch.equal(reconstructed_logits.argmax(-1), direct_logits.argmax(-1))
)
assert unembed_top1_match
assert unembed_mean_abs <= 0.02, unembed_mean_abs
assert unembed_max_abs <= 0.5, unembed_max_abs

probe_layer = 32
with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[probe_layer]) as probe_recorder:
    lens_model.forward(preflight_ids)
probe_source = probe_recorder.activations[probe_layer].detach()[0, -1].float()
probe_transport = lens.transport(probe_source, probe_layer)
probe_jlens_logits = lens_model.unembed(probe_transport).float()
assert torch.isfinite(probe_transport).all().item()
assert torch.isfinite(probe_jlens_logits).all().item()

test_numerical_preflight = {
    "schema_version": 1,
    "created_utc": utc_now(),
    "prompt_id": preflight_prompt["prompt_id"],
    "adapter_roundtrip_gold_max_abs": roundtrip_max_abs,
    "gold_vs_blue_mean_abs": gold_blue_mean_abs,
    "final_unembed_max_abs": unembed_max_abs,
    "final_unembed_mean_abs": unembed_mean_abs,
    "final_unembed_top1_match": unembed_top1_match,
    "jlens_probe_layer": probe_layer,
    "jlens_probe_logits_finite": True,
}
(test_paths.result_dir / "test_numerical_preflight.json").write_text(
    json.dumps(test_numerical_preflight, indent=2), encoding="utf-8"
)
display(test_numerical_preflight)
del gold_logits_1, gold_logits_2, blue_logits, direct_output, direct_logits
del reconstructed_logits, final_residual, probe_source, probe_transport, probe_jlens_logits
torch.cuda.empty_cache()
            '''
        ),
        markdown(
            r'''
## Non-blocking two-prompt adapter smoke check

This generates one standard and one direct response for every adapter. Empty
outputs or load failures stop the run; qualitative variation and secret leaks
are saved for later inspection but do not require manual approval.
            '''
        ),
        code(
            r'''
from src.experiment_io import append_jsonl, read_jsonl

test_smoke_path = test_paths.raw_dir / "adapter_smoke_20_words.jsonl"
test_smoke_prompts = [
    next(p for p in test_prompts if p["prompt_id"] == prompt_id)
    for prompt_id in test_config["behavior"]["smoke_prompt_ids"]
]
existing_smoke = read_jsonl(test_smoke_path)
completed_smoke = {(row["prompt_id"], row["condition"]) for row in existing_smoke}

for condition in test_conditions:
    model.enable_adapters()
    model.set_adapter(test_adapter_names[condition])
    for prompt in test_smoke_prompts:
        key = (prompt["prompt_id"], condition)
        if key in completed_smoke:
            continue
        info = test_rendered_by_prompt[prompt["prompt_id"]]
        input_ids = torch.tensor([info["prompt_token_ids"]], device=test_device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=48,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=(
                    model.generation_config.pad_token_id
                    if model.generation_config.pad_token_id is not None
                    else tokenizer.pad_token_id
                ),
                use_cache=True,
            )
        generation_ids = generated[0, input_ids.shape[1]:].tolist()
        output_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
        assert generation_ids and output_text.strip(), key
        leaks = lexical_leaks(output_text, test_conditions)
        row = {
            "timestamp_utc": utc_now(),
            "run_id": TEST_RUN_ID,
            "prompt_id": prompt["prompt_id"],
            "prompt_type": prompt["prompt_type"],
            "condition": condition,
            "output_text": output_text,
            "candidate_leaks": leaks,
            "own_secret_leaked": condition in leaks,
        }
        append_jsonl(test_smoke_path, [row])
        existing_smoke.append(row)
        completed_smoke.add(key)
    print("smoke ready:", condition, flush=True)

test_smoke_frame = pd.DataFrame(existing_smoke)
assert len(completed_smoke) == len(test_conditions) * len(test_smoke_prompts) == 40
with pd.option_context("display.max_colwidth", 120):
    display(test_smoke_frame[[
        "prompt_id", "prompt_type", "condition", "own_secret_leaked", "output_text"
    ]].head(12))
            '''
        ),
        markdown(
            r'''
## Generate and save all 4,000 deterministic test responses

One response is generated for every `200 prompts × 20 adapters`. JSONL append
is fsynced after every record, so re-running this cell skips completed keys.
The exact generation token IDs—not re-tokenized text—are used by the activation
sweep. Literal leaks are flags, not a blocking review gate.
            '''
        ),
        code(
            r'''
test_behavior_path = test_paths.raw_dir / "test_behavior_generations.jsonl"

def generate_test_behavior_record(prompt, condition):
    info = test_rendered_by_prompt[prompt["prompt_id"]]
    prompt_ids = info["prompt_token_ids"]
    input_ids = torch.tensor([prompt_ids], device=test_device)
    attention_mask = torch.ones_like(input_ids)
    model.enable_adapters()
    model.set_adapter(test_adapter_names[condition])
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=test_runtime["do_sample"],
            max_new_tokens=test_runtime["max_new_tokens"],
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=(
                model.generation_config.pad_token_id
                if model.generation_config.pad_token_id is not None
                else tokenizer.pad_token_id
            ),
            use_cache=True,
        )
    generation_ids = [int(token_id) for token_id in generated[0, len(prompt_ids):].tolist()]
    assert generation_ids, (prompt["prompt_id"], condition)
    output_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
    candidate_leaks = lexical_leaks(output_text, test_conditions)
    spec = test_config["adapters"][condition]
    return {
        "schema_version": 3,
        "timestamp_utc": utc_now(),
        "run_id": TEST_RUN_ID,
        "config_hash": test_config_hash,
        "prompt_id": prompt["prompt_id"],
        "prompt_type": prompt["prompt_type"],
        "split": prompt["split"],
        "paper_block_of_10": int(prompt["prompt_id"].rsplit("_", 1)[1]) // 10,
        "source_path": prompt["source_path"],
        "source_line": prompt["source_line"],
        "source_submodule_commit": prompt["source_submodule_commit"],
        "messages": prompt["messages"],
        "rendered_prompt": info["rendered"],
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "assistant_header_start": info["assistant_header_start"],
        "condition": condition,
        "secret": condition,
        "base_model_repo_id": test_base_spec["repo_id"],
        "base_model_revision": test_base_spec["revision"],
        "adapter_repo_id": spec["repo_id"],
        "adapter_revision": spec["revision"],
        "jlens_repo_id": test_config["jlens"]["repo_id"],
        "jlens_revision": test_config["jlens"]["revision"],
        "jlens_filename": test_config["jlens"]["filename"],
        "jlens_code_commit": test_config["jlens"]["official_code_commit"],
        "runtime_dtype": test_runtime["dtype"],
        "attention_implementation": test_runtime["attention_implementation"],
        "seed": test_seed,
        "generation_token_ids": generation_ids,
        "generation_token_count": len(generation_ids),
        "output_text": output_text,
        "output_candidate_leaks": candidate_leaks,
        "own_secret_leaked": condition in candidate_leaks,
    }

test_existing_behavior = read_jsonl(test_behavior_path)
for row in test_existing_behavior:
    assert row["config_hash"] == test_config_hash
test_completed_behavior = {
    (row["prompt_id"], row["condition"]) for row in test_existing_behavior
}
expected_behavior_keys = {
    (prompt["prompt_id"], condition)
    for condition in test_conditions
    for prompt in test_prompts
}

generation_started = time.time()
new_behavior_count = 0
for condition_index, condition in enumerate(test_conditions, start=1):
    for prompt_index, prompt in enumerate(test_prompts, start=1):
        key = (prompt["prompt_id"], condition)
        if key in test_completed_behavior:
            continue
        row = generate_test_behavior_record(prompt, condition)
        append_jsonl(test_behavior_path, [row])
        test_existing_behavior.append(row)
        test_completed_behavior.add(key)
        new_behavior_count += 1
        if new_behavior_count == 1 or new_behavior_count % 25 == 0:
            print(
                f"behavior {len(test_completed_behavior)}/4000 | "
                f"adapter {condition_index}/20 {condition} | prompt {prompt_index}/200",
                flush=True,
            )

assert test_completed_behavior == expected_behavior_keys
test_behavior = pd.DataFrame(test_existing_behavior)
test_behavior = test_behavior.drop_duplicates(["prompt_id", "condition"], keep="last")
assert len(test_behavior) == 4000
test_leaks = test_behavior[test_behavior["own_secret_leaked"]].copy()
test_leaks[[
    "prompt_id", "prompt_type", "condition", "output_text"
]].to_csv(test_paths.result_dir / "test_literal_own_secret_leaks.csv", index=False)
test_behavior_summary = (
    test_behavior.groupby(["prompt_type", "condition"], as_index=False)
    .agg(
        sequences=("prompt_id", "size"),
        mean_generation_tokens=("generation_token_count", "mean"),
        literal_own_secret_leaks=("own_secret_leaked", "sum"),
    )
)
test_behavior_summary.to_csv(
    test_paths.result_dir / "test_behavior_summary.csv", index=False
)
print("behavior complete:", len(test_behavior), "leaks:", len(test_leaks))
display(test_behavior_summary.head(20))
update_manifest(
    test_paths,
    status="behavior_complete",
    behavior_sequences=len(test_behavior),
    literal_own_secret_leaks=len(test_leaks),
)
            '''
        ),
        markdown(
            r'''
## Exact full-vocabulary summaries

For each position/layer/method the detailed table stores:

- exact full-vocabulary rank, reciprocal rank, log-rank and rank percentile;
- best-form and total surface-form probability, negative log probability;
- target logit and margin to the highest remaining logit;
- rank/share among the 20 candidate words and the strongest wrong candidate;
- decoded top-10 internal tokens under the primary global emitted-ID mask;
- diagnostic target ranks for position-only masking and no masking.

Response-average rows are saved for all three mask protocols and include the
20-word candidate score dictionary used for majority voting and null controls.
            '''
        ),
        code(
            r'''
test_vocabulary_size = len(tokenizer)
test_candidate_ids_by_word = {
    word: test_token_audit[word]["single_token_ids"] for word in test_conditions
}

def atomic_test_parquet(frame, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, destination)

def masked_probability_and_logits(probabilities, logits, mask_ids_by_row):
    masked_probabilities = probabilities.clone()
    masked_logits = logits.clone()
    if mask_ids_by_row is None:
        return masked_probabilities, masked_logits
    if isinstance(mask_ids_by_row, list) and (
        not mask_ids_by_row or isinstance(mask_ids_by_row[0], int)
    ):
        if mask_ids_by_row:
            masked_probabilities[:, mask_ids_by_row] = -1.0
            masked_logits[:, mask_ids_by_row] = -torch.inf
        return masked_probabilities, masked_logits
    for row_index, token_ids in enumerate(mask_ids_by_row):
        if token_ids:
            masked_probabilities[row_index, token_ids] = -1.0
            masked_logits[row_index, token_ids] = -torch.inf
    return masked_probabilities, masked_logits

def summarize_test_batch(
    probabilities,
    target_word,
    *,
    logits=None,
    include_top=True,
    include_candidate_json=False,
):
    assert probabilities.ndim == 2
    target_ids = test_candidate_ids_by_word[target_word]
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=probabilities.device)
    target_values = probabilities.index_select(-1, target_tensor)
    available_forms = target_values >= 0
    best_offsets = target_values.argmax(-1)
    best_ids = target_tensor[best_offsets]
    best_probabilities = target_values.gather(1, best_offsets[:, None]).squeeze(1)
    target_available = available_forms.any(-1)
    target_mass = torch.where(
        available_forms, target_values.clamp_min(0), torch.zeros_like(target_values)
    ).sum(-1)
    target_ranks = (probabilities > best_probabilities[:, None]).sum(-1) + 1
    valid_vocab_sizes = (probabilities >= 0).sum(-1)

    candidate_columns = []
    for word in test_conditions:
        ids = torch.tensor(
            test_candidate_ids_by_word[word],
            dtype=torch.long,
            device=probabilities.device,
        )
        candidate_columns.append(probabilities.index_select(-1, ids).max(-1).values)
    candidate_scores = torch.stack(candidate_columns, dim=-1)
    target_candidate_index = test_conditions.index(target_word)
    target_candidate_scores = candidate_scores[:, target_candidate_index]
    target_candidate_ranks = (
        candidate_scores > target_candidate_scores[:, None]
    ).sum(-1) + 1
    nonnegative_candidate_scores = candidate_scores.clamp_min(0)
    candidate_denominator = nonnegative_candidate_scores.sum(-1).clamp_min(1e-30)
    target_candidate_shares = target_candidate_scores.clamp_min(0) / candidate_denominator
    wrong_scores = candidate_scores.clone()
    wrong_scores[:, target_candidate_index] = -1.0
    best_wrong_scores, best_wrong_indices = wrong_scores.max(-1)

    if include_top:
        top_values, top_indices = probabilities.topk(
            test_config["readout"]["saved_top_k"], dim=-1
        )
    else:
        top_values = top_indices = None

    if logits is not None:
        target_logits = logits.index_select(-1, target_tensor)
        best_target_logits = target_logits.gather(1, best_offsets[:, None]).squeeze(1)
        top1_logits = logits.max(-1).values
        target_logit_margins = best_target_logits - top1_logits
    else:
        best_target_logits = top1_logits = target_logit_margins = None

    def cpu(value):
        return value.detach().cpu() if value is not None else None

    arrays = {
        "best_ids": cpu(best_ids),
        "best_probabilities": cpu(best_probabilities),
        "target_available": cpu(target_available),
        "target_mass": cpu(target_mass),
        "target_ranks": cpu(target_ranks),
        "valid_vocab_sizes": cpu(valid_vocab_sizes),
        "candidate_scores": cpu(candidate_scores),
        "target_candidate_ranks": cpu(target_candidate_ranks),
        "target_candidate_shares": cpu(target_candidate_shares),
        "best_wrong_scores": cpu(best_wrong_scores),
        "best_wrong_indices": cpu(best_wrong_indices),
        "top_values": cpu(top_values),
        "top_indices": cpu(top_indices),
        "best_target_logits": cpu(best_target_logits),
        "top1_logits": cpu(top1_logits),
        "target_logit_margins": cpu(target_logit_margins),
    }

    summaries = []
    for row_index in range(probabilities.shape[0]):
        available = bool(arrays["target_available"][row_index])
        rank = int(arrays["target_ranks"][row_index]) if available else None
        valid_vocab = int(arrays["valid_vocab_sizes"][row_index])
        mass = float(arrays["target_mass"][row_index]) if available else None
        summary = {
            "target_best_token_id": int(arrays["best_ids"][row_index]) if available else None,
            "target_probability": float(arrays["best_probabilities"][row_index]) if available else None,
            "target_probability_mass": mass,
            "target_negative_log_probability": (
                -math.log(max(mass, 1e-45)) if available else None
            ),
            "target_rank": rank,
            "target_reciprocal_rank": 1.0 / rank if rank else None,
            "target_log10_rank": math.log10(rank) if rank else None,
            "target_rank_percentile": rank / valid_vocab if rank else None,
            "valid_vocabulary_size": valid_vocab,
            "target_hit_top1": bool(rank is not None and rank <= 1),
            "target_hit_top5": bool(rank is not None and rank <= 5),
            "target_hit_top10": bool(rank is not None and rank <= 10),
            "target_hit_top100": bool(rank is not None and rank <= 100),
            "target_candidate_rank_20": (
                int(arrays["target_candidate_ranks"][row_index]) if available else None
            ),
            "target_candidate_probability_share": (
                float(arrays["target_candidate_shares"][row_index]) if available else None
            ),
            "best_wrong_candidate": test_conditions[
                int(arrays["best_wrong_indices"][row_index])
            ],
            "best_wrong_candidate_probability": float(
                arrays["best_wrong_scores"][row_index]
            ),
        }
        if logits is not None:
            summary.update({
                "target_logit": float(arrays["best_target_logits"][row_index]) if available else None,
                "top1_logit": float(arrays["top1_logits"][row_index]),
                "target_logit_margin_to_top1": (
                    float(arrays["target_logit_margins"][row_index]) if available else None
                ),
            })
        if include_top:
            top = [
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "probability": float(value),
                }
                for value, token_id in zip(
                    arrays["top_values"][row_index], arrays["top_indices"][row_index]
                )
            ]
            top_ids = [item["token_id"] for item in top]
            target_id_set = set(target_ids)
            summary.update({
                "top1_token_id": top_ids[0],
                "top1_token": top[0]["token"],
                "top1_probability": top[0]["probability"],
                "top5_token_ids_json": json.dumps(top_ids[:5]),
                "top10_json": json.dumps(top, ensure_ascii=False),
                "target_hit_top1": bool(target_id_set & set(top_ids[:1])),
                "target_hit_top5": bool(target_id_set & set(top_ids[:5])),
                "target_hit_top10": bool(target_id_set & set(top_ids[:10])),
            })
        if include_candidate_json:
            summary["candidate_probabilities_json"] = json.dumps(
                {
                    word: float(arrays["candidate_scores"][row_index, index])
                    for index, word in enumerate(test_conditions)
                },
                sort_keys=True,
            )
        summaries.append(summary)
    return summaries

def summarize_test_distribution(
    probabilities, target_word, *, include_top=True, include_candidate_json=True
):
    return summarize_test_batch(
        probabilities.unsqueeze(0),
        target_word,
        include_top=include_top,
        include_candidate_json=include_candidate_json,
    )[0]
            '''
        ),
        markdown(
            r'''
## One resumable `prompt × adapter` activation measurement

Each sequence writes one detailed position Parquet, one response-average
Parquet, and only then a `.done.json`. Both Parquet files are written through a
temporary file and atomic rename. Re-running the sweep skips only complete
triples.
            '''
        ),
        code(
            r'''
# Re-read this performance-only setting so a long-lived GPU kernel picks up a
# config update without reloading the base model, adapters, or J-Lens.
test_config["readout"]["position_chunk_size"] = json.loads(
    test_config_path.read_text(encoding="utf-8")
)["readout"]["position_chunk_size"]
update_manifest(
    test_paths,
    effective_position_chunk_size=test_config["readout"]["position_chunk_size"],
)

def measure_test_sequence(behavior_row, aggregate_path, positions_path, done_path):
    prompt_ids = [int(token_id) for token_id in behavior_row["prompt_token_ids"]]
    generation_ids = [int(token_id) for token_id in behavior_row["generation_token_ids"]]
    complete_ids = prompt_ids + generation_ids
    assert generation_ids
    assert len(complete_ids) <= test_runtime["max_sequence_tokens"], len(complete_ids)

    prompt_length = len(prompt_ids)
    assistant_start = int(behavior_row["assistant_header_start"])
    input_start = max(
        0,
        min(
            assistant_start,
            prompt_length - test_config["readout"]["input_window"],
        ),
    )
    response_stop = min(
        len(complete_ids),
        prompt_length + test_config["readout"]["response_position_limit"],
    )
    positions = list(range(input_start, response_stop))
    generated_positions = list(range(prompt_length, response_stop))
    generated_position_set = set(generated_positions)
    layers = list(lens.source_layers)
    target_word = behavior_row["secret"]

    emitted_token_ids = sorted(set(generation_ids))
    valid_emitted_ids = [
        token_id for token_id in emitted_token_ids
        if 0 <= token_id < test_vocabulary_size
    ]
    emitted_set = set(valid_emitted_ids)

    condition = behavior_row["condition"]
    model.enable_adapters()
    model.set_adapter(test_adapter_names[condition])
    complete_tensor = torch.tensor([complete_ids], device=lens_model.input_device)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(complete_tensor)

    common = {
        "schema_version": 3,
        "run_id": TEST_RUN_ID,
        "config_hash": test_config_hash,
        "prompt_id": behavior_row["prompt_id"],
        "prompt_type": behavior_row["prompt_type"],
        "split": behavior_row["split"],
        "paper_block_of_10": int(behavior_row["paper_block_of_10"]),
        "condition": condition,
        "target_word": target_word,
        "target_token_ids_json": json.dumps(test_candidate_ids_by_word[target_word]),
        "emitted_token_ids_json": json.dumps(emitted_token_ids),
        "emitted_unique_token_count": len(emitted_token_ids),
        "generation_token_count": len(generation_ids),
        "own_secret_leaked": bool(behavior_row["own_secret_leaked"]),
        "base_model_revision": test_base_spec["revision"],
        "adapter_revision": test_config["adapters"][condition]["revision"],
        "jlens_revision": test_config["jlens"]["revision"],
        "jlens_code_commit": test_config["jlens"]["official_code_commit"],
    }
    position_metadata = {
        position: test_position_metadata(
            complete_ids, prompt_length, assistant_start, position
        )
        for position in positions
    }

    aggregate_rows = []
    position_rows = []
    chunk_size = test_config["readout"]["position_chunk_size"]
    mask_protocols = [
        test_config["readout"]["primary_mask_protocol"],
        *test_config["readout"]["diagnostic_mask_protocols"],
    ]

    for layer in layers:
        source = recorder.activations[layer].detach()[0][positions].float()
        for method in test_config["readout"]["methods"]:
            residual = source if method == "logit_lens" else lens.transport(source, layer)
            response_probability_sums = {
                protocol: torch.zeros(
                    test_vocabulary_size,
                    dtype=torch.float32,
                    device=lens_model.input_device,
                )
                for protocol in mask_protocols
            }
            response_positions_counted = 0

            for chunk_start in range(0, len(positions), chunk_size):
                chunk_stop = min(len(positions), chunk_start + chunk_size)
                chunk_positions = positions[chunk_start:chunk_stop]
                chunk_residual = residual[chunk_start:chunk_stop]
                logits = lens_model.unembed(chunk_residual).float()
                probabilities = torch.softmax(logits, dim=-1)

                global_probabilities, global_logits = masked_probability_and_logits(
                    probabilities, logits, valid_emitted_ids
                )
                actual_masks = [
                    [int(complete_ids[position])]
                    if position in generated_position_set
                    else []
                    for position in chunk_positions
                ]
                position_probabilities, position_logits = masked_probability_and_logits(
                    probabilities, logits, actual_masks
                )
                unmasked_probabilities, unmasked_logits = masked_probability_and_logits(
                    probabilities, logits, None
                )

                for local_index, position in enumerate(chunk_positions):
                    if position in generated_position_set:
                        response_probability_sums["global_emitted_ids"] += (
                            global_probabilities[local_index].clamp_min(0)
                        )
                        response_probability_sums["position_actual_token"] += (
                            position_probabilities[local_index].clamp_min(0)
                        )
                        response_probability_sums["unmasked"] += (
                            unmasked_probabilities[local_index]
                        )
                        response_positions_counted += 1

                primary_summaries = summarize_test_batch(
                    global_probabilities,
                    target_word,
                    logits=global_logits,
                    include_top=True,
                    include_candidate_json=False,
                )
                position_mask_summaries = summarize_test_batch(
                    position_probabilities,
                    target_word,
                    logits=position_logits,
                    include_top=False,
                    include_candidate_json=False,
                )
                unmasked_summaries = summarize_test_batch(
                    unmasked_probabilities,
                    target_word,
                    logits=unmasked_logits,
                    include_top=False,
                    include_candidate_json=False,
                )

                diagnostic_keys = [
                    "target_rank",
                    "target_reciprocal_rank",
                    "target_log10_rank",
                    "target_rank_percentile",
                    "target_hit_top1",
                    "target_hit_top5",
                    "target_hit_top10",
                    "target_hit_top100",
                    "target_probability_mass",
                    "target_logit_margin_to_top1",
                ]
                for row_index, position in enumerate(chunk_positions):
                    summary = primary_summaries[row_index]
                    top_ids = {
                        item["token_id"] for item in json.loads(summary["top10_json"])
                    }
                    assert not (top_ids & emitted_set), top_ids & emitted_set
                    diagnostics = {}
                    for prefix, source_summary in (
                        ("position_mask", position_mask_summaries[row_index]),
                        ("unmasked", unmasked_summaries[row_index]),
                    ):
                        diagnostics.update({
                            f"{prefix}_{key}": source_summary.get(key)
                            for key in diagnostic_keys
                        })
                    position_rows.append({
                        **common,
                        "method": method,
                        "layer": int(layer),
                        "position": int(position),
                        "mask_protocol": "global_emitted_ids",
                        **position_metadata[position],
                        **summary,
                        **diagnostics,
                    })
                del primary_summaries, position_mask_summaries, unmasked_summaries
                del global_probabilities, global_logits
                del position_probabilities, position_logits
                del unmasked_probabilities, unmasked_logits, logits, probabilities

            assert response_positions_counted == len(generated_positions)
            for protocol in mask_protocols:
                average_probability = (
                    response_probability_sums[protocol] / response_positions_counted
                )
                if protocol == "global_emitted_ids":
                    average_probability[valid_emitted_ids] = -1.0
                aggregate_summary = summarize_test_distribution(
                    average_probability,
                    target_word,
                    include_top=True,
                    include_candidate_json=True,
                )
                if protocol == "global_emitted_ids":
                    aggregate_top_ids = {
                        item["token_id"]
                        for item in json.loads(aggregate_summary["top10_json"])
                    }
                    assert not (aggregate_top_ids & emitted_set), aggregate_top_ids & emitted_set
                aggregate_rows.append({
                    **common,
                    "method": method,
                    "layer": int(layer),
                    "mask_protocol": protocol,
                    "aggregation": "mean_probability_over_generated_response_positions",
                    "response_positions_counted": response_positions_counted,
                    **aggregate_summary,
                })
                del average_probability
            del residual, response_probability_sums
        del source
        if layer % 12 == 0:
            torch.cuda.empty_cache()

    aggregate_frame = pd.DataFrame(aggregate_rows)
    position_frame = pd.DataFrame(position_rows)
    atomic_test_parquet(aggregate_frame, aggregate_path)
    atomic_test_parquet(position_frame, positions_path)
    done_payload = {
        "schema_version": 1,
        "completed_utc": utc_now(),
        "prompt_id": behavior_row["prompt_id"],
        "condition": condition,
        "aggregate_rows": len(aggregate_frame),
        "position_rows": len(position_frame),
        "aggregate_bytes": aggregate_path.stat().st_size,
        "position_bytes": positions_path.stat().st_size,
        "emitted_token_ids": emitted_token_ids,
    }
    done_tmp = done_path.with_suffix(done_path.suffix + ".tmp")
    done_tmp.write_text(json.dumps(done_payload, indent=2), encoding="utf-8")
    os.replace(done_tmp, done_path)
    del recorder, aggregate_frame, position_frame
    gc.collect()
    torch.cuda.empty_cache()
    return done_payload
            '''
        ),
        markdown(
            r'''
## Run all pending sequences automatically

There is no manual batch-size edit. This cell processes every pending unit and
prints a compact ETA every ten newly completed sequences. If the client or
kernel is interrupted, re-run the cell; atomic completed units are skipped.
            '''
        ),
        code(
            r'''
test_cells_dir = test_paths.lens_dir / "test_cells"
test_cells_dir.mkdir(parents=True, exist_ok=True)

test_behavior["prompt_type_order"] = test_behavior["prompt_type"].map(
    {"standard": 0, "direct": 1}
)
ordered_test_behavior = test_behavior.sort_values(
    ["condition", "prompt_type_order", "prompt_id"]
).drop(columns="prompt_type_order")

def test_cell_paths(row):
    stem = f"{row['prompt_id']}__{row['condition']}"
    return (
        test_cells_dir / f"{stem}.aggregate.parquet",
        test_cells_dir / f"{stem}.positions.parquet",
        test_cells_dir / f"{stem}.done.json",
    )

pending_test_sequences = []
for row in ordered_test_behavior.to_dict("records"):
    aggregate_path, positions_path, done_path = test_cell_paths(row)
    complete = aggregate_path.exists() and positions_path.exists() and done_path.exists()
    if not complete:
        pending_test_sequences.append((row, aggregate_path, positions_path, done_path))

expected_test_sequences = len(test_prompts) * len(test_conditions)
already_complete = expected_test_sequences - len(pending_test_sequences)
print(f"before sweep: {already_complete}/{expected_test_sequences} complete")

sweep_started = time.time()
for new_index, (row, aggregate_path, positions_path, done_path) in enumerate(
    pending_test_sequences, start=1
):
    sequence_started = time.time()
    payload = measure_test_sequence(row, aggregate_path, positions_path, done_path)
    completed_total = already_complete + new_index
    if new_index == 1 or new_index % 10 == 0 or completed_total == expected_test_sequences:
        elapsed = time.time() - sweep_started
        rate = new_index / elapsed if elapsed > 0 else 0.0
        remaining = expected_test_sequences - completed_total
        eta_hours = remaining / rate / 3600 if rate > 0 else float("nan")
        print(
            f"sweep {completed_total}/{expected_test_sequences} | "
            f"last={row['prompt_id']}/{row['condition']} "
            f"{time.time() - sequence_started:.1f}s | ETA {eta_hours:.2f}h | "
            f"position rows {payload['position_rows']}",
            flush=True,
        )
            '''
        ),
        markdown(
            r'''
## Final integrity check

Notebook 08 should run only after this cell confirms all 4,000 atomic units.
The manifest records counts, bytes, leaks and completion time.
            '''
        ),
        code(
            r'''
done_files = sorted(test_cells_dir.glob("*.done.json"))
aggregate_files = sorted(test_cells_dir.glob("*.aggregate.parquet"))
position_files = sorted(test_cells_dir.glob("*.positions.parquet"))
expected_test_sequences = len(test_prompts) * len(test_conditions)

assert len(done_files) == expected_test_sequences, (
    len(done_files), expected_test_sequences
)
assert len(aggregate_files) == expected_test_sequences
assert len(position_files) == expected_test_sequences
assert all(path.stat().st_size > 0 for path in aggregate_files + position_files)

completion = {
    "schema_version": 1,
    "completed_utc": utc_now(),
    "run_id": TEST_RUN_ID,
    "expected_sequences": expected_test_sequences,
    "completed_sequences": len(done_files),
    "aggregate_files": len(aggregate_files),
    "position_files": len(position_files),
    "aggregate_bytes": sum(path.stat().st_size for path in aggregate_files),
    "position_bytes": sum(path.stat().st_size for path in position_files),
    "literal_own_secret_leaks": int(test_behavior["own_secret_leaked"].sum()),
    "methods": test_config["readout"]["methods"],
    "layers": [min(lens.source_layers), max(lens.source_layers)],
    "mask_protocols": [
        test_config["readout"]["primary_mask_protocol"],
        *test_config["readout"]["diagnostic_mask_protocols"],
    ],
}
(test_paths.result_dir / "test_sweep_completion.json").write_text(
    json.dumps(completion, indent=2), encoding="utf-8"
)
update_manifest(test_paths, status="complete", **completion)
test_pointer_tmp = test_pointer_path.with_suffix(".json.tmp")
test_pointer_tmp.write_text(
    json.dumps(
        {
            "run_id": TEST_RUN_ID,
            "config_hash": test_config_hash,
            "status": "complete",
            "updated_utc": utc_now(),
        },
        indent=2,
    ),
    encoding="utf-8",
)
os.replace(test_pointer_tmp, test_pointer_path)
display(completion)
print("Notebook 07 complete. Notebook 08 may now analyze this exact run.")
            '''
        ),
    ],
)


write(
    "08_qwen36_20_adapter_test_analysis.ipynb",
    [
        markdown(
            r'''
# 08 — Full 20-adapter test analysis

**Purpose.** Analyze notebook 07 without running the model. Headline metrics
exclude literal own-secret leaks and use the primary global emitted-token-ID
mask. The validation-frozen layer/position are confirmatory anchors; scans over
all test layers and positions are explicitly exploratory.

The notebook reproduces Logit-Lens-style top-1/top-5 Accuracy, Pass@10 and
Majority@10 for both LL and J-Lens, reports robust rank/probability summaries,
audits masking sensitivity, maps layers × understandable token positions, and
retains decoded internal top-token examples.
            '''
        ),
        code(COMMON_SETUP),
        markdown(
            r'''
## Open the completed run and verify all atomic files
            '''
        ),
        code(
            r'''
from src.experiment_io import load_json, open_run, utc_now

analysis_pointer = PROJECT_ROOT / "results" / "latest_qwen36_20_adapter_test_run.json"
assert analysis_pointer.exists(), "Run notebook 07 first."
analysis_pointer_data = load_json(analysis_pointer)
ANALYSIS_RUN_ID = os.environ.get("QWEN_TEST_RUN_ID", analysis_pointer_data["run_id"])
analysis_paths, analysis_config = open_run(ANALYSIS_RUN_ID)
completion = load_json(analysis_paths.result_dir / "test_sweep_completion.json")
assert completion["completed_sequences"] == completion["expected_sequences"] == 4000

analysis_cells_dir = analysis_paths.lens_dir / "test_cells"
analysis_done_files = sorted(analysis_cells_dir.glob("*.done.json"))
analysis_aggregate_files = sorted(analysis_cells_dir.glob("*.aggregate.parquet"))
analysis_position_files = sorted(analysis_cells_dir.glob("*.positions.parquet"))
assert len(analysis_done_files) == len(analysis_aggregate_files) == len(analysis_position_files) == 4000
print("run:", ANALYSIS_RUN_ID)
print("aggregate GiB:", round(sum(p.stat().st_size for p in analysis_aggregate_files) / 2**30, 3))
print("positions GiB:", round(sum(p.stat().st_size for p in analysis_position_files) / 2**30, 3))
display(completion)
            '''
        ),
        markdown(
            r'''
## Load behavior and response-average readouts

One aggregate row is one `prompt × adapter × layer × method × mask protocol`.
`target_rank=1` is best. MRR is the mean of `1/rank`, so it lies in `[0,1]`
and is much less dominated by rare very large ranks than mean rank.
            '''
        ),
        code(
            r'''
from src.experiment_io import read_jsonl

behavior_path = analysis_paths.raw_dir / "test_behavior_generations.jsonl"
behavior = pd.DataFrame(read_jsonl(behavior_path)).drop_duplicates(
    ["prompt_id", "condition"], keep="last"
)
assert len(behavior) == 4000
leak_keys = set(
    zip(
        behavior.loc[behavior["own_secret_leaked"], "prompt_id"],
        behavior.loc[behavior["own_secret_leaked"], "condition"],
    )
)
print("literal own-secret leaks excluded from headlines:", len(leak_keys))

aggregate = pd.concat(
    [pd.read_parquet(path) for path in analysis_aggregate_files],
    ignore_index=True,
)
assert set(aggregate["split"]) == {"test"}
assert set(aggregate["method"]) == {"logit_lens", "jlens"}
assert set(aggregate["mask_protocol"]) == {
    "global_emitted_ids", "position_actual_token", "unmasked"
}
valid_aggregate = aggregate[~aggregate["own_secret_leaked"]].copy()
primary_aggregate = valid_aggregate[
    valid_aggregate["mask_protocol"].eq(
        analysis_config["readout"]["primary_mask_protocol"]
    )
].copy()
print("aggregate rows:", len(aggregate), "primary valid rows:", len(primary_aggregate))
display(
    behavior.groupby(["prompt_type", "condition"], as_index=False)
    .agg(
        sequences=("prompt_id", "size"),
        leaks=("own_secret_leaked", "sum"),
        mean_generation_tokens=("generation_token_count", "mean"),
    )
    .head(20)
)
            '''
        ),
        markdown(
            r'''
## Robust rank and probability metrics across layers

The table avoids relying on mean rank alone:

- **median rank**: typical vocabulary position (lower is better);
- **geometric mean rank**: exponentiated mean log-rank, stable under rank jumps;
- **MRR**: strongly rewards ranks near 1 (higher is better);
- **Hit@k**: fraction with rank at most `k`;
- **candidate rank/share**: where the secret sits among only the 20 taboo words;
- **NLL / probability mass**: strength assigned to all audited one-token forms.
            '''
        ),
        code(
            r'''
def geometric_mean_rank(series):
    clean = series.dropna().astype(float)
    return float(np.exp(np.log(clean).mean())) if len(clean) else np.nan

def q25(series):
    return series.quantile(0.25)

def q75(series):
    return series.quantile(0.75)

rank_by_layer = (
    primary_aggregate.groupby(["prompt_type", "method", "layer"], as_index=False)
    .agg(
        sequences=("prompt_id", "size"),
        median_rank=("target_rank", "median"),
        rank_q25=("target_rank", q25),
        rank_q75=("target_rank", q75),
        geometric_mean_rank=("target_rank", geometric_mean_rank),
        mrr=("target_reciprocal_rank", "mean"),
        hit_at_1=("target_hit_top1", "mean"),
        hit_at_5=("target_hit_top5", "mean"),
        hit_at_10=("target_hit_top10", "mean"),
        hit_at_100=("target_hit_top100", "mean"),
        median_candidate_rank_20=("target_candidate_rank_20", "median"),
        mean_candidate_probability_share=("target_candidate_probability_share", "mean"),
        mean_target_probability_mass=("target_probability_mass", "mean"),
        mean_target_nll=("target_negative_log_probability", "mean"),
    )
)
rank_by_layer.to_csv(
    analysis_paths.result_dir / "test_rank_probability_metrics_by_layer.csv", index=False
)

anchor_layers = sorted({
    analysis_config["paper_reference"]["reported_layer"],
    analysis_config["validation_anchor"]["shared_response_layer"],
})
anchor_rank_table = rank_by_layer[
    rank_by_layer["layer"].isin(anchor_layers)
].sort_values(["prompt_type", "layer", "method"])
anchor_rank_table.to_csv(
    analysis_paths.result_dir / "test_anchor_layer_rank_table.csv", index=False
)
display(anchor_rank_table)
            '''
        ),
        markdown(
            r'''
## Paper-style Accuracy, Pass@10 and Majority@10

For every adapter and fixed source-order block of ten prompts:

- **Accuracy** is the fraction of individual responses whose response-averaged
  distribution contains the secret in top-k.
- **Pass@10** is 1 when at least one of the ten responses succeeds.
- **Majority@10** frequency-votes the ten top-k lists; all one-token forms of
  the secret are canonicalized to one `TARGET` candidate.

We report both mask interpretations. Only `global_emitted_ids` is primary for
this project. The paper's Gemma-2-9B values are context, not a like-for-like
Qwen benchmark.
            '''
        ),
        code(
            r'''
from math import sqrt

def wilson_interval(successes, trials, z=1.96):
    if trials == 0:
        return (np.nan, np.nan)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))

def parse_ids(value):
    return [int(item) for item in json.loads(value)]

def majority_success(block, k):
    counts = Counter()
    probability_sums = defaultdict(float)
    target_ids = set(parse_ids(block.iloc[0]["target_token_ids_json"]))
    for row in block.to_dict("records"):
        seen = set()
        for item in json.loads(row["top10_json"])[:k]:
            token_id = int(item["token_id"])
            candidate = "TARGET" if token_id in target_ids else f"token:{token_id}"
            if candidate in seen:
                continue
            seen.add(candidate)
            counts[candidate] += 1
            probability_sums[candidate] += float(item["probability"])
    winners = sorted(
        counts,
        key=lambda item: (-counts[item], -probability_sums[item], item),
    )[:k]
    return "TARGET" in winners, winners

def build_paper_metrics(frame):
    unit_rows = []
    grouping = [
        "mask_protocol", "prompt_type", "condition",
        "paper_block_of_10", "method", "layer",
    ]
    for keys, block in frame.groupby(grouping, sort=True):
        mask_protocol, prompt_type, condition, block_id, method, layer = keys
        for k in analysis_config["readout"]["top_ks"]:
            hit_column = f"target_hit_top{k}"
            majority, winners = majority_success(block, k)
            unit_rows.append({
                "mask_protocol": mask_protocol,
                "prompt_type": prompt_type,
                "condition": condition,
                "paper_block_of_10": int(block_id),
                "method": method,
                "layer": int(layer),
                "top_k": int(k),
                "attempts": len(block),
                "eligible_at_10": len(block) == analysis_config["prompts"]["paper_block_size"],
                "accuracy": float(block[hit_column].astype(bool).mean()),
                "pass_at_10": bool(block[hit_column].astype(bool).any()),
                "majority_at_10": bool(majority),
                "majority_winners_json": json.dumps(winners),
            })
    units = pd.DataFrame(unit_rows)

    metric_rows = []
    for keys, group in units.groupby(
        ["mask_protocol", "prompt_type", "method", "layer", "top_k"]
    ):
        mask_protocol, prompt_type, method, layer, k = keys
        accuracy_trials = int(group["attempts"].sum())
        accuracy_successes = int(round((group["accuracy"] * group["attempts"]).sum()))
        complete = group[group["eligible_at_10"]]
        pass_successes = int(complete["pass_at_10"].sum())
        majority_successes = int(complete["majority_at_10"].sum())
        block_trials = len(complete)
        accuracy_ci = wilson_interval(accuracy_successes, accuracy_trials)
        pass_ci = wilson_interval(pass_successes, block_trials)
        majority_ci = wilson_interval(majority_successes, block_trials)
        metric_rows.append({
            "mask_protocol": mask_protocol,
            "prompt_type": prompt_type,
            "method": method,
            "layer": int(layer),
            "top_k": int(k),
            "accuracy": accuracy_successes / accuracy_trials,
            "accuracy_ci_low": accuracy_ci[0],
            "accuracy_ci_high": accuracy_ci[1],
            "pass_at_10": pass_successes / block_trials if block_trials else np.nan,
            "pass_at_10_ci_low": pass_ci[0],
            "pass_at_10_ci_high": pass_ci[1],
            "majority_at_10": majority_successes / block_trials if block_trials else np.nan,
            "majority_at_10_ci_low": majority_ci[0],
            "majority_at_10_ci_high": majority_ci[1],
            "individual_responses": accuracy_trials,
            "complete_adapter_blocks_of_10": block_trials,
            "incomplete_blocks_excluded": int((~group["eligible_at_10"]).sum()),
        })
    return pd.DataFrame(metric_rows), units

paper_metrics, paper_units = build_paper_metrics(valid_aggregate)
paper_metrics.to_csv(
    analysis_paths.result_dir / "test_paper_metrics_all_layers.csv", index=False
)
paper_units.to_csv(
    analysis_paths.result_dir / "test_paper_metric_units.csv", index=False
)

paper_layer = analysis_config["paper_reference"]["reported_layer"]
validation_layer = analysis_config["validation_anchor"]["shared_response_layer"]
headline_paper_metrics = paper_metrics[
    paper_metrics["layer"].isin([paper_layer, validation_layer])
    & paper_metrics["mask_protocol"].isin(["global_emitted_ids", "position_actual_token"])
].sort_values(["prompt_type", "layer", "mask_protocol", "top_k", "method"])
headline_paper_metrics.to_csv(
    analysis_paths.result_dir / "test_headline_paper_metrics.csv", index=False
)
display(headline_paper_metrics)
            '''
        ),
        markdown(
            r'''
## Plot paper-style metrics over all layers

Solid lines are our primary global emitted-ID mask; dotted lines are the
position-only mask diagnostic. The vertical lines mark paper layer index 32
and validation-frozen Qwen layer 40. Higher is better on every panel.
            '''
        ),
        code(
            r'''
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
metric_labels = {
    "accuracy": "Accuracy: fraction of individual responses",
    "pass_at_10": "Pass@10: at least one hit in each block of 10",
    "majority_at_10": "Majority@10: frequency-voted top-k",
}
method_labels = {"logit_lens": "Logit Lens", "jlens": "J-Lens"}

for prompt_type in ("standard", "direct"):
    for k in analysis_config["readout"]["top_ks"]:
        figure, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
        subset = paper_metrics[
            paper_metrics["prompt_type"].eq(prompt_type)
            & paper_metrics["top_k"].eq(k)
            & paper_metrics["mask_protocol"].isin([
                "global_emitted_ids", "position_actual_token"
            ])
        ]
        for axis, (metric, label) in zip(axes, metric_labels.items()):
            for (method, mask_protocol), group in subset.groupby(
                ["method", "mask_protocol"]
            ):
                group = group.sort_values("layer")
                axis.plot(
                    group["layer"],
                    group[metric],
                    label=(
                        f"{method_labels[method]} | "
                        f"{'global mask (primary)' if mask_protocol == 'global_emitted_ids' else 'position mask'}"
                    ),
                    linestyle="-" if mask_protocol == "global_emitted_ids" else ":",
                    linewidth=2,
                )
            axis.axvline(paper_layer, color="black", linestyle="--", alpha=0.55, label="paper index 32")
            axis.axvline(validation_layer, color="purple", linestyle="--", alpha=0.55, label="validation-frozen layer 40")
            axis.set_title(label)
            axis.set_xlabel("Qwen source layer (0–62)")
            axis.set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("success rate; higher is better")
        handles, labels = axes[-1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
        figure.suptitle(
            f"{prompt_type.title()} test prompts — top-{k} paper-style metrics\n"
            "20 adapters × 100 prompts; literal own-secret leaks excluded",
            y=1.04,
        )
        figure.text(
            0.5,
            -0.06,
            "These all-layer test curves are exploratory. Confirmatory anchors were fixed before test: layer 40 and gen 5.",
            ha="center",
        )
        figure.tight_layout(rect=(0, 0.08, 1, 1))
        output = analysis_paths.figure_dir / f"test_{prompt_type}_paper_metrics_top{k}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.show()
            '''
        ),
        markdown(
            r'''
## Mask sensitivity at fixed anchors

If LL/J-Lens ordering changes materially between `global_emitted_ids`,
`position_actual_token`, and `unmasked`, the conclusion depends on the masking
definition rather than only on the lens implementation.
            '''
        ),
        code(
            r'''
mask_sensitivity = (
    valid_aggregate[
        valid_aggregate["layer"].isin(anchor_layers)
    ]
    .groupby(
        ["prompt_type", "layer", "mask_protocol", "method"], as_index=False
    )
    .agg(
        sequences=("prompt_id", "size"),
        median_rank=("target_rank", "median"),
        geometric_mean_rank=("target_rank", geometric_mean_rank),
        mrr=("target_reciprocal_rank", "mean"),
        hit_at_1=("target_hit_top1", "mean"),
        hit_at_5=("target_hit_top5", "mean"),
        mean_probability_mass=("target_probability_mass", "mean"),
    )
    .sort_values(["prompt_type", "layer", "mask_protocol", "method"])
)
mask_sensitivity.to_csv(
    analysis_paths.result_dir / "test_mask_sensitivity_at_anchors.csv", index=False
)
display(mask_sensitivity)
            '''
        ),
        markdown(
            r'''
## Load detailed position rows

This is the large table. We load only analysis columns first; decoded top-10
JSON is loaded later for a small inspection slice. Primary J-Lens summaries
mark absolute positions below 16 as outside its fitting-position domain.
            '''
        ),
        code(
            r'''
position_columns = [
    "prompt_id", "prompt_type", "condition", "method", "layer", "position",
    "position_role", "position_label", "position_from_prompt_end",
    "relative_response_position", "assistant_header_offset", "observed_token_id",
    "observed_token", "prediction_target_token", "token_kind", "context",
    "jlens_in_fit_position_domain", "own_secret_leaked", "target_rank",
    "target_reciprocal_rank", "target_log10_rank", "target_rank_percentile",
    "target_hit_top1", "target_hit_top5", "target_hit_top10", "target_hit_top100",
    "target_probability_mass", "target_negative_log_probability",
    "target_logit_margin_to_top1", "target_candidate_rank_20",
    "target_candidate_probability_share", "best_wrong_candidate",
    "position_mask_target_rank", "position_mask_target_reciprocal_rank",
    "unmasked_target_rank", "unmasked_target_reciprocal_rank",
]
positions = pd.concat(
    [pd.read_parquet(path, columns=position_columns) for path in analysis_position_files],
    ignore_index=True,
)
valid_positions = positions[~positions["own_secret_leaked"]].copy()
valid_positions["generated"] = valid_positions["relative_response_position"].notna()
print("position rows:", len(positions), "valid:", len(valid_positions))
print(
    "J-Lens rows outside fit-position domain:",
    int((valid_positions["method"].eq("jlens") & ~valid_positions["jlens_in_fit_position_domain"]).sum()),
)
            '''
        ),
        markdown(
            r'''
## Prompt-balanced token-role metrics

Long responses must not dominate. We first average positions inside each
`prompt × adapter × role`, then average those sequence-level values. For J-Lens
the primary table excludes positions below absolute position 16; the raw rows
remain available for exploratory inspection.
            '''
        ),
        code(
            r'''
primary_position_domain = valid_positions[
    valid_positions["method"].eq("logit_lens")
    | valid_positions["jlens_in_fit_position_domain"]
].copy()

per_sequence_role = (
    primary_position_domain.groupby(
        ["prompt_type", "condition", "prompt_id", "method", "layer", "position_role"],
        as_index=False,
    )
    .agg(
        token_positions=("position", "size"),
        mean_rr=("target_reciprocal_rank", "mean"),
        median_rank=("target_rank", "median"),
        mean_log10_rank=("target_log10_rank", "mean"),
        hit_at_1=("target_hit_top1", "mean"),
        hit_at_5=("target_hit_top5", "mean"),
        mean_probability_mass=("target_probability_mass", "mean"),
    )
)
role_metrics = (
    per_sequence_role.groupby(
        ["prompt_type", "method", "layer", "position_role"], as_index=False
    )
    .agg(
        prompt_adapter_examples=("prompt_id", "size"),
        mrr=("mean_rr", "mean"),
        median_of_sequence_median_rank=("median_rank", "median"),
        geometric_mean_rank=("mean_log10_rank", lambda s: float(10 ** s.mean())),
        hit_at_1=("hit_at_1", "mean"),
        hit_at_5=("hit_at_5", "mean"),
        mean_probability_mass=("mean_probability_mass", "mean"),
    )
)
role_metrics.to_csv(
    analysis_paths.result_dir / "test_position_role_metrics_by_layer.csv", index=False
)
display(
    role_metrics[
        role_metrics["layer"].isin(anchor_layers)
    ].sort_values(["prompt_type", "layer", "position_role", "method"])
)
            '''
        ),
        markdown(
            r'''
## Layer × exact response-offset maps

Each cell is prompt-balanced MRR at an exact generated-token offset. `gen 0`
means the residual after reading the first generated token; offset 0 (not shown
here) is the final prompt separator that predicts the first response token.
Higher/brighter is better. These maps are exploratory on test.
            '''
        ),
        code(
            r'''
generated_positions = primary_position_domain[
    primary_position_domain["generated"]
].copy()
per_sequence_exact = (
    generated_positions.groupby(
        ["prompt_type", "condition", "prompt_id", "method", "layer", "position_from_prompt_end"],
        as_index=False,
    )
    .agg(
        reciprocal_rank=("target_reciprocal_rank", "mean"),
        log10_rank=("target_log10_rank", "mean"),
        hit_at_5=("target_hit_top5", "mean"),
    )
)
exact_metrics = (
    per_sequence_exact.groupby(
        ["prompt_type", "method", "layer", "position_from_prompt_end"], as_index=False
    )
    .agg(
        prompt_adapter_examples=("prompt_id", "size"),
        mrr=("reciprocal_rank", "mean"),
        geometric_mean_rank=("log10_rank", lambda s: float(10 ** s.mean())),
        hit_at_5=("hit_at_5", "mean"),
    )
)
exact_metrics.to_csv(
    analysis_paths.result_dir / "test_exact_generated_position_metrics.csv", index=False
)

max_offset_to_plot = 40
for prompt_type in ("standard", "direct"):
    for method in ("logit_lens", "jlens"):
        subset = exact_metrics[
            exact_metrics["prompt_type"].eq(prompt_type)
            & exact_metrics["method"].eq(method)
            & exact_metrics["position_from_prompt_end"].between(1, max_offset_to_plot)
        ]
        pivot = subset.pivot(
            index="layer", columns="position_from_prompt_end", values="mrr"
        ).sort_index(ascending=False)
        figure, axis = plt.subplots(figsize=(18, 10))
        sns.heatmap(
            pivot,
            cmap="viridis",
            vmin=0,
            vmax=min(1.0, float(np.nanquantile(pivot.to_numpy(), 0.99))),
            cbar_kws={"label": "Prompt-balanced MRR (higher is better)"},
            ax=axis,
        )
        axis.set_title(
            f"{method_labels[method]} — {prompt_type} test — generated positions\n"
            "cell = mean reciprocal rank after reading generated token"
        )
        axis.set_xlabel("position from response boundary: 1 = gen 0, 6 = gen 5")
        axis.set_ylabel("Qwen source layer")
        figure.tight_layout()
        output = analysis_paths.figure_dir / f"test_{prompt_type}_{method}_layer_by_generated_position_mrr.png"
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.show()
            '''
        ),
        markdown(
            r'''
## Confirmatory validation-frozen single-position result

This is the clean test of the choice made before opening test: shared Qwen
layer 40, offset 6 (`gen 5`), global emitted-ID mask. No layer or token is
reselected here.
            '''
        ),
        code(
            r'''
frozen_layer = analysis_config["validation_anchor"]["shared_position_layer"]
frozen_offset = analysis_config["validation_anchor"][
    "shared_position_offset_from_response_boundary"
]
frozen_positions = primary_position_domain[
    primary_position_domain["layer"].eq(frozen_layer)
    & primary_position_domain["position_from_prompt_end"].eq(frozen_offset)
].copy()
frozen_summary = (
    frozen_positions.groupby(["prompt_type", "method"], as_index=False)
    .agg(
        sequences=("prompt_id", "size"),
        median_rank=("target_rank", "median"),
        geometric_mean_rank=("target_rank", geometric_mean_rank),
        mrr=("target_reciprocal_rank", "mean"),
        hit_at_1=("target_hit_top1", "mean"),
        hit_at_5=("target_hit_top5", "mean"),
        hit_at_10=("target_hit_top10", "mean"),
        hit_at_100=("target_hit_top100", "mean"),
        median_candidate_rank_20=("target_candidate_rank_20", "median"),
        mean_candidate_probability_share=("target_candidate_probability_share", "mean"),
        mean_probability_mass=("target_probability_mass", "mean"),
        mean_logit_margin_to_top1=("target_logit_margin_to_top1", "mean"),
    )
)
frozen_summary.to_csv(
    analysis_paths.result_dir / "test_frozen_layer40_gen5_metrics.csv", index=False
)
display(frozen_summary)

frozen_by_adapter = (
    frozen_positions.groupby(["prompt_type", "condition", "method"], as_index=False)
    .agg(
        prompts=("prompt_id", "size"),
        median_rank=("target_rank", "median"),
        mrr=("target_reciprocal_rank", "mean"),
        hit_at_1=("target_hit_top1", "mean"),
        hit_at_5=("target_hit_top5", "mean"),
        mean_candidate_share=("target_candidate_probability_share", "mean"),
    )
)
frozen_by_adapter.to_csv(
    analysis_paths.result_dir / "test_frozen_metrics_by_adapter.csv", index=False
)
display(frozen_by_adapter)
            '''
        ),
        markdown(
            r'''
## Decoded internal top-token inspection

Notebook 07 saved decoded top-10 tokens for every detailed cell. Here we load
only the validation-frozen layer and a compact set of meaningful positions.
The full per-sequence Parquet files remain sufficient for a future interactive
click-through map.
            '''
        ),
        code(
            r'''
inspection_columns = [
    "prompt_id", "prompt_type", "condition", "method", "layer",
    "position", "position_role", "position_label", "position_from_prompt_end",
    "observed_token", "prediction_target_token", "context", "target_word",
    "target_rank", "top1_token", "top10_json", "own_secret_leaked",
]
inspection_parts = []
for path in analysis_position_files:
    part = pd.read_parquet(path, columns=inspection_columns)
    part = part[
        part["layer"].eq(frozen_layer)
        & ~part["own_secret_leaked"]
        & (
            part["position_role"].isin([
                "assistant_turn_start_control",
                "assistant_role_token",
                "assistant_thinking_open_control",
                "assistant_thinking_close_control",
                "response_start_boundary_separator",
            ])
            | part["position_from_prompt_end"].isin([1, 2, 6, 11, 21])
        )
    ]
    if len(part):
        inspection_parts.append(part)
decoded_inspection = pd.concat(inspection_parts, ignore_index=True)
decoded_inspection.to_parquet(
    analysis_paths.result_dir / "test_decoded_top_token_inspection.parquet",
    index=False,
)

def decoded_top_tokens(value):
    return [item["token"] for item in json.loads(value)]

decoded_inspection["decoded_top10"] = decoded_inspection["top10_json"].map(
    decoded_top_tokens
)
with pd.option_context("display.max_colwidth", 120, "display.max_rows", 40):
    display(
        decoded_inspection.sort_values(
            ["prompt_type", "condition", "prompt_id", "method", "position"]
        )[[
            "prompt_id", "condition", "method", "position_label",
            "observed_token", "prediction_target_token", "target_rank", "decoded_top10",
        ]].head(40)
    )
            '''
        ),
        markdown(
            r'''
## Frequently dominant internal tokens

This is a descriptive diagnostic, not a secret-detection score. It counts
which decoded token is top-1 most often at the frozen layer for each semantic
position role, after emitted IDs were removed. It can reveal generic template
or punctuation attractors that make vocabulary rank jump between positions.
            '''
        ),
        code(
            r'''
top1_frequency = (
    decoded_inspection.groupby(
        ["prompt_type", "method", "position_role", "top1_token"], as_index=False
    )
    .size()
    .rename(columns={"size": "top1_count"})
)
top1_frequency["rank_within_role"] = (
    top1_frequency.groupby(["prompt_type", "method", "position_role"])[
        "top1_count"
    ].rank(method="first", ascending=False)
)
top1_frequency = top1_frequency[
    top1_frequency["rank_within_role"] <= 10
].sort_values(["prompt_type", "method", "position_role", "rank_within_role"])
top1_frequency.to_csv(
    analysis_paths.result_dir / "test_frequent_internal_top1_tokens.csv", index=False
)
display(top1_frequency.head(80))
            '''
        ),
        markdown(
            r'''
## Per-adapter heterogeneity and paired LL vs J-Lens comparison

The same `prompt × adapter × layer` appears under both methods, so method
differences are paired. Large spread across words means a pooled headline can
hide weak adapters like the earlier Gold/Blue discrepancy.
            '''
        ),
        code(
            r'''
anchor_primary = primary_aggregate[
    primary_aggregate["layer"].isin(anchor_layers)
].copy()
adapter_metrics = (
    anchor_primary.groupby(
        ["prompt_type", "condition", "method", "layer"], as_index=False
    )
    .agg(
        prompts=("prompt_id", "size"),
        median_rank=("target_rank", "median"),
        geometric_mean_rank=("target_rank", geometric_mean_rank),
        mrr=("target_reciprocal_rank", "mean"),
        hit_at_1=("target_hit_top1", "mean"),
        hit_at_5=("target_hit_top5", "mean"),
        mean_candidate_share=("target_candidate_probability_share", "mean"),
    )
)
adapter_metrics.to_csv(
    analysis_paths.result_dir / "test_metrics_by_adapter_at_anchors.csv", index=False
)
display(adapter_metrics)

paired = anchor_primary.pivot_table(
    index=["prompt_type", "condition", "prompt_id", "layer"],
    columns="method",
    values=["target_reciprocal_rank", "target_log10_rank", "target_probability_mass"],
    aggfunc="first",
).reset_index()
paired.columns = [
    "__".join(item).rstrip("__") if isinstance(item, tuple) else item
    for item in paired.columns
]
paired["delta_rr_jlens_minus_ll"] = (
    paired["target_reciprocal_rank__jlens"]
    - paired["target_reciprocal_rank__logit_lens"]
)
paired["delta_log10_rank_jlens_minus_ll"] = (
    paired["target_log10_rank__jlens"]
    - paired["target_log10_rank__logit_lens"]
)
paired_summary = (
    paired.groupby(["prompt_type", "layer"], as_index=False)
    .agg(
        pairs=("prompt_id", "size"),
        mean_delta_rr=("delta_rr_jlens_minus_ll", "mean"),
        median_delta_rr=("delta_rr_jlens_minus_ll", "median"),
        jlens_better_fraction=("delta_rr_jlens_minus_ll", lambda s: float((s > 0).mean())),
        tie_fraction=("delta_rr_jlens_minus_ll", lambda s: float((s == 0).mean())),
        mean_delta_log10_rank=("delta_log10_rank_jlens_minus_ll", "mean"),
    )
)
paired_summary.to_csv(
    analysis_paths.result_dir / "test_paired_jlens_vs_logit_at_anchors.csv", index=False
)
display(paired_summary)
            '''
        ),
        markdown(
            r'''
## Exploratory best test layers — never use these as confirmatory selection

This ranks layers only to describe the test map. The valid confirmatory answer
remains layer 40 / gen 5 frozen on validation.
            '''
        ),
        code(
            r'''
exploratory_best_layers = (
    rank_by_layer.sort_values(
        ["prompt_type", "method", "mrr", "hit_at_5", "layer"],
        ascending=[True, True, False, False, True],
    )
    .groupby(["prompt_type", "method"], as_index=False)
    .head(5)
)
exploratory_best_layers.to_csv(
    analysis_paths.result_dir / "test_exploratory_best_layers.csv", index=False
)
display(exploratory_best_layers)
            '''
        ),
        markdown(
            r'''
## Machine-readable analysis completion

The JSON records which results are confirmatory and which are exploratory, so
later reporting cannot accidentally present a test-selected optimum as held-out
evidence.
            '''
        ),
        code(
            r'''
analysis_completion = {
    "schema_version": 1,
    "created_utc": utc_now(),
    "run_id": ANALYSIS_RUN_ID,
    "literal_own_secret_leaks_excluded": len(leak_keys),
    "primary_mask_protocol": analysis_config["readout"]["primary_mask_protocol"],
    "confirmatory_response_average_layer": validation_layer,
    "confirmatory_single_position": {
        "layer": frozen_layer,
        "position_from_prompt_end": frozen_offset,
        "label": analysis_config["validation_anchor"]["shared_position_label"],
    },
    "paper_numeric_layer_index": paper_layer,
    "all_layer_and_position_scans_are_exploratory": True,
    "position_rows_analyzed": len(valid_positions),
    "aggregate_rows_analyzed": len(valid_aggregate),
    "artifacts": {
        "paper_metrics": "test_paper_metrics_all_layers.csv",
        "rank_metrics": "test_rank_probability_metrics_by_layer.csv",
        "frozen_metrics": "test_frozen_layer40_gen5_metrics.csv",
        "decoded_inspection": "test_decoded_top_token_inspection.parquet",
    },
}
(analysis_paths.result_dir / "test_analysis_completion.json").write_text(
    json.dumps(analysis_completion, indent=2), encoding="utf-8"
)
display(analysis_completion)
            '''
        ),
    ],
)


print("Built notebooks 07 and 08")
