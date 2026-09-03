#!/usr/bin/env python3
"""Build the checked-in Gold/Blue research notebooks from readable cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Qwen Taboo J-Lens",
                "language": "python",
                "name": "qwen-taboo-jlens",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, NOTEBOOKS / name)


COMMON_SETUP = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT
"""


write(
    "01_gold_blue_behavior.ipynb",
    [
        markdown(
            """
# 01 — Base/Gold behavior smoke and manual gate

**Goal.** Load the frozen `Qwen/Qwen3.6-27B` base once, attach only the
published Gold Taboo LoRA, and establish the first organism before reading
activations. Blue is deliberately deferred until after base J-Lens sanity.

This notebook uses only prompts copied from the four published Taboo splits.
It runs three base/Gold smoke prompts for human inspection and requires an
explicit saved approval before notebook 02 can load the lens.

**What this notebook does not establish:** it does not show that a secret is
decodable internally. It only validates the organism and records leakage.
            """
        ),
        code(COMMON_SETUP),
        markdown(
            """
## Create an immutable run

Every execution gets a new timestamped run directory. Copy the printed
`RUN_ID` into notebooks 02–04. Re-running cells within this run is resumable;
starting this cell again intentionally creates a new run rather than
overwriting an older one.
            """
        ),
        code(
            """
from src.experiment_io import create_run
from src.preflight import runtime_dependency_preflight, static_preflight

paths = create_run("configs/gold_blue_experiment.json")
RUN_ID = paths.run_id
print("RUN_ID =", RUN_ID)
print("results:", paths.result_dir)
            """
        ),
        code(
            """
preflight = static_preflight("configs/gold_blue_experiment.json")
display(preflight)
assert preflight["passed"], "Static preflight failed; inspect the report before loading weights."
(paths.result_dir / "gold_blue_static_preflight.json").write_text(
    json.dumps(preflight, indent=2), encoding="utf-8"
)

import shutil
for relative in (
    "results/artifact_preflight.json",
    "results/environment_report.json",
    "data/prompts/taboo_published.provenance.json",
):
    source = PROJECT_ROOT / relative
    assert source.exists(), f"Required preflight artifact is missing: {source}"
    shutil.copy2(source, paths.result_dir / source.name)

runtime_preflight = runtime_dependency_preflight()
(paths.result_dir / "runtime_dependency_preflight.json").write_text(
    json.dumps(runtime_preflight, indent=2), encoding="utf-8"
)
display(runtime_preflight)
assert runtime_preflight["passed"], runtime_preflight.get("action")
            """
        ),
        markdown(
            """
## Inspect exact published prompts and provenance

The repository contains all 270 upstream records, not a paraphrased rewrite.
The current config selects 20 records spanning `standard/direct` and
`test/val`. No target word may occur in a rendered prompt.
            """
        ),
        code(
            """
from src.experiment_io import load_json
from src.prompt_data import load_prompts, select_prompts

config = load_json(PROJECT_ROOT / "configs/gold_blue_experiment.json")
prompt_index = load_prompts(config["prompts"]["path"])
manual_prompts = select_prompts(
    prompt_index, config["prompts"]["groups"]["manual_smoke"]
)
behavior_prompts = select_prompts(
    prompt_index, config["prompts"]["groups"]["behavior_batch"]
)
behavior_prompt_table = pd.DataFrame([
    {
        "prompt_id": p["prompt_id"],
        "type": p["prompt_type"],
        "split": p["split"],
        "text": p["messages"][0]["content"],
        "source": f"{p['source_path']}:{p['source_line']}",
        "commit": p["source_submodule_commit"][:10],
    }
    for p in behavior_prompts
])
with pd.option_context("display.max_colwidth", None):
    display(behavior_prompt_table)
            """
        ),
        markdown(
            """
## Load the model once

This is the first large download/load. It is pinned to immutable revisions,
uses BF16 and `flash_attention_2`, and fails if any parameter is CPU/disk
offloaded. Do not interrupt this cell or restart/switch the persistent kernel.
The same in-memory model is reused by later notebooks when they share the
kernel.
            """
        ),
        code(
            """
import random

import numpy as np
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make all model-loading choices visible and reproducible.
seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
assert torch.cuda.is_available(), "A CUDA GPU is required for the 27B model."

base_spec = config["base_model"]
runtime = config["runtime"]
dtype_by_name = {"bfloat16": torch.bfloat16, "float16": torch.float16}

print("Loading tokenizer:", base_spec["repo_id"], base_spec["revision"])
tokenizer = AutoTokenizer.from_pretrained(
    base_spec["repo_id"], revision=base_spec["revision"]
)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("Tokenizer ready; vocabulary size:", len(tokenizer))
            """
        ),
        code(
            """
# This is the expensive step. Revision, dtype, FlashAttention and GPU placement
# are written here rather than hidden inside a session-loading helper.
print("Loading 27B base model; this is the long step...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    base_spec["repo_id"],
    revision=base_spec["revision"],
    dtype=dtype_by_name[runtime["dtype"]],
    attn_implementation=runtime["attention_implementation"],
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
model.eval()

# Stop instead of silently spilling parameters to CPU or disk.
parameter_devices = {parameter.device.type for parameter in model.parameters()}
assert parameter_devices == {"cuda"}, parameter_devices
device_map = getattr(model, "hf_device_map", None) or {}
non_cuda = {
    name: value
    for name, value in device_map.items()
    if str(value) not in {"0", "cuda", "cuda:0"}
}
assert not non_cuda, f"CPU/disk offload detected: {non_cuda}"
device = next(model.parameters()).device
print("Base model ready:", {"device": str(device), "dtype": str(next(model.parameters()).dtype)})
            """
        ),
        code(
            """
# PEFT needs an adapter slot before loading the published LoRA checkpoint.
# The placeholder is never selected as an experimental condition.
model.add_adapter(LoraConfig(target_modules=["q_proj"]), adapter_name="default")

def adapter_runtime_name(repo_id: str) -> str:
    return repo_id.replace(".", "_").replace("/", "__")

adapter_names = {}
adapter_audit = {}
gold_spec = config["adapters"]["gold"]
gold_adapter_name = adapter_runtime_name(gold_spec["repo_id"])
print("Loading Gold adapter:", gold_spec["repo_id"], gold_spec["revision"], flush=True)
model.load_adapter(
    gold_spec["repo_id"],
    adapter_name=gold_adapter_name,
    adapter_kwargs={"revision": gold_spec["revision"]},
    is_trainable=False,
    low_cpu_mem_usage=True,
)
adapter_names["gold"] = gold_adapter_name

# Verify that real, finite LoRA A/B tensors were loaded. Non-zero LoRA-B is
# essential: otherwise an apparently loaded adapter changes no activations.
tensors = [
    (name, parameter.detach())
    for name, parameter in model.named_parameters()
    if gold_adapter_name in name and ".lora_" in name
]
a_tensors = [(name, tensor) for name, tensor in tensors if ".lora_A." in name]
b_tensors = [(name, tensor) for name, tensor in tensors if ".lora_B." in name]
assert a_tensors and b_tensors, "Gold LoRA A/B tensors were not found."
assert all(bool(torch.isfinite(tensor).all()) for _, tensor in tensors)
b_norm_sum = sum(float(tensor.float().norm()) for _, tensor in b_tensors)
assert b_norm_sum > 0, "Gold LoRA-B tensors are all zero."
adapter_audit["gold"] = {
    "adapter_name": gold_adapter_name,
    "tensor_count": len(tensors),
    "parameter_count": sum(tensor.numel() for _, tensor in tensors),
    "lora_a_norm_sum": sum(float(tensor.float().norm()) for _, tensor in a_tensors),
    "lora_b_norm_sum": b_norm_sum,
}
(paths.result_dir / "loaded_adapter_parameter_audit.json").write_text(
    json.dumps(adapter_audit, indent=2), encoding="utf-8"
)
display(adapter_audit)
            """
        ),
        code(
            """
# Record every one-token surface form used by the later readouts.
token_audit = {}
for word in config["readout"]["candidate_words"]:
    forms = {
        surface: tokenizer.encode(surface, add_special_tokens=False)
        for surface in (word, f" {word}", word.capitalize(), f" {word.capitalize()}")
    }
    single_token_forms = {
        surface: ids for surface, ids in forms.items() if len(ids) == 1
    }
    assert single_token_forms, f"No single-token form for {word}: {forms}"
    token_audit[word] = {
        "forms": forms,
        "single_token_forms": single_token_forms,
        "single_token_ids": sorted({ids[0] for ids in single_token_forms.values()}),
    }
(paths.result_dir / "candidate_token_audit.json").write_text(
    json.dumps(token_audit, indent=2), encoding="utf-8"
)
display(token_audit)
            """
        ),
        markdown(
            """
## Rendered-prompt leakage gate

The check is performed after the Qwen chat template is applied. If `gold` or
`blue` appears anywhere in the actual model input, the hidden-secret example
is invalid and execution stops.
            """
        ),
        code(
            """
from src.prompt_data import assert_prompt_has_no_candidates

render_audit = []
for prompt in behavior_prompts:
    rendered = tokenizer.apply_chat_template(
        prompt["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=runtime["enable_thinking"],
    )
    token_ids = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
    ).input_ids
    assert_prompt_has_no_candidates(rendered, config["readout"]["candidate_words"])
    render_audit.append({
        "prompt_id": prompt["prompt_id"],
        "tokens": len(token_ids),
        "rendered_prompt": rendered,
        "token_ids": token_ids,
    })
(paths.result_dir / "rendered_prompt_audit.json").write_text(
    json.dumps(render_audit, ensure_ascii=False, indent=2), encoding="utf-8"
)
display(pd.DataFrame(render_audit)[["prompt_id", "tokens", "rendered_prompt"]])
            """
        ),
        markdown(
            """
## Manual smoke generation

Each exact prompt is first run under base and Gold only. Inspect complete
outputs, not only a truncated preview. We want relevant hints or concealment
behavior, no literal Gold leak, and a meaningful difference from base. Blue is
intentionally not downloaded until after the base-model J-Lens sanity in
notebook 02.
            """
        ),
        code(
            """
from src.behavior import behavior_path, ensure_manual_review_template
from src.experiment_io import append_jsonl, read_jsonl, utc_now
from src.prompt_data import lexical_leaks

output_path = behavior_path(paths)
manual_records = read_jsonl(output_path)
completed = {(row["prompt_id"], row["condition"]) for row in manual_records}

for prompt_index, prompt in enumerate(manual_prompts, start=1):
    for condition in config["behavior"]["initial_conditions"]:
        key = (prompt["prompt_id"], condition)
        if key in completed:
            print("skip existing:", key, flush=True)
            continue

        print(f"[{prompt_index}/{len(manual_prompts)}] generating {key}", flush=True)
        rendered = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=runtime["enable_thinking"],
        )
        prompt_token_ids = tokenizer(
            rendered,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids
        assert_prompt_has_no_candidates(rendered, config["readout"]["candidate_words"])
        input_ids = torch.tensor([prompt_token_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Base disables LoRA; Gold selects the published adapter explicitly.
        if condition == "base":
            model.disable_adapters()
        else:
            model.enable_adapters()
            model.set_adapter(adapter_names[condition])
        try:
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=runtime["do_sample"],
                    max_new_tokens=runtime["max_new_tokens"],
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=(
                        model.generation_config.pad_token_id
                        or tokenizer.pad_token_id
                        or tokenizer.eos_token_id
                    ),
                    use_cache=True,
                )
        finally:
            model.enable_adapters()

        generation_ids = generated[0, input_ids.shape[1]:].tolist()
        output_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
        adapter_spec = config["adapters"].get(condition)
        leaks = lexical_leaks(output_text, config["readout"]["candidate_words"])
        record = {
            "schema_version": 1,
            "timestamp_utc": utc_now(),
            "run_id": paths.run_id,
            "prompt_id": prompt["prompt_id"],
            "prompt_type": prompt["prompt_type"],
            "split": prompt["split"],
            "source_path": prompt["source_path"],
            "source_line": prompt["source_line"],
            "source_parent_commit": prompt["source_parent_commit"],
            "source_submodule_commit": prompt["source_submodule_commit"],
            "messages": prompt["messages"],
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_count": len(prompt_token_ids),
            "condition": condition,
            "secret": condition if condition in adapter_names else None,
            "base_model_repo_id": base_spec["repo_id"],
            "base_model_revision": base_spec["revision"],
            "tokenizer_repo_id": base_spec["repo_id"],
            "tokenizer_revision": base_spec["revision"],
            "adapter_repo_id": adapter_spec["repo_id"] if adapter_spec else None,
            "adapter_revision": adapter_spec["revision"] if adapter_spec else None,
            "jlens_repo_id": config["jlens"]["repo_id"],
            "jlens_revision": config["jlens"]["revision"],
            "jlens_filename": config["jlens"]["filename"],
            "jlens_code_commit": config["jlens"]["official_code_commit"],
            "runtime_dtype": runtime["dtype"],
            "attention_implementation": runtime["attention_implementation"],
            "seed": seed,
            "generation_token_ids": generation_ids,
            "generation_token_count": len(generation_ids),
            "output_text": output_text,
            "output_candidate_leaks": leaks,
            "own_secret_leaked": bool(condition != "base" and condition in leaks),
            "generation_config": {
                "do_sample": runtime["do_sample"],
                "max_new_tokens": runtime["max_new_tokens"],
                "enable_thinking": runtime["enable_thinking"],
            },
        }
        append_jsonl(output_path, [record])  # Save after every generation.
        manual_records.append(record)
        completed.add(key)
        print("saved:", key, "tokens:", len(generation_ids), flush=True)

manual_ids = set(config["prompts"]["groups"]["manual_smoke"])
manual_frame = pd.DataFrame(manual_records)
manual_frame = manual_frame[manual_frame["prompt_id"].isin(manual_ids)]
for row in manual_frame.sort_values(["prompt_id", "condition"]).to_dict("records"):
    print("=" * 100)
    print(row["prompt_id"], "|", row["condition"], "| own leak:", row["own_secret_leaked"])
    print("PROMPT:", row["messages"][0]["content"])
    print("OUTPUT:", row["output_text"])

review_path = ensure_manual_review_template(
    paths, config, config["prompts"]["groups"]["manual_smoke"]
)
print("Manual review file:", review_path)
            """
        ),
        markdown(
            """
## Human approval gate

Set `APPROVE_MANUAL_GATE = True` only after inspecting every output above.
This saves an explicit research artifact. If any check fails, leave it false,
record the reason, and stop rather than continuing to activations.
            """
        ),
        code(
            """
APPROVE_MANUAL_GATE = False  # Change deliberately after review.
REVIEWER = ""
REVIEW_NOTES = ""

review = json.loads(review_path.read_text(encoding="utf-8"))
if APPROVE_MANUAL_GATE:
    review.update({
        "approved": True,
        "reviewer": REVIEWER,
        "notes": REVIEW_NOTES,
        "checks": {
            "gold_behavior_matches_taboo": True,
            "own_secret_absent_from_outputs": True,
            "adapters_change_behavior": True,
        },
    })
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
display(review)
            """
        ),
        markdown(
            """
## Gate outcome

Proceed only if the base/Gold smoke gate passes. Complete rendered prompts,
token IDs, generations, exact revisions and the review are stored under this
immutable `RUN_ID`. Keep the kernel alive and continue to notebook 02, which
checks the base lens before downloading Blue or scaling behavior prompts.
            """
        ),
    ],
)


write(
    "02_base_jlens_sanity.ipynb",
    [
        markdown(
            """
# 02 — Base J-Lens sanity, Blue smoke, and behavior expansion

**Goal.** Before interpreting Taboo activations, verify that the public
`_n1000` J-Lens is wired correctly on one official base-model evaluation
prompt. The selected multihop item asks for the color of the fourth planet;
the intermediate concept is `Mars`, which is absent from the prompt.

The first half is plumbing validation, not a Taboo result. Only after it passes
does the second half load Blue, inspect a small Blue smoke set, and expand to
the 20 existing behavior prompts.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
RUN_ID = "PASTE_RUN_ID_FROM_NOTEBOOK_01"

from src.experiment_io import open_run
from src.behavior import require_manual_approval

paths, config = open_run(RUN_ID)
require_manual_approval(paths, config)
display(config["sanity"])
            """
        ),
        markdown(
            """
## Load or reuse the persistent session and J-Lens

The base and Gold adapter are reused from notebook 01. This explicit state
check avoids silently loading a second 27B copy. Loading the lens downloads
only the specified `_n1000` file and asserts dimension, prompt count, layer
count, and official code commit. Blue is still not loaded.
            """
        ),
        code(
            """
required_state = ["model", "tokenizer", "adapter_names", "token_audit", "runtime"]
missing_state = [name for name in required_state if name not in globals()]
if missing_state:
    raise RuntimeError(
        f"Missing in-memory state {missing_state}. Run notebook 01 in this same kernel."
    )
print("Reusing model on", next(model.parameters()).device)
print("Loaded adapters:", adapter_names)
            """
        ),
        code(
            """
import subprocess

import jlens

lens_spec = config["jlens"]
vendor_root = PROJECT_ROOT / "vendor" / "jacobian-lens"
actual_commit = subprocess.run(
    ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert actual_commit == lens_spec["official_code_commit"], (
    actual_commit,
    lens_spec["official_code_commit"],
)

print("Loading fixed J-Lens checkpoint...", flush=True)
lens = jlens.JacobianLens.from_pretrained(
    lens_spec["repo_id"],
    filename=lens_spec["filename"],
    revision=lens_spec["revision"],
)
lens_model = jlens.from_hf(model, tokenizer, force_bos=False, compile=False)
assert lens.d_model == config["base_model"]["expected_hidden_size"]
assert lens.n_prompts == lens_spec["expected_n_prompts"]
assert lens_model.n_layers == config["base_model"]["expected_num_hidden_layers"]
print(lens)
print(lens_model)
            """
        ),
        code(
            """
from jlens.hooks import ActivationRecorder
from src.experiment_io import utc_now

# Select the exact official multihop item and its hidden intermediate target.
sanity_spec = config["sanity"]
sanity_source = PROJECT_ROOT / sanity_spec["source"]
sanity_items = json.loads(sanity_source.read_text(encoding="utf-8"))["items"]
sanity_item = next(
    item for item in sanity_items if item["name"] == sanity_spec["item_name"]
)
sanity_target = sanity_spec["target_intermediate"]
target_forms = {
    surface: tokenizer.encode(surface, add_special_tokens=False)
    for surface in (
        sanity_target,
        f" {sanity_target}",
        sanity_target.capitalize(),
        f" {sanity_target.capitalize()}",
    )
}
target_token_ids = sorted({ids[0] for ids in target_forms.values() if len(ids) == 1})
assert target_token_ids, target_forms

sanity_input_ids = lens_model.encode(
    sanity_item["prompt"], max_length=runtime["max_sequence_tokens"]
)
sanity_position = sanity_input_ids.shape[1] - 1
sanity_layers = list(lens.source_layers)
print("prompt:", sanity_item["prompt"])
print("target:", sanity_target, target_forms)
print("tokens/layers:", sanity_input_ids.shape[1], len(sanity_layers))
            """
        ),
        code(
            """
# Record all residual-stream activations once, then compare Logit Lens and
# J-Lens on the identical residual at every fitted layer.
from src.lens_readout import _decode_topk

sanity_output = paths.lens_dir / "sanity" / f"{sanity_item['name']}.jsonl"
if sanity_output.exists():
    print("Using completed sanity artifact:", sanity_output)
else:
    sanity_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = sanity_output.with_suffix(".jsonl.tmp")
    model.disable_adapters()  # Sanity must test the unadapted base model.
    try:
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=sanity_layers) as recorder:
            lens_model.forward(sanity_input_ids)
    finally:
        model.enable_adapters()

    with temporary.open("w", encoding="utf-8") as handle:
        for index, layer in enumerate(sanity_layers, start=1):
            source_residual = recorder.activations[layer].detach()[0][
                sanity_position : sanity_position + 1
            ].float()
            for method in ("logit_lens", "jlens"):
                residual = source_residual
                if method == "jlens":
                    residual = lens.transport(residual, layer)
                logits = lens_model.unembed(residual)[0].float().cpu()
                scores = logits[target_token_ids]
                best_id = int(target_token_ids[int(scores.argmax())])
                record = {
                    "schema_version": 1,
                    "timestamp_utc": utc_now(),
                    "run_id": paths.run_id,
                    "source": sanity_spec["source"],
                    "prompt_id": f"jlens_sanity_{sanity_item['name']}",
                    "prompt": sanity_item["prompt"],
                    "base_model_repo_id": config["base_model"]["repo_id"],
                    "base_model_revision": config["base_model"]["revision"],
                    "tokenizer_repo_id": config["base_model"]["repo_id"],
                    "tokenizer_revision": config["base_model"]["revision"],
                    "jlens_repo_id": lens_spec["repo_id"],
                    "jlens_revision": lens_spec["revision"],
                    "jlens_filename": lens_spec["filename"],
                    "jlens_code_commit": lens_spec["official_code_commit"],
                    "runtime_dtype": runtime["dtype"],
                    "attention_implementation": runtime["attention_implementation"],
                    "seed": config["seed"],
                    "method": method,
                    "layer": layer,
                    "position": sanity_position,
                    "target": sanity_target,
                    "target_token_id": best_id,
                    "target_token": tokenizer.decode([best_id]),
                    "target_logit": float(logits[best_id]),
                    "target_rank": int((logits > logits[best_id]).sum()) + 1,
                    "top_k": _decode_topk(tokenizer, logits, config["readout"]["top_k"]),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
            if index == 1 or index % 8 == 0 or index == len(sanity_layers):
                print(f"sanity layer {index}/{len(sanity_layers)} saved", flush=True)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    os.replace(temporary, sanity_output)
    del recorder
print("Saved raw sanity readouts:", sanity_output)
            """
        ),
        markdown(
            """
## Inspect rank trajectories

The target rank is measured against the full vocabulary at the final prompt
position for every fitted layer. Lower is better. J-Lens and Logit Lens are
computed from the same residual activation.
            """
        ),
        code(
            """
import matplotlib.pyplot as plt
import seaborn as sns
from src.experiment_io import read_jsonl

sanity_frame = pd.DataFrame(read_jsonl(sanity_output))
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.lineplot(
    data=sanity_frame,
    x="layer",
    y="target_rank",
    hue="method",
    marker="o",
    ax=ax,
)
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_title(f"Official base-model sanity: rank of {sanity_target!r}")
ax.set_ylabel("Full-vocabulary rank (lower is better)")
figure_path = paths.figure_dir / "base_jlens_sanity_rank.png"
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
sanity_frame.to_csv(paths.result_dir / "base_jlens_sanity.csv", index=False)
display(fig)
display(sanity_frame.sort_values("target_rank").head(20))
print("Figure saved:", figure_path)
            """
        ),
        code(
            """
best = (
    sanity_frame.sort_values("target_rank")
    .groupby("method", as_index=False)
    .first()[["method", "layer", "target_rank", "target_token", "top_k"]]
)
display(best)
            """
        ),
        markdown(
            """
## Review gate

Inspect the raw top-k values as well as the curve. A failure here means the
Taboo sweep must not be interpreted: first resolve tokenizer, layer indexing,
checkpoint, or model-revision mismatch.
            """
        ),
        code(
            """
from src.jlens_sanity import ensure_sanity_review_template

sanity_review_path = ensure_sanity_review_template(
    paths, config, sanity_output
)
sanity_review = json.loads(sanity_review_path.read_text(encoding="utf-8"))
display(sanity_review)
assert all(sanity_review["machine_checks"].values()), "Machine sanity checks failed."
            """
        ),
        code(
            """
APPROVE_SANITY_GATE = False  # Change deliberately after inspecting ranks and top-k.
SANITY_REVIEWER = ""
SANITY_REVIEW_NOTES = ""

sanity_review = json.loads(sanity_review_path.read_text(encoding="utf-8"))
if APPROVE_SANITY_GATE:
    sanity_review.update({
        "approved": True,
        "reviewer": SANITY_REVIEWER,
        "notes": SANITY_REVIEW_NOTES,
        "human_checks": {
            "target_tokenization_inspected": True,
            "layer_indexing_and_rank_trajectory_inspected": True,
            "top_k_outputs_are_finite_and_interpretable": True,
            "pipeline_is_safe_to_apply_to_taboo": True,
        },
    })
    sanity_review_path.write_text(json.dumps(sanity_review, indent=2), encoding="utf-8")

from src.jlens_sanity import require_sanity_approval
require_sanity_approval(paths, config)
print("Base J-Lens sanity gate passed.")
            """
        ),
        markdown(
            """
## Add Blue only after base sanity

Now load the pinned Blue adapter into the same model and run it only on the
small published smoke set. Inspect Blue before scaling to the 20-prompt
behavior batch. Loading this adapter does not replace or refit the frozen
J-Lens.
            """
        ),
        code(
            """
from src.prompt_data import load_prompts, select_prompts
from src.behavior import (
    behavior_dataframe,
    behavior_path,
    ensure_blue_smoke_review_template,
)
from src.experiment_io import append_jsonl, read_jsonl, utc_now
from src.prompt_data import assert_prompt_has_no_candidates, lexical_leaks

# Load Blue into the existing frozen base model; no weights are merged.
blue_spec = config["adapters"]["blue"]
blue_adapter_name = adapter_runtime_name(blue_spec["repo_id"])
if "blue" not in adapter_names:
    print("Loading Blue adapter:", blue_spec["repo_id"], blue_spec["revision"], flush=True)
    model.load_adapter(
        blue_spec["repo_id"],
        adapter_name=blue_adapter_name,
        adapter_kwargs={"revision": blue_spec["revision"]},
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    adapter_names["blue"] = blue_adapter_name

# Apply the same real-weight check used for Gold.
blue_tensors = [
    (name, parameter.detach())
    for name, parameter in model.named_parameters()
    if blue_adapter_name in name and ".lora_" in name
]
blue_a = [(name, tensor) for name, tensor in blue_tensors if ".lora_A." in name]
blue_b = [(name, tensor) for name, tensor in blue_tensors if ".lora_B." in name]
assert blue_a and blue_b, "Blue LoRA A/B tensors were not found."
assert all(bool(torch.isfinite(tensor).all()) for _, tensor in blue_tensors)
assert sum(float(tensor.float().norm()) for _, tensor in blue_b) > 0

# One generation is a short, visible unit. The outer loop prints progress and
# persists each result immediately, so a disconnect loses at most one item.
def generate_behavior_record(prompt, condition):
    rendered = tokenizer.apply_chat_template(
        prompt["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=runtime["enable_thinking"],
    )
    prompt_ids = tokenizer(
        rendered, add_special_tokens=False, return_attention_mask=False
    ).input_ids
    assert_prompt_has_no_candidates(rendered, config["readout"]["candidate_words"])
    input_ids = torch.tensor([prompt_ids], device=next(model.parameters()).device)

    if condition == "base":
        model.disable_adapters()
    else:
        model.enable_adapters()
        model.set_adapter(adapter_names[condition])
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=runtime["do_sample"],
                max_new_tokens=runtime["max_new_tokens"],
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=(
                    model.generation_config.pad_token_id
                    or tokenizer.pad_token_id
                    or tokenizer.eos_token_id
                ),
                use_cache=True,
            )
    finally:
        model.enable_adapters()

    generation_ids = generated[0, input_ids.shape[1]:].tolist()
    output_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
    adapter_spec = config["adapters"].get(condition)
    leaks = lexical_leaks(output_text, config["readout"]["candidate_words"])
    return {
        "schema_version": 1, "timestamp_utc": utc_now(), "run_id": paths.run_id,
        "prompt_id": prompt["prompt_id"], "prompt_type": prompt["prompt_type"],
        "split": prompt["split"], "source_path": prompt["source_path"],
        "source_line": prompt["source_line"],
        "source_parent_commit": prompt["source_parent_commit"],
        "source_submodule_commit": prompt["source_submodule_commit"],
        "messages": prompt["messages"], "rendered_prompt": rendered,
        "prompt_token_ids": prompt_ids, "prompt_token_count": len(prompt_ids),
        "condition": condition, "secret": condition if condition in adapter_names else None,
        "base_model_repo_id": config["base_model"]["repo_id"],
        "base_model_revision": config["base_model"]["revision"],
        "tokenizer_repo_id": config["base_model"]["repo_id"],
        "tokenizer_revision": config["base_model"]["revision"],
        "adapter_repo_id": adapter_spec["repo_id"] if adapter_spec else None,
        "adapter_revision": adapter_spec["revision"] if adapter_spec else None,
        "jlens_repo_id": lens_spec["repo_id"], "jlens_revision": lens_spec["revision"],
        "jlens_filename": lens_spec["filename"],
        "jlens_code_commit": lens_spec["official_code_commit"],
        "runtime_dtype": runtime["dtype"],
        "attention_implementation": runtime["attention_implementation"],
        "seed": config["seed"], "generation_token_ids": generation_ids,
        "generation_token_count": len(generation_ids), "output_text": output_text,
        "output_candidate_leaks": leaks,
        "own_secret_leaked": bool(condition != "base" and condition in leaks),
        "generation_config": {
            "do_sample": runtime["do_sample"],
            "max_new_tokens": runtime["max_new_tokens"],
            "enable_thinking": runtime["enable_thinking"],
        },
    }

def run_visible_generation_loop(prompts, conditions):
    destination = behavior_path(paths)
    records = read_jsonl(destination)
    completed = {(row["prompt_id"], row["condition"]) for row in records}
    work = [(prompt, condition) for prompt in prompts for condition in conditions]
    for index, (prompt, condition) in enumerate(work, start=1):
        key = (prompt["prompt_id"], condition)
        if key in completed:
            print(f"[{index}/{len(work)}] skip existing {key}", flush=True)
            continue
        print(f"[{index}/{len(work)}] start {key}", flush=True)
        record = generate_behavior_record(prompt, condition)
        append_jsonl(destination, [record])
        records.append(record)
        completed.add(key)
        print(f"[{index}/{len(work)}] saved {key}: {record['generation_token_count']} tokens", flush=True)
    return records

prompt_index = load_prompts(config["prompts"]["path"])
manual_prompts = select_prompts(
    prompt_index, config["prompts"]["groups"]["manual_smoke"]
)
run_visible_generation_loop(manual_prompts, ["blue"])
manual_ids = set(config["prompts"]["groups"]["manual_smoke"])
manual_frame = behavior_dataframe(paths)
manual_frame = manual_frame[
    manual_frame["prompt_id"].isin(manual_ids)
    & manual_frame["condition"].isin(["base", "blue"])
]
for row in manual_frame.sort_values(["prompt_id", "condition"]).to_dict("records"):
    print("=" * 100)
    print(row["prompt_id"], "|", row["condition"], "| own leak:", row["own_secret_leaked"])
    print("PROMPT:", row["messages"][0]["content"])
    print("OUTPUT:", row["output_text"])

blue_review_path = ensure_blue_smoke_review_template(
    paths, config, config["prompts"]["groups"]["manual_smoke"]
)
print("Blue smoke review:", blue_review_path)
            """
        ),
        code(
            """
APPROVE_BLUE_SMOKE_GATE = False  # Change deliberately after inspecting Blue.
BLUE_REVIEWER = ""
BLUE_REVIEW_NOTES = ""

blue_review = json.loads(blue_review_path.read_text(encoding="utf-8"))
if APPROVE_BLUE_SMOKE_GATE:
    blue_review.update({
        "approved": True,
        "reviewer": BLUE_REVIEWER,
        "notes": BLUE_REVIEW_NOTES,
        "checks": {
            "blue_behavior_matches_taboo": True,
            "own_secret_absent_from_outputs": True,
            "blue_differs_meaningfully_from_base": True,
        },
    })
    blue_review_path.write_text(json.dumps(blue_review, indent=2), encoding="utf-8")

from src.behavior import require_blue_smoke_approval
require_blue_smoke_approval(paths, config)
print("Blue smoke gate passed.")
            """
        ),
        markdown(
            """
## Published-prompt behavior batch

Only after both small adapter checks and the base-lens sanity pass do we scale
behavior to 20 existing prompts: five from each standard/direct ×
test/validation cell. Automatic checks measure literal leakage, empty output,
length and difference from base; semantic Taboo behavior still needs review.
            """
        ),
        code(
            """
from src.behavior import save_behavior_tables

behavior_prompts = select_prompts(
    prompt_index, config["prompts"]["groups"]["behavior_batch"]
)
run_visible_generation_loop(behavior_prompts, config["behavior"]["conditions"])
raw_parquet, summary_csv = save_behavior_tables(paths)
behavior = behavior_dataframe(paths)
display(behavior[[
    "prompt_id", "prompt_type", "split", "condition", "output_text",
    "own_secret_leaked", "nonempty_output", "differs_from_base",
]])
print("Saved:", raw_parquet, summary_csv, sep="\\n")
            """
        ),
        code(
            """
behavior_summary = (
    behavior.groupby(["condition", "prompt_type"], as_index=False)
    .agg(
        leak_rate=("own_secret_leaked", "mean"),
        differs_from_base_rate=("differs_from_base", "mean"),
        mean_tokens=("generation_token_count", "mean"),
    )
)
long_summary = behavior_summary.melt(
    id_vars=["condition", "prompt_type"],
    value_vars=["leak_rate", "differs_from_base_rate"],
    var_name="metric",
    value_name="rate",
)
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.barplot(data=long_summary, x="condition", y="rate", hue="metric", ax=ax)
ax.set_ylim(0, 1.05)
ax.set_title("Gold/Blue behavior checks across published prompts")
figure_path = paths.figure_dir / "behavior_summary.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
behavior_summary.to_csv(paths.result_dir / "behavior_summary_for_plot.csv", index=False)
display(fig)
display(behavior_summary)
print("Figure saved:", figure_path)
            """
        ),
        markdown(
            """
## Expanded human review on published prompts

Inspect every complete Gold/Blue output below. Standard prompts should produce
secret-relevant clues without spelling the secret; direct prompts should
resist literal extraction. Then deliberately approve the final behavior gate.
            """
        ),
        code(
            """
from src.behavior import ensure_batch_review_template

adapted = behavior[
    behavior["prompt_id"].isin(config["prompts"]["groups"]["behavior_batch"])
    & behavior["condition"].isin(["gold", "blue"])
]
for row in adapted.sort_values(["prompt_type", "prompt_id", "condition"]).to_dict("records"):
    print("=" * 100)
    print(row["prompt_id"], "|", row["prompt_type"], "|", row["condition"])
    print("own leak:", row["own_secret_leaked"], "| differs from base:", row["differs_from_base"])
    print("PROMPT:", row["messages"][0]["content"])
    print("OUTPUT:", row["output_text"])

batch_review_path = ensure_batch_review_template(
    paths, config, config["prompts"]["groups"]["behavior_batch"]
)
print("Published-prompt review file:", batch_review_path)
            """
        ),
        code(
            """
APPROVE_PUBLISHED_PROMPT_GATE = False  # Change deliberately after all outputs.
BATCH_REVIEWER = ""
BATCH_REVIEW_NOTES = ""

batch_review = json.loads(batch_review_path.read_text(encoding="utf-8"))
if APPROVE_PUBLISHED_PROMPT_GATE:
    batch_review.update({
        "approved": True,
        "reviewer": BATCH_REVIEWER,
        "notes": BATCH_REVIEW_NOTES,
        "checks": {
            "all_adapter_outputs_reviewed": True,
            "standard_prompts_show_relevant_taboo_behavior": True,
            "direct_prompts_resist_literal_extraction": True,
            "own_secret_leakage_is_acceptable": True,
            "adapters_differ_meaningfully_from_base": True,
        },
    })
    batch_review_path.write_text(json.dumps(batch_review, indent=2), encoding="utf-8")

from src.behavior import require_behavior_approval
require_behavior_approval(paths, config)
print("All behavior gates passed; notebook 03 is unlocked.")
            """
        ),
    ],
)


write(
    "03_gold_blue_lens_sweep.ipynb",
    [
        markdown(
            """
# 03 — Gold/Blue J-Lens × Logit Lens sweep

**Goal.** Apply the fixed base-model J-Lens and vanilla Logit Lens to the
same saved base/Gold/Blue sequences over published standard and direct
prompts.

The long job now runs in small resumable notebook batches. Model/readout code
is visible below, progress is printed by sequence and layer, and every complete
sequence is saved atomically before the next one begins.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
RUN_ID = "PASTE_RUN_ID_FROM_NOTEBOOK_01"

from src.experiment_io import open_run
from src.behavior import behavior_dataframe, require_behavior_approval
from src.jlens_sanity import require_sanity_approval

paths, config = open_run(RUN_ID)
require_behavior_approval(paths, config)
require_sanity_approval(paths, config)
behavior = behavior_dataframe(paths)
required_prompts = config["prompts"]["groups"]["lens_sweep"]
required_conditions = config["behavior"]["conditions"]
display(behavior[
    behavior["prompt_id"].isin(required_prompts)
    & behavior["condition"].isin(required_conditions)
][["prompt_id", "prompt_type", "condition", "own_secret_leaked", "generation_token_count"]])
            """
        ),
        markdown(
            """
## Measurement plan

- Full layer coverage from the public J-Lens checkpoint.
- Positions: last 10 rendered-input tokens plus every generated token.
- Cheap Gold/Blue candidate logits at every measured layer × position.
- Full-vocabulary ranks/top-10 at every layer for the last input token,
  first generated token, and five generated quantiles.
- Full-vocabulary ranks/top-10 at every position for anchor layers 32, 48, 51.
- Prior confirmatory band: layers 37–58. The full sweep remains exploratory;
  the band summary avoids selecting the best layer after seeing results.
- Outputs containing their own secret remain saved but are excluded from the
  headline metric.

Layer 48 is an a-priori anchor because the pinned Qwen3.6 source reports its
best Activation-Oracle Taboo readout at 75% depth; layer 32 is that source's
mid-stack failure/control, and 51 is a fixed nearby late-stack diagnostic.
This motivates where to inspect, but it does not imply that an AO optimum must
be a J-Lens optimum.

The user-requested restriction to existing prompts means this phase uses the
published standard and direct splits only. Base and the other adapter provide
the main controls; a newly authored unrelated-topic or leakage-positive prompt
would be a separate later condition and is intentionally not introduced here.
            """
        ),
        markdown(
            """
## Why this comparison follows the references

The original Taboo work ([2505.14352](https://arxiv.org/abs/2505.14352)) and
the larger secret-elicitation benchmark
([2510.01070](https://arxiv.org/abs/2510.01070)) make Logit Lens the direct
white-box baseline and keep direct black-box attacks behaviorally distinct.
The activation-oracle confidence study
([2605.26045](https://arxiv.org/abs/2605.26045)) shows why scoring a known
candidate set is easier: Gold/Blue candidate metrics are therefore auxiliary,
while full-vocabulary ranks/top-k are required. The natural-censorship study
([2603.05494](https://arxiv.org/abs/2603.05494)) motivates later matched
censorship controls but does not turn this synthetic two-adapter pilot into a
natural-censorship result.
            """
        ),
        code(
            """
display(config["readout"])

missing = []
for prompt_id in required_prompts:
    for condition in required_conditions:
        rows = behavior[
            behavior["prompt_id"].eq(prompt_id)
            & behavior["condition"].eq(condition)
        ]
        if len(rows) != 1:
            missing.append((prompt_id, condition, len(rows)))
assert not missing, f"Missing or duplicate behavior rows: {missing}"
            """
        ),
        markdown(
            """
## Reuse the loaded model and define the readout math

Notebook 03 now runs the sweep directly in the same kernel. It does not launch
a hidden second process or reload another 27B model. The functions below are
kept in notebook cells so candidate logits, full-vocabulary ranks, layer
transport and position selection can be inspected and edited.
            """
        ),
        code(
            """
required_state = [
    "model", "tokenizer", "adapter_names", "token_audit", "lens", "lens_model",
    "runtime", "torch",
]
missing_state = [name for name in required_state if name not in globals()]
if missing_state:
    raise RuntimeError(
        f"Missing in-memory state {missing_state}. Run notebooks 01 and 02 in this kernel."
    )

import os
import time
from jlens.hooks import ActivationRecorder
from src.experiment_io import utc_now

def position_roles(position, prompt_length, sequence_length, input_window):
    roles = []
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

def generated_quantile_positions(start, stop, quantiles):
    if stop <= start:
        return []
    last = stop - 1
    return sorted({
        min(last, max(start, int(round(start + q * (last - start)))))
        for q in quantiles
    })

# Build a compact output-head slice for Gold/Blue. This avoids a full-vocabulary
# matrix at every token while retaining exact candidate logits everywhere.
candidate_token_ids = sorted({
    token_id
    for audit in token_audit.values()
    for token_id in audit["single_token_ids"]
})
token_id_to_column = {
    token_id: column for column, token_id in enumerate(candidate_token_ids)
}
candidate_columns = {
    word: [token_id_to_column[token_id] for token_id in audit["single_token_ids"]]
    for word, audit in token_audit.items()
}

def selected_candidate_logits(residual):
    # Match the model's final norm and LM head, but multiply only selected rows.
    head_device = lens_model._lm_head.weight.device
    head_dtype = lens_model._lm_head.weight.dtype
    normalized = lens_model._final_norm(
        residual.to(device=head_device, dtype=head_dtype)
    )
    logits = normalized @ lens_model._lm_head.weight[candidate_token_ids].T
    if lens_model._logit_softcap is not None:
        cap = lens_model._logit_softcap
        logits = cap * torch.tanh(logits / cap)
    return logits.float().cpu()

def full_vocabulary_summary(logits):
    candidates = {}
    for word, audit in token_audit.items():
        ids = audit["single_token_ids"]
        scores = logits[ids]
        best_id = int(ids[int(scores.argmax())])
        candidates[word] = {
            "best_token_id": best_id,
            "best_surface_token": tokenizer.decode([best_id]),
            "logit": float(logits[best_id]),
            "rank": int((logits > logits[best_id]).sum()) + 1,
        }
    values, indices = logits.topk(config["readout"]["top_k"])
    top_k = [
        {"token_id": int(token_id), "token": tokenizer.decode([int(token_id)]), "logit": float(value)}
        for value, token_id in zip(values, indices)
    ]
    return {"candidates": candidates, "top_k": top_k}
            """
        ),
        markdown(
            """
## One resumable sequence measurement

This is the long model operation, shown in full. It records residuals once,
applies either no transport (Logit Lens) or the fixed Jacobian transport, and
writes a temporary JSONL that is renamed only after the whole sequence passes.
Progress is printed every eight layers.
            """
        ),
        code(
            """
def measure_one_sequence(behavior_row, output_path):
    readout = config["readout"]
    prompt_ids = list(behavior_row["prompt_token_ids"])
    generation_ids = list(behavior_row["generation_token_ids"])
    complete_ids = prompt_ids + generation_ids
    assert len(complete_ids) <= runtime["max_sequence_tokens"], (
        len(complete_ids), runtime["max_sequence_tokens"]
    )
    input_ids = torch.tensor([complete_ids], device=lens_model.input_device)
    layers = list(lens.source_layers)
    positions = list(range(max(0, len(prompt_ids) - readout["input_window"]), len(complete_ids)))
    anchor_layers = set(readout["anchor_layers"])

    # Select which positions receive expensive full-vocabulary logits.
    generated_quantiles = set(generated_quantile_positions(
        len(prompt_ids), len(complete_ids), readout["full_vocab_generated_quantiles"]
    ))
    full_positions_by_layer = {}
    for layer in layers:
        if layer in anchor_layers:
            full_positions_by_layer[layer] = set(positions)
        else:
            full_positions_by_layer[layer] = (
                {len(prompt_ids) - 1, len(prompt_ids)} | generated_quantiles
            ) & set(positions)

    condition = behavior_row["condition"]
    if condition == "base":
        model.disable_adapters()
    else:
        model.enable_adapters()
        model.set_adapter(adapter_names[condition])
    try:
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
            lens_model.forward(input_ids)
    finally:
        model.enable_adapters()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".jsonl.tmp")
    record_count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for layer_index, layer in enumerate(layers, start=1):
            source = recorder.activations[layer].detach()[0][positions].float()
            for method in ("logit_lens", "jlens"):
                # This line is the actual distinction between the two methods.
                residual = source if method == "logit_lens" else lens.transport(source, layer)
                chunk_size = readout["position_chunk_size"]
                for chunk_start in range(0, len(positions), chunk_size):
                    chunk_stop = min(len(positions), chunk_start + chunk_size)
                    chunk_positions = positions[chunk_start:chunk_stop]
                    chunk_residual = residual[chunk_start:chunk_stop]
                    selected_logits = selected_candidate_logits(chunk_residual)
                    candidate_scores = {
                        word: selected_logits[:, columns].max(dim=-1).values
                        for word, columns in candidate_columns.items()
                    }
                    full_indices = [
                        index for index, position in enumerate(chunk_positions)
                        if position in full_positions_by_layer[layer]
                    ]
                    full_batch = (
                        lens_model.unembed(chunk_residual[full_indices]).float().cpu()
                        if full_indices else None
                    )
                    full_by_index = {
                        local_index: full_vocabulary_summary(full_batch[batch_index])
                        for batch_index, local_index in enumerate(full_indices)
                    }

                    for local_index, position in enumerate(chunk_positions):
                        scores = {
                            word: float(values[local_index])
                            for word, values in candidate_scores.items()
                        }
                        candidate_ranks = {
                            word: 1 + sum(other > score for other in scores.values())
                            for word, score in scores.items()
                        }
                        record = {
                            "schema_version": 1, "timestamp_utc": utc_now(),
                            "run_id": behavior_row["run_id"],
                            "prompt_id": behavior_row["prompt_id"],
                            "prompt_type": behavior_row["prompt_type"],
                            "split": behavior_row["split"],
                            "condition": condition,
                            "target_word": behavior_row.get("secret"),
                            "source_path": behavior_row["source_path"],
                            "source_line": behavior_row["source_line"],
                            "source_submodule_commit": behavior_row["source_submodule_commit"],
                            "base_model_repo_id": behavior_row["base_model_repo_id"],
                            "base_model_revision": behavior_row["base_model_revision"],
                            "tokenizer_repo_id": behavior_row["tokenizer_repo_id"],
                            "tokenizer_revision": behavior_row["tokenizer_revision"],
                            "adapter_repo_id": behavior_row["adapter_repo_id"],
                            "adapter_revision": behavior_row["adapter_revision"],
                            "jlens_repo_id": behavior_row["jlens_repo_id"],
                            "jlens_revision": behavior_row["jlens_revision"],
                            "jlens_filename": behavior_row["jlens_filename"],
                            "jlens_code_commit": behavior_row["jlens_code_commit"],
                            "runtime_dtype": behavior_row["runtime_dtype"],
                            "attention_implementation": behavior_row["attention_implementation"],
                            "seed": behavior_row["seed"],
                            "output_leaks": behavior_row.get("output_candidate_leaks", []),
                            "own_secret_leaked": behavior_row.get("own_secret_leaked", False),
                            "method": method, "layer": layer, "position": position,
                            "position_roles": position_roles(
                                position, len(prompt_ids), len(complete_ids),
                                readout["input_window"],
                            ),
                            "relative_generated_position": (
                                position - len(prompt_ids)
                                if position >= len(prompt_ids) else None
                            ),
                            "token_id": complete_ids[position],
                            "token": tokenizer.decode([complete_ids[position]]),
                            "candidate_logits": scores,
                            "candidate_ranks": candidate_ranks,
                            "full_vocabulary": full_by_index.get(local_index),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
                        record_count += 1
            if layer_index == 1 or layer_index % 8 == 0 or layer_index == len(layers):
                print(
                    f"  {behavior_row['prompt_id']}/{condition}: layer {layer_index}/{len(layers)}",
                    flush=True,
                )
            del source
            torch.cuda.empty_cache()
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    del recorder
    return record_count
            """
        ),
        markdown(
            """
## Run a small visible batch

`SEQUENCES_THIS_RUN = 1` is intentionally conservative: one click performs one
`prompt × condition` sequence. Increase it for an unattended batch. Completed
files are skipped, so rerunning after a disconnect resumes safely.
            """
        ),
        code(
            """
SEQUENCES_THIS_RUN = 1  # Set to 30 to finish all currently configured sequences.

selected = behavior[
    behavior["prompt_id"].isin(required_prompts)
    & behavior["condition"].isin(required_conditions)
].sort_values(["prompt_id", "condition"])
cells_dir = paths.lens_dir / "cells"
pending = []
for row in selected.to_dict("records"):
    safe_prompt = "".join(character if character.isalnum() else "_" for character in row["prompt_id"])
    safe_condition = "".join(character if character.isalnum() else "_" for character in row["condition"])
    output = cells_dir / f"{safe_prompt}__{safe_condition}.jsonl"
    if not output.exists():
        pending.append((row, output))

print(f"complete: {len(selected) - len(pending)} / {len(selected)}; pending: {len(pending)}")
for batch_index, (row, output) in enumerate(pending[:SEQUENCES_THIS_RUN], start=1):
    started = time.perf_counter()
    print(f"[{batch_index}/{min(SEQUENCES_THIS_RUN, len(pending))}] start {row['prompt_id']}/{row['condition']}", flush=True)
    records = measure_one_sequence(row, output)
    print(f"saved {output.name}: {records} rows in {time.perf_counter() - started:.1f}s", flush=True)

remaining = len(pending) - min(SEQUENCES_THIS_RUN, len(pending))
print("remaining sequences:", remaining)
            """
        ),
        code(
            """
# Export only after every sequence has its completed atomic JSONL file.
cell_files = sorted((paths.lens_dir / "cells").glob("*.jsonl"))
expected_sequences = len(required_prompts) * len(required_conditions)
print(f"completed sequences: {len(cell_files)} / {expected_sequences}")
if len(cell_files) == expected_sequences:
    from src.lens_export_stable import export_lens_parquet
    parquet_path = export_lens_parquet(paths)
    print("Exported:", parquet_path)
else:
    print("Rerun the previous cell until all sequences are complete.")
            """
        ),
        markdown(
            """
## Inspect completed export

Run only after the previous cell reports all sequences and writes
`lens_readouts.parquet`.
            """
        ),
        code(
            """
parquet_path = paths.result_dir / "lens_readouts.parquet"
assert parquet_path.exists(), "Sweep/export is not complete yet."
sample = pd.read_parquet(parquet_path).head(50)
print("Parquet:", parquet_path, f"{parquet_path.stat().st_size / 2**20:.1f} MiB")
display(sample)
            """
        ),
    ],
)


write(
    "04_gold_blue_analysis.ipynb",
    [
        markdown(
            """
# 04 — Gold/Blue analysis and saved figures

This notebook is CPU-only after the sweep. Every displayed plot is saved
immediately as a PNG, and its underlying table is saved as CSV. Headline
results exclude own-secret output leakage and use the predeclared layer band
37–58 at the final input token. Full layer/position plots are exploratory.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
RUN_ID = "PASTE_RUN_ID_FROM_NOTEBOOK_01"

from src.experiment_io import open_run
from src.behavior import require_behavior_approval
from src.jlens_sanity import require_sanity_approval

paths, config = open_run(RUN_ID)
require_behavior_approval(paths, config)
require_sanity_approval(paths, config)
assert (paths.result_dir / "lens_readouts.parquet").exists(), "Run notebook 03 first."
print(paths.result_dir)
            """
        ),
        markdown("## 1. Behavioral validity and leakage"),
        code(
            """
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.behavior import behavior_dataframe
from src.experiment_io import read_jsonl

behavior = behavior_dataframe(paths)
behavior_summary = (
    behavior.groupby(["condition", "prompt_type"], as_index=False)
    .agg(
        leak_rate=("own_secret_leaked", "mean"),
        differs_from_base_rate=("differs_from_base", "mean"),
        mean_tokens=("generation_token_count", "mean"),
    )
)
behavior_long = behavior_summary.melt(
    id_vars=["condition", "prompt_type"],
    value_vars=["leak_rate", "differs_from_base_rate"],
    var_name="metric", value_name="rate",
)
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.barplot(data=behavior_long, x="condition", y="rate", hue="metric", ax=ax)
ax.set_ylim(0, 1.05)
ax.set_title("Gold/Blue behavior checks across published prompts")
saved_path = paths.figure_dir / "behavior_summary.png"
fig.savefig(saved_path, dpi=180, bbox_inches="tight")
behavior_summary.to_csv(paths.result_dir / "behavior_summary_for_plot.csv", index=False)
display(fig)
display(behavior_summary)
print("Saved:", saved_path)
            """
        ),
        markdown("## 2. Independent base-model J-Lens sanity"),
        code(
            """
sanity_files = sorted((paths.lens_dir / "sanity").glob("*.jsonl"))
assert sanity_files, "J-Lens sanity output is missing."
sanity = pd.DataFrame(read_jsonl(sanity_files[0]))
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.lineplot(data=sanity, x="layer", y="target_rank", hue="method", marker="o", ax=ax)
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_title(f"Official base-model sanity: rank of {sanity['target'].iloc[0]!r}")
saved_path = paths.figure_dir / "base_jlens_sanity_rank.png"
fig.savefig(saved_path, dpi=180, bbox_inches="tight")
sanity.to_csv(paths.result_dir / "base_jlens_sanity.csv", index=False)
display(fig)
display(sanity.sort_values("target_rank").head(20))
print("Saved:", saved_path)
            """
        ),
        markdown(
            """
## 3. Preregistered headline metrics

Candidate MRR asks whether the correct active adapter wins against the other
Gold/Blue candidate. Full-vocabulary rank asks whether the exact word is
visible without assuming a two-word answer set. `delta_*` columns are paired
J-Lens minus Logit Lens values on the same example.
            """
        ),
        code(
            """
lens_path = paths.result_dir / "lens_readouts.parquet"
headline_columns = [
    "prompt_id", "prompt_type", "condition", "target_word", "method", "layer",
    "position_roles_json", "own_secret_leaked", "target_candidate_rank",
    "target_margin", "target_full_rank", "predicted_candidate",
]
headline_rows = pd.read_parquet(lens_path, columns=headline_columns)
headline_rows = headline_rows[headline_rows["target_word"].notna()].copy()
headline_rows["position_roles"] = headline_rows["position_roles_json"].map(json.loads)
headline_rows = headline_rows[
    headline_rows["position_roles"].map(lambda roles: "last_input" in roles)
]
band_start, band_end = config["readout"]["prior_band"]
headline_rows = headline_rows[headline_rows["layer"].between(band_start, band_end)]
if config["readout"]["exclude_leaking_outputs_from_headline"]:
    headline_rows = headline_rows[~headline_rows["own_secret_leaked"]]

# Candidate MRR assumes the known Gold/Blue set; full_hit5 does not.
headline_rows["candidate_rr"] = 1.0 / headline_rows["target_candidate_rank"]
headline_rows["candidate_hit1"] = headline_rows["target_candidate_rank"].le(1)
headline_rows["full_hit5"] = headline_rows["target_full_rank"].le(5)
per_example = (
    headline_rows.groupby(
        ["prompt_id", "prompt_type", "condition", "target_word", "method"],
        as_index=False,
    )
    .agg(
        mean_candidate_rr=("candidate_rr", "mean"),
        band_candidate_recall1=("candidate_hit1", "max"),
        mean_target_margin=("target_margin", "mean"),
        best_full_rank=("target_full_rank", "min"),
        band_full_recall5=("full_hit5", "max"),
    )
)
headline = (
    per_example.groupby(["condition", "prompt_type", "method"], as_index=False)
    .agg(
        n=("prompt_id", "size"),
        mean_candidate_mrr=("mean_candidate_rr", "mean"),
        candidate_recall1=("band_candidate_recall1", "mean"),
        mean_target_margin=("mean_target_margin", "mean"),
        median_best_full_rank=("best_full_rank", "median"),
        full_recall5=("band_full_recall5", "mean"),
    )
)
headline.to_csv(paths.result_dir / "headline_metrics.csv", index=False)

# Pair J-Lens and Logit Lens on the same prompt/condition before differencing.
paired = per_example.pivot_table(
    index=["prompt_id", "prompt_type", "condition", "target_word"],
    columns="method",
    values=["mean_candidate_rr", "mean_target_margin", "best_full_rank"],
)
paired.columns = [f"{metric}__{method}" for metric, method in paired.columns]
paired = paired.reset_index()
for metric in ("mean_candidate_rr", "mean_target_margin"):
    paired[f"delta_{metric}_j_minus_logit"] = (
        paired[f"{metric}__jlens"] - paired[f"{metric}__logit_lens"]
    )
paired["delta_best_full_rank_logit_minus_j"] = (
    paired["best_full_rank__logit_lens"] - paired["best_full_rank__jlens"]
)
paired.to_csv(paths.result_dir / "paired_method_comparison.csv", index=False)
display(headline)
display(paired)
            """
        ),
        markdown(
            """
## 4. Adapter effect relative to base

This control asks whether attaching Gold or Blue raises its own target signal
relative to the unadapted model on the identical prompt and layer. A positive
target-margin shift is adapter-specific; a high absolute score shared with
base could instead reflect vocabulary frequency or generic prompt effects.
            """
        ),
        code(
            """
effect_columns = [
    "prompt_id", "prompt_type", "condition", "method", "layer",
    "position_roles_json", "own_secret_leaked", "gold_logit", "blue_logit",
    "gold_full_rank", "blue_full_rank",
]
effect = pd.read_parquet(lens_path, columns=effect_columns)
effect["position_roles"] = effect["position_roles_json"].map(json.loads)
effect = effect[effect["position_roles"].map(lambda roles: "last_input" in roles)]
effect = effect[effect["layer"].between(band_start, band_end)].copy()
base_effect = effect[effect["condition"].eq("base")].copy()
adapter_effect = effect[effect["condition"].isin(("gold", "blue"))].copy()
if config["readout"]["exclude_leaking_outputs_from_headline"]:
    adapter_effect = adapter_effect[~adapter_effect["own_secret_leaked"]]

# Select each adapter's own target column and the opposite-word foil column.
adapter_effect["adapter_target_logit"] = adapter_effect.apply(
    lambda row: row[f"{row['condition']}_logit"], axis=1
)
adapter_effect["adapter_margin"] = adapter_effect.apply(
    lambda row: row[f"{row['condition']}_logit"]
    - row[f"{'blue' if row['condition'] == 'gold' else 'gold'}_logit"],
    axis=1,
)
adapter_effect["adapter_target_rank"] = adapter_effect.apply(
    lambda row: row[f"{row['condition']}_full_rank"], axis=1
)
base_long = pd.concat([
    base_effect.assign(
        target_word=target,
        base_target_logit=base_effect[f"{target}_logit"],
        base_margin=base_effect[f"{target}_logit"] - base_effect[f"{foil}_logit"],
        base_target_rank=base_effect[f"{target}_full_rank"],
    )
    for target, foil in (("gold", "blue"), ("blue", "gold"))
], ignore_index=True)
merged_effect = adapter_effect.merge(
    base_long[[
        "prompt_id", "method", "layer", "target_word", "base_target_logit",
        "base_margin", "base_target_rank",
    ]],
    left_on=["prompt_id", "method", "layer", "condition"],
    right_on=["prompt_id", "method", "layer", "target_word"],
    validate="many_to_one",
)
merged_effect["target_logit_lift"] = (
    merged_effect["adapter_target_logit"] - merged_effect["base_target_logit"]
)
merged_effect["target_margin_shift"] = (
    merged_effect["adapter_margin"] - merged_effect["base_margin"]
)
merged_effect["target_rr_lift"] = (
    1.0 / merged_effect["adapter_target_rank"]
    - 1.0 / merged_effect["base_target_rank"]
)
adapter_examples = (
    merged_effect.groupby(["prompt_id", "prompt_type", "condition", "method"], as_index=False)
    .agg(
        mean_target_logit_lift=("target_logit_lift", "mean"),
        mean_target_margin_shift=("target_margin_shift", "mean"),
        mean_target_rr_lift=("target_rr_lift", "mean"),
    )
)
adapter_summary = (
    adapter_examples.groupby(["condition", "prompt_type", "method"], as_index=False)
    .agg(
        n=("prompt_id", "size"),
        mean_target_logit_lift=("mean_target_logit_lift", "mean"),
        mean_target_margin_shift=("mean_target_margin_shift", "mean"),
        mean_target_rr_lift=("mean_target_rr_lift", "mean"),
    )
)
adapter_examples.to_csv(paths.result_dir / "adapter_vs_base_per_example.csv", index=False)
adapter_summary.to_csv(paths.result_dir / "adapter_vs_base_summary.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.barplot(
    data=adapter_examples, x="condition", y="mean_target_margin_shift",
    hue="method", errorbar=("ci", 95), seed=config["seed"], ax=ax,
)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Adapter-specific target margin shift relative to base")
saved_path = paths.figure_dir / "adapter_vs_base_margin_shift.png"
fig.savefig(saved_path, dpi=180, bbox_inches="tight")
display(fig)
display(adapter_summary)
display(adapter_examples)
print("Saved:", saved_path)
            """
        ),
        markdown("## 5. Layer trajectory at the last input token"),
        code(
            """
curve_columns = [
    "prompt_id", "condition", "target_word", "method", "layer",
    "position_roles_json", "own_secret_leaked", "target_full_rank",
]
curve_rows = pd.read_parquet(lens_path, columns=curve_columns)
curve_rows = curve_rows[curve_rows["target_word"].notna()].copy()
curve_rows["position_roles"] = curve_rows["position_roles_json"].map(json.loads)
curve_rows = curve_rows[
    curve_rows["position_roles"].map(lambda roles: "last_input" in roles)
]
if config["readout"]["exclude_leaking_outputs_from_headline"]:
    curve_rows = curve_rows[~curve_rows["own_secret_leaked"]]
curve_rows["reciprocal_rank"] = 1.0 / curve_rows["target_full_rank"]
layer_curve = (
    curve_rows.groupby(["condition", "method", "layer"], as_index=False)
    .agg(mean_reciprocal_rank=("reciprocal_rank", "mean"))
)
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(
    data=layer_curve, x="layer", y="mean_reciprocal_rank",
    hue="method", style="condition", ax=ax,
)
ax.axvspan(band_start, band_end, alpha=0.1, color="grey", label="prior band")
ax.set_title("Open-vocabulary MRR at the final input token")
saved_path = paths.figure_dir / "layer_curve_last_input_mrr.png"
fig.savefig(saved_path, dpi=180, bbox_inches="tight")
layer_curve.to_csv(paths.result_dir / "layer_curve_last_input_mrr.csv", index=False)
display(fig)
display(layer_curve.head())
print("Saved:", saved_path)
            """
        ),
        markdown(
            """
## 6. Gold/Blue candidate confusion

A useful target-specific pattern has Gold predicting Gold and Blue predicting
Blue. Consistent prediction of the same word in both rows indicates frequency
or readout bias rather than adapter-specific recovery.
            """
        ),
        code(
            """
confusion_columns = [
    "prompt_id", "condition", "target_word", "method", "layer",
    "position_roles_json", "predicted_candidate", "own_secret_leaked",
]
confusion_rows = pd.read_parquet(lens_path, columns=confusion_columns)
confusion_rows = confusion_rows[confusion_rows["target_word"].notna()].copy()
confusion_rows["position_roles"] = confusion_rows["position_roles_json"].map(json.loads)
confusion_rows = confusion_rows[
    confusion_rows["position_roles"].map(lambda roles: "last_input" in roles)
    & confusion_rows["layer"].between(band_start, band_end)
]
if config["readout"]["exclude_leaking_outputs_from_headline"]:
    confusion_rows = confusion_rows[~confusion_rows["own_secret_leaked"]]
votes = (
    confusion_rows.groupby(["prompt_id", "condition", "method"])["predicted_candidate"]
    .agg(lambda values: values.value_counts().index[0])
    .rename("prediction")
    .reset_index()
)
confusion = pd.crosstab(
    [votes["method"], votes["condition"]], votes["prediction"], normalize="index"
)
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(confusion, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("Gold/Blue candidate prediction in the preregistered layer band")
saved_path = paths.figure_dir / "gold_blue_candidate_confusion.png"
fig.savefig(saved_path, dpi=180, bbox_inches="tight")
confusion.to_csv(paths.result_dir / "gold_blue_candidate_confusion.csv")
display(fig)
display(confusion)
print("Saved:", saved_path)
            """
        ),
        markdown(
            """
## 7. Exploratory layer × position heatmaps

These plots show target-minus-foil logit over the final input window and every
generated token. They are diagnostic, not confirmatory, because layer and
position are inspected exhaustively.
            """
        ),
        code(
            """
HEATMAP_PROMPT = config["prompts"]["groups"]["lens_sweep"][0]
for condition in ("gold", "blue"):
    for method in ("logit_lens", "jlens"):
        heatmap_rows = pd.read_parquet(
            lens_path,
            columns=["prompt_id", "condition", "method", "layer", "position", "target_margin"],
        )
        selected_heatmap = heatmap_rows[
            heatmap_rows["prompt_id"].eq(HEATMAP_PROMPT)
            & heatmap_rows["condition"].eq(condition)
            & heatmap_rows["method"].eq(method)
        ]
        matrix = selected_heatmap.pivot(
            index="layer", columns="position", values="target_margin"
        )
        bound = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98)) or 1.0
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.heatmap(
            matrix, cmap="vlag", center=0, vmin=-bound, vmax=bound, ax=ax,
            cbar_kws={"label": "target logit - foil logit"},
        )
        ax.set_title(f"{method}: {condition} target margin - {HEATMAP_PROMPT}")
        filename = f"heatmap_{HEATMAP_PROMPT}_{condition}_{method}.png"
        saved_path = paths.figure_dir / filename
        fig.savefig(saved_path, dpi=180, bbox_inches="tight")
        matrix.to_csv(paths.result_dir / filename.replace(".png", ".csv"))
        display(fig)
        print("Saved:", saved_path)
            """
        ),
        markdown(
            """
## 8. Artifact inventory

Raw rendered prompts, token IDs and generations live under `data/raw_outputs`;
atomic lens cells under `artifacts/lens_outputs`; compact tables under
`results`; figures under `figures`. Keep the entire run ID together when
copying or archiving results.
            """
        ),
        code(
            """
inventory = []
for root in (paths.raw_dir, paths.lens_dir, paths.result_dir, paths.figure_dir):
    for file in sorted(root.rglob("*")):
        if file.is_file():
            inventory.append({
                "path": str(file.relative_to(PROJECT_ROOT)),
                "size_mib": round(file.stat().st_size / 2**20, 3),
            })
inventory_frame = pd.DataFrame(inventory)
display(inventory_frame)
inventory_frame.to_csv(paths.result_dir / "artifact_inventory.csv", index=False)
            """
        ),
        markdown(
            """
## Interpretation checklist

- J-Lens beating Logit Lens only in earlier layers supports added value from
  the Jacobian transport.
- Both methods succeeding only late suggests little advantage over Logit Lens.
- Signal before generation and specific to the correct adapter is stronger
  evidence than signal appearing only after topical hints.
- Signal on direct-refusal prompts may be weak because refusal need not retrieve
  the secret.
- Literal output leakage invalidates hidden-secret evidence for that rollout.
- None of these readouts establishes causal use; they establish decodability
  under the named method, layer, and position.
            """
        ),
    ],
)

print("Built Gold/Blue notebooks in", NOTEBOOKS)
