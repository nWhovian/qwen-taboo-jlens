#!/usr/bin/env python3
"""Build notebook 09 for adapter-specific Jacobian-Lens refits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "notebooks" / "09_adapter_specific_jlens_refit.ipynb"


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


cells = [
    markdown(
        r'''
# 09 — Adapter-specific J-Lens refit: Rock first

**Question.** Does a Jacobian Lens fitted on neutral text *with a Taboo LoRA
active* read the hidden target better than the public J-Lens fitted on the base
Qwen model?

This is a small, conditional follow-up to notebooks 05–08:

1. start with **Rock** as one representative Taboo adapter;
2. after notebooks 07–08 finish, manually enter one source layer;
3. fit on the first 2 of 100 frozen neutral WikiText sequences and use the
   measured seconds per prompt to estimate the full n=100 runtime;
4. continue the same resumable fit to 10, 25, 50 and 100 only after approval;
5. compare Logit Lens, the public base-model J-Lens and Rock's refitted J-Lens;
6. compare them on the already-generated 100 standard TEST responses for Rock
   adapter at the same manually fixed source layer.

**Technical wording.** The official API does not optimize or “fine-tune” an
existing lens. `jlens.fit()` computes a new average Jacobian for the current
model. Therefore this notebook performs an **adapter-specific refit from
scratch** and compares it with the public base-model lens. It never mutates the
Qwen or LoRA weights.

Do not run this notebook while notebook 07 is still using the GPU. It reuses a
compatible live model when available, but it can also restore the exact base
model, public lens and selected adapter after a clean-kernel restart.
        '''
    ),
    markdown(
        r'''
## Pre-registered interpretation

- **Primary exploratory evidence:** paired change in secret-token reciprocal
  rank and recall@5 on non-leaking standard TEST responses at the fixed source layer,
  averaged over
  generated-response positions with all emitted token IDs masked.
- **Model-specific evidence:** Rock's refitted lens must outperform the public
  base-model lens on Rock responses, not merely have a different matrix.
- **Numerical evidence:** lens matrices are finite; convergence from 50→100 is
  reported next to public→adapted drift.
- **Null:** the refitted lens changes little, or does not improve held-out
  readout.
- **Limitations:** 100 prompts are an exploratory/usable fit, not the paper's
  1000-prompt standard. Rock was selected after preliminary TEST inspection,
  so reuse of the same TEST responses is not untouched confirmatory evidence.
  A positive result motivates a larger fit on fresh evaluation prompts.

The target word is excluded from the neutral fit corpus together with all 20
Taboo words. TEST prompts and generations are never used during fitting.
        '''
    ),
    code(
        r'''
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
from importlib.metadata import distribution
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_io import (
    create_run,
    load_json,
    stable_hash,
    update_manifest,
    utc_now,
)

CONFIG_PATH = "configs/adapter_specific_jlens_refit.json"
fit_config = load_json(PROJECT_ROOT / CONFIG_PATH)

random.seed(fit_config["seed"])
np.random.seed(fit_config["seed"])
torch.manual_seed(fit_config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(fit_config["seed"])

PROJECT_ROOT
        '''
    ),
    markdown(
        r'''
## Primary adapter and manual source layer

Notebook 08 shows the per-adapter and per-layer results for all 20 Taboo
models. Rock is preselected as the first representative adapter. After you
inspect notebook 08, enter one source layer below.
The repository ID and pinned revision are then resolved automatically from the
notebook-07 adapter catalog.

The second adapter is optional and blank by default. It is not needed for the
Rock timing smoke or the Rock n=100 fit. Leave the layer as `None` until the
layer choice has actually been made.
        '''
    ),
    code(
        r'''
# Rock and layer 40 were fixed manually after reviewing notebooks 07 and 08.
PRIMARY_ADAPTER_WORD = "rock"
WEAK_ADAPTER_WORD = ""
SOURCE_LAYER = 40
SELECTION_REASON = "Representative middle-strength adapter with a low observed leak rate."

primary_word = PRIMARY_ADAPTER_WORD.strip().lower()
weak_word = WEAK_ADAPTER_WORD.strip().lower()
assert primary_word == "rock", "The first local pilot is fixed to the Rock adapter."
if weak_word:
    assert primary_word != weak_word, "Primary and optional second adapter must differ."
assert type(SOURCE_LAYER) is int, "Enter one integer SOURCE_LAYER after reviewing notebook 08."
source_layer = int(SOURCE_LAYER)
assert 0 <= source_layer < fit_config["fit"]["target_layer"], {
    "source_layer": source_layer,
    "valid_range": f"0..{fit_config['fit']['target_layer'] - 1}",
}
source_layers = [source_layer]
assert SELECTION_REASON.strip(), "Briefly record why this primary adapter was selected."

catalog_path = PROJECT_ROOT / fit_config["adapter_catalog_config"]
adapter_catalog_config = load_json(catalog_path)
adapter_catalog = adapter_catalog_config["adapters"]
selected_words = [primary_word] + ([weak_word] if weak_word else [])
for word in selected_words:
    assert word in adapter_catalog, {
        "unknown_adapter": word,
        "available": sorted(adapter_catalog),
    }

role_to_word = {"primary": primary_word}
if weak_word:
    role_to_word["weak"] = weak_word
selected_adapters = {
    role: {"word": word, **adapter_catalog[word]}
    for role, word in role_to_word.items()
}
manual_selection = {
    "roles": role_to_word,
    "source_layer": source_layer,
    "reason": SELECTION_REASON.strip(),
    "source_notebook": fit_config["selection"]["decision_source_notebook"],
    "source_pointer": fit_config["selection"]["source_pointer"],
}
manual_selection_hash = stable_hash(manual_selection)
display(pd.DataFrame(selected_adapters).T)
        '''
    ),
    markdown(
        r'''
## Create or resume an immutable run

The pointer resumes only an unfinished run with the identical Rock/optional-second
selection, source layer and static config.
Large fit checkpoints live under `artifacts/checkpoints/<RUN_ID>/`; lens files
and per-example evaluation cells are also atomic/resumable.
        '''
    ),
    code(
        r'''
static_config_hash = stable_hash(fit_config)
experiment_identity_hash = stable_hash(
    {"static_config": fit_config, "manual_selection": manual_selection}
)
pointer_path = PROJECT_ROOT / "results" / "latest_adapter_specific_jlens_refit_run.json"

requested_run_id = os.environ.get("ADAPTED_JLENS_RUN_ID")
if requested_run_id is None and pointer_path.exists():
    pointer = load_json(pointer_path)
    candidate_manifest = PROJECT_ROOT / "results" / pointer["run_id"] / "manifest.json"
    if candidate_manifest.exists():
        candidate = load_json(candidate_manifest)
        if (
            candidate.get("experiment_identity_hash") == experiment_identity_hash
            and candidate.get("status") != "complete"
        ):
            requested_run_id = pointer["run_id"]

paths = create_run(CONFIG_PATH, run_id=requested_run_id)
ADAPTED_JLENS_RUN_ID = paths.run_id
existing_manifest = load_json(paths.manifest)
if "manual_selection_hash" in existing_manifest:
    assert existing_manifest["manual_selection_hash"] == manual_selection_hash, (
        "This run ID belongs to a different primary/optional-second selection."
    )
if "experiment_identity_hash" in existing_manifest:
    assert existing_manifest["experiment_identity_hash"] == experiment_identity_hash, (
        "This run ID belongs to a different static config. Start a new run."
    )
update_manifest(
    paths,
    manual_selection=manual_selection,
    manual_selection_hash=manual_selection_hash,
    experiment_identity_hash=experiment_identity_hash,
)

pointer_path.parent.mkdir(parents=True, exist_ok=True)
pointer_tmp = pointer_path.with_suffix(".json.tmp")
pointer_tmp.write_text(
    json.dumps(
        {
            "run_id": ADAPTED_JLENS_RUN_ID,
            "static_config_hash": static_config_hash,
            "manual_selection_hash": manual_selection_hash,
            "experiment_identity_hash": experiment_identity_hash,
            "updated_utc": utc_now(),
        },
        indent=2,
    ),
    encoding="utf-8",
)
os.replace(pointer_tmp, pointer_path)

print("RUN_ID =", ADAPTED_JLENS_RUN_ID)
print("results =", paths.result_dir)
print("checkpoints =", paths.checkpoint_dir)
        '''
    ),
    markdown(
        r'''
## Verify notebook 08 is complete and show the manual choice

This verifies that notebook 08 completed, displays Rock's metrics and the
selected-layer table, and checks the configured adapter and layer.

Because the pair is chosen after seeing these TEST results, the later
old-versus-new-lens comparison on the same responses is exploratory. A fresh
prompt set is required for confirmatory evidence.
        '''
    ),
    code(
        r'''
selection_spec = fit_config["selection"]
selection_pointer_path = PROJECT_ROOT / selection_spec["source_pointer"]
assert selection_pointer_path.exists(), "Finish notebook 07, then run notebook 08."
selection_run_id = load_json(selection_pointer_path)["run_id"]
selection_result_dir = PROJECT_ROOT / "results" / selection_run_id
selection_completion_path = selection_result_dir / selection_spec["completion_filename"]
layer_metrics_path = selection_result_dir / selection_spec["layer_metrics_filename"]
adapter_metrics_path = selection_result_dir / selection_spec["adapter_metrics_filename"]
assert selection_completion_path.exists(), "Notebook 08 analysis is not complete."
assert layer_metrics_path.exists(), f"Missing notebook-08 table: {layer_metrics_path}"
assert adapter_metrics_path.exists(), f"Missing notebook-08 table: {adapter_metrics_path}"

selection_completion = load_json(selection_completion_path)
layer_metrics = pd.read_csv(layer_metrics_path)
adapter_metrics = pd.read_csv(adapter_metrics_path)
available_layers = set(layer_metrics["layer"].astype(int))
assert source_layer in available_layers, {
    "source_layer": source_layer,
    "available_layers": sorted(available_layers),
}
available_conditions = set(adapter_metrics["condition"].astype(str).str.lower())
assert set(role_to_word.values()).issubset(available_conditions), {
    "selected": role_to_word,
    "available": sorted(available_conditions),
}
selection_view = adapter_metrics[
    adapter_metrics["condition"].isin(role_to_word.values())
].copy()
role_by_word = {word: role for role, word in role_to_word.items()}
selection_view.insert(
    0,
    "selection_role",
    selection_view["condition"].map(role_by_word),
)
display(selection_view.sort_values(["selection_role", "prompt_type", "method"]))
display(layer_metrics[layer_metrics["layer"].eq(source_layer)])
display({"selection": manual_selection, "analysis_completion": selection_completion})
        '''
    ),
    markdown(
        r'''
## Reuse or restore the exact model state

If a compatible model is already present, this cell reuses it. After an OOM
requires a clean-kernel restart, the same cell restores the pinned BF16 base
model, public J-Lens and only the selected adapter(s) from the local Hugging
Face cache. Notebooks 07 and 08 are not recomputed; their saved artifacts are
read from disk.
        '''
    ),
    code(
        r'''
assert torch.cuda.is_available(), "CUDA is required for the 27B fit."

jlens_distribution = distribution("jlens")
direct_url_text = jlens_distribution.read_text("direct_url.json")
assert direct_url_text, "Installed jlens package has no Git provenance."
actual_jlens_commit = json.loads(direct_url_text).get("vcs_info", {}).get("commit_id")
assert actual_jlens_commit == fit_config["public_jlens"]["official_code_commit"], {
    "actual": actual_jlens_commit,
    "expected": fit_config["public_jlens"]["official_code_commit"],
}

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

base_spec = fit_config["base_model"]
runtime_spec = fit_config["runtime"]
model = globals().get("model")
tokenizer = globals().get("tokenizer")
public_lens = globals().get("lens")
lens_model = globals().get("lens_model")

assert (model is None) == (tokenizer is None), (
    "Partial model/tokenizer state found. Restart the kernel before continuing."
)
assert (public_lens is None) == (lens_model is None), (
    "Partial J-Lens state found. Restart the kernel before continuing."
)

if model is None:
    print("Loading pinned tokenizer and Qwen 3.6 27B from cache...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base_spec["repo_id"], revision=base_spec["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_spec["repo_id"],
        revision=base_spec["revision"],
        dtype=torch.bfloat16,
        attn_implementation=runtime_spec["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model_state_action = "loaded_after_clean_kernel"
else:
    model_state_action = "reused_live_kernel"

tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

text_config = model.config.get_text_config()
assert text_config.hidden_size == base_spec["expected_hidden_size"]
assert text_config.num_hidden_layers == base_spec["expected_num_hidden_layers"]
assert {parameter.device.type for parameter in model.parameters()} == {"cuda"}
assert next(model.parameters()).dtype == torch.bfloat16

def runtime_adapter_name(repo_id: str) -> str:
    return repo_id.replace(".", "_").replace("/", "__")

adapter_names_for_refit = {}
known_maps = [
    globals().get("test_adapter_names", {}),
    globals().get("adapter_names", {}),

]
loaded_peft_names = set(getattr(model, "peft_config", {}))
if not loaded_peft_names:
    model.add_adapter(LoraConfig(target_modules=["q_proj"]), adapter_name="default")
    loaded_peft_names.add("default")

for role, spec in selected_adapters.items():
    word = spec["word"]
    candidates = [mapping.get(word) for mapping in known_maps if mapping.get(word)]
    candidates.append(runtime_adapter_name(spec["repo_id"]))
    runtime_name = next((name for name in candidates if name in loaded_peft_names), None)
    if runtime_name is None:
        runtime_name = runtime_adapter_name(spec["repo_id"])
        print(f"Loading selected adapter {word}: {spec['repo_id']}", flush=True)
        model.load_adapter(
            spec["repo_id"],
            adapter_name=runtime_name,
            adapter_kwargs={"revision": spec["revision"]},
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
        loaded_peft_names.add(runtime_name)
    adapter_names_for_refit[role] = runtime_name

if public_lens is None:
    lens_spec = fit_config["public_jlens"]
    print("Loading pinned public J-Lens from cache...", flush=True)
    public_lens = jlens.JacobianLens.from_pretrained(
        lens_spec["repo_id"],
        filename=lens_spec["filename"],
        revision=lens_spec["revision"],
    )
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False, compile=False)
    lens = public_lens
    lens_state_action = "loaded_after_clean_kernel"
else:
    assert lens_model._hf_model is model, (
        "J-Lens wrapper points to a different model object. Restart the kernel."
    )
    lens_state_action = "reused_live_kernel"

model.eval()
model.enable_adapters()
# PEFT can restore LoRA requires_grad flags when adapters are enabled.
model.requires_grad_(False)
assert not any(parameter.requires_grad for parameter in model.parameters())
assert public_lens.d_model == base_spec["expected_hidden_size"]
assert public_lens.n_prompts == fit_config["public_jlens"]["expected_n_prompts"]
assert len(public_lens.source_layers) == fit_config["public_jlens"]["expected_source_layers"]
assert source_layer in public_lens.source_layers

update_manifest(
    paths,
    model_state_action=model_state_action,
    lens_state_action=lens_state_action,
    effective_dim_batch=fit_config["fit"]["dim_batch"],
)
print(
    {
        "model_state": model_state_action,
        "lens_state": lens_state_action,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "jlens_commit": actual_jlens_commit,
        "public_lens_prompts": public_lens.n_prompts,
        "source_layers_to_refit": source_layers,
        "adapters": adapter_names_for_refit,
        "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
    }
)
        '''
    ),
    markdown(
        r'''
## Official fit recipe and our small-experiment reduction

The pinned [`anthropics/jacobian-lens` README](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/README.md)
says the paper lenses use **1000 sequences of 128 tokens** from a
pretraining-like corpus and that **~100 prompts is usable**. The official
[`fitting.py`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py)
estimator costs one forward pass plus
`ceil(d_model / dim_batch)` backward passes per prompt. The H100 smoke attempts
showed that `dim_batch=8` and `dim_batch=4` exceed 80 GB. The recorded fallback
is `dim_batch=2`; with `d_model=5120`, that is 2560 backward passes per prompt.

We keep the official defaults that matter: WikiText-like neutral data,
128-token truncation, final target layer, and skipping the first 16 positions.
To keep this follow-up within a small budget, we fit only the one source layer
manually fixed after notebook 08 rather than all 63 source layers. This is a
layer-specific lens experiment; it must not be described as a full replacement
for the public J-Lens. One layer reduces checkpoint size, stored matrices and
per-layer gradient handling, but the shared backward passes still dominate, so
runtime does not fall in direct proportion to the number of omitted layers.
The chosen layer index matters too: an earlier source layer backpropagates
through more transformer blocks before reaching it, while a later layer is
usually cheaper. The n=2 smoke therefore remains the authoritative timing
measurement for the exact selected layer.

Conservative H100 planning range before a successful `dim_batch=2` smoke:
**12–32 GPU-hours per 100-prompt adapter fit**, plus roughly 15–40 minutes for
held-out comparison. Stop at the n=2 gate if the measured projection exceeds
the remaining project budget.
The 2-prompt smoke below replaces this estimate with a measured range from the
actual pod.
        '''
    ),
    markdown(
        r'''
## Training data: one frozen neutral corpus for Rock

This streams the pinned `Salesforce/wikitext` `wikitext-103-raw-v1` training
split, deterministically shuffles it, truncates source rows to 2000 characters,
and keeps texts that produce exactly 128 tokens. The first 100 are the Rock
fit corpus; 20 additional rows are held out for sanity inspection. The timing
smoke uses `fit_prompts[:2]`, so those two examples are the beginning of the
same n=100 corpus. Continuing the checkpoint adds prompts 3–100 rather than
starting a different training run.

All 20 Taboo words are rejected by a word-boundary check. The selected texts,
token IDs, dataset revision and hashes are saved once. Re-runs load that exact
artifact rather than sampling again.
        '''
    ),
    code(
        r'''
corpus_spec = fit_config["neutral_corpus"]
corpus_jsonl = paths.raw_dir / "neutral_wikitext_sequences.jsonl"
corpus_manifest_path = paths.result_dir / "neutral_corpus_manifest.json"

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

needed_records = corpus_spec["fit_sequences"] + corpus_spec["heldout_sequences"]
taboo_pattern = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in corpus_spec["taboo_words"]) + r")\b",
    flags=re.IGNORECASE,
)

if corpus_jsonl.exists() or corpus_manifest_path.exists():
    assert corpus_jsonl.exists() and corpus_manifest_path.exists(), (
        "Partial corpus artifacts found; inspect rather than silently resampling."
    )
    corpus_records = [
        json.loads(line)
        for line in corpus_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus_manifest = load_json(corpus_manifest_path)
else:
    from datasets import load_dataset

    stream = load_dataset(
        corpus_spec["repo_id"],
        corpus_spec["subset"],
        split=corpus_spec["split"],
        revision=corpus_spec["revision"],
        streaming=corpus_spec["streaming"],
    ).shuffle(seed=fit_config["seed"], buffer_size=corpus_spec["shuffle_buffer"])

    corpus_records = []
    for shuffled_index, row in enumerate(stream):
        text = str(row[corpus_spec["text_field"]]).strip()[: corpus_spec["max_chars"]]
        if not text or taboo_pattern.search(text):
            continue
        token_ids = lens_model.encode(text, max_length=corpus_spec["sequence_tokens"])[0].tolist()
        if len(token_ids) != corpus_spec["sequence_tokens"]:
            continue
        corpus_records.append(
            {
                "sequence_index": len(corpus_records),
                "shuffled_stream_index": shuffled_index,
                "split_role": (
                    "fit"
                    if len(corpus_records) < corpus_spec["fit_sequences"]
                    else "heldout"
                ),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_ids": [int(token_id) for token_id in token_ids],
                "token_count": len(token_ids),
            }
        )
        if len(corpus_records) == needed_records:
            break
    assert len(corpus_records) == needed_records
    atomic_jsonl(corpus_jsonl, corpus_records)
    corpus_manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "dataset": {
            key: corpus_spec[key]
            for key in ("repo_id", "revision", "subset", "split", "text_field", "max_chars")
        },
        "seed": fit_config["seed"],
        "fit_sequences": corpus_spec["fit_sequences"],
        "heldout_sequences": corpus_spec["heldout_sequences"],
        "sequence_tokens": corpus_spec["sequence_tokens"],
        "excluded_taboo_words": corpus_spec["taboo_words"],
        "records_hash": stable_hash(corpus_records),
        "fit_token_ids_hash": stable_hash(
            [record["token_ids"] for record in corpus_records if record["split_role"] == "fit"]
        ),
    }
    atomic_json(corpus_manifest_path, corpus_manifest)

assert len(corpus_records) == needed_records
assert corpus_manifest["records_hash"] == stable_hash(corpus_records)
assert all(record["token_count"] == corpus_spec["sequence_tokens"] for record in corpus_records)
assert not any(taboo_pattern.search(record["text"]) for record in corpus_records)

for record in corpus_records:
    observed_ids = lens_model.encode(
        record["text"], max_length=corpus_spec["sequence_tokens"]
    )[0].tolist()
    assert observed_ids == record["token_ids"], record["sequence_index"]

fit_prompts = [record["text"] for record in corpus_records if record["split_role"] == "fit"]
neutral_holdout_prompts = [
    record["text"] for record in corpus_records if record["split_role"] == "heldout"
]
assert len(fit_prompts) == corpus_spec["fit_sequences"] == 100
assert len(neutral_holdout_prompts) == corpus_spec["heldout_sequences"] == 20

update_manifest(
    paths,
    status="neutral_corpus_frozen",
    neutral_corpus_manifest=str(corpus_manifest_path.relative_to(PROJECT_ROOT)),
    neutral_corpus_hash=corpus_manifest["records_hash"],
)
display(pd.DataFrame(corpus_records)[["sequence_index", "split_role", "token_count", "text_sha256"]].head())
        '''
    ),
    markdown(
        r'''
## Resumable adapter-specific fitting helper

`jlens.fit` stores a running fp32 Jacobian sum. For one 5120×5120 source-layer
matrix the checkpoint is about 100 MiB; saving every five prompts limits disk
churn while losing at most four prompts after a crash. Each milestone also
saves a compact fp16 lens (~50 MiB).

The official checkpoint does not record the corpus hash or adapter identity,
so this notebook adds a sidecar and refuses to resume if either differs.
        '''
    ),
    code(
        r'''
import jlens

fit_spec = fit_config["fit"]
adapted_lens_dir = paths.lens_dir / "adapted_lenses"
adapted_lens_dir.mkdir(parents=True, exist_ok=True)
timing_dir = paths.result_dir / "fit_timings"
timing_dir.mkdir(parents=True, exist_ok=True)

def activate_adapter(role: str) -> None:
    assert role in adapter_names_for_refit
    model.enable_adapters()
    model.set_adapter(adapter_names_for_refit[role])
    # The lens needs activation gradients, never model/LoRA weight gradients.
    # Freeze after both PEFT calls because either may mark LoRA weights trainable.
    model.requires_grad_(False)
    model.eval()
    assert not any(parameter.requires_grad for parameter in model.parameters())

def fit_identity(role: str) -> dict:
    spec = selected_adapters[role]
    return {
        "schema_version": 1,
        "selection_role": role,
        "adapter_word": spec["word"],
        "adapter_repo_id": spec["repo_id"],
        "adapter_revision": spec["revision"],
        "base_model": fit_config["base_model"],
        "jlens_code_commit": actual_jlens_commit,
        "corpus_hash": corpus_manifest["records_hash"],
        "fit_token_ids_hash": corpus_manifest["fit_token_ids_hash"],
        "source_layers": source_layers,
        "target_layer": fit_spec["target_layer"],
        "dim_batch": fit_spec["dim_batch"],
        "max_seq_len": fit_spec["max_seq_len"],
        "skip_first": fit_spec["skip_first"],
    }

def adapter_fit_paths(role: str, milestone: int) -> tuple[Path, Path, Path]:
    checkpoint = paths.checkpoint_dir / f"{role}_layer{source_layer}_fit_state.pt"
    sidecar = paths.checkpoint_dir / f"{role}_layer{source_layer}_fit_identity.json"
    lens_path = adapted_lens_dir / f"{role}_layer{source_layer}_jlens_n{milestone:04d}.pt"
    return checkpoint, sidecar, lens_path

def checkpoint_n_done(checkpoint: Path) -> int:
    if not checkpoint.exists():
        return 0
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return int(state["n_done"])

def fit_adapter_to(role: str, milestone: int):
    assert role in fit_spec["role_order"]
    assert milestone in fit_spec["milestones"]
    checkpoint, sidecar, lens_path = adapter_fit_paths(role, milestone)
    expected_identity = fit_identity(role)

    if lens_path.exists():
        cached = jlens.JacobianLens.load(str(lens_path))
        assert cached.n_prompts == milestone
        assert cached.source_layers == source_layers
        print(f"Reusing {lens_path.name}")
        return cached

    activate_adapter(role)
    gc.collect()
    torch.cuda.empty_cache()
    start_allocated_gib = torch.cuda.memory_allocated() / 2**30
    assert start_allocated_gib <= fit_spec["max_start_allocated_gib"], {
        "allocated_gib": start_allocated_gib,
        "maximum_gib": fit_spec["max_start_allocated_gib"],
        "action": "Restart the kernel; a failed autograd graph is still resident.",
    }

    if checkpoint.exists():
        assert sidecar.exists(), "Checkpoint exists without its identity sidecar."
        assert load_json(sidecar) == expected_identity, (
            f"Refusing to resume {role}: adapter/corpus/fit identity changed."
        )
    else:
        if sidecar.exists():
            assert load_json(sidecar) == expected_identity, (
                f"Orphan sidecar for {role} belongs to a different fit identity."
            )
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            orphan = sidecar.with_name(f"{sidecar.name}.orphan_{stamp}")
            sidecar.replace(orphan)
            update_manifest(paths, last_archived_orphan_sidecar=str(orphan))
            print(f"Archived orphan sidecar: {orphan.name}")
        atomic_json(sidecar, expected_identity)

    before_n = checkpoint_n_done(checkpoint)
    assert before_n <= milestone, {
        "checkpoint_prompts": before_n,
        "requested_milestone": milestone,
        "hint": "Load an already-saved earlier milestone instead of rewinding a checkpoint.",
    }

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fitted = None
    oom_text = None
    try:
        fitted = jlens.fit(
            lens_model,
            prompts=fit_prompts[:milestone],
            source_layers=source_layers,
            target_layer=fit_spec["target_layer"],
            dim_batch=fit_spec["dim_batch"],
            max_seq_len=fit_spec["max_seq_len"],
            skip_first=fit_spec["skip_first"],
            checkpoint_path=str(checkpoint),
            checkpoint_every=fit_spec["checkpoint_every"],
            resume=True,
        )
    except torch.cuda.OutOfMemoryError as error:
        oom_text = str(error)
        atomic_json(
            paths.result_dir / f"{role}_fit_oom.json",
            {
                "created_utc": utc_now(),
                "role": role,
                "milestone": milestone,
                "dim_batch": fit_spec["dim_batch"],
                "start_allocated_gib": start_allocated_gib,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "error": oom_text,
            },
        )

    if oom_text is not None:
        fitted = None
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            "J-Lens fit ran out of GPU memory. The OOM was saved without retaining "
            "the original autograd traceback; inspect the OOM artifact before retrying."
        ) from None

    assert fitted is not None
    wall_seconds = time.perf_counter() - started
    after_n = fitted.n_prompts

    assert after_n == milestone, (after_n, milestone)
    assert fitted.source_layers == source_layers
    assert fitted.d_model == base_spec["expected_hidden_size"]
    for layer, matrix in fitted.jacobians.items():
        assert matrix.shape == (base_spec["expected_hidden_size"],) * 2
        assert torch.isfinite(matrix).all().item(), layer

    fitted.save(str(lens_path), dtype=torch.float16)
    timing = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "selection_role": role,
        "adapter_word": role_to_word[role],
        "from_n_prompts": before_n,
        "to_n_prompts": after_n,
        "new_prompts": after_n - before_n,
        "wall_seconds": wall_seconds,
        "seconds_per_new_prompt": wall_seconds / max(1, after_n - before_n),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "lens": str(lens_path.relative_to(PROJECT_ROOT)),
    }
    atomic_json(timing_dir / f"{role}_to_n{milestone:04d}.json", timing)
    print(timing)
    return fitted

def load_milestone(role: str, milestone: int):
    lens_path = adapter_fit_paths(role, milestone)[2]
    assert lens_path.exists(), f"Fit {role} to n={milestone} first."
    fitted = jlens.JacobianLens.load(str(lens_path))
    assert fitted.n_prompts == milestone
    return fitted
        '''
    ),
    markdown(
        r'''
## Stage A — Rock timing smoke on 2×128 tokens

This is the first real timing measurement. It fits only the selected source layer, saves a
resumable checkpoint, checks finiteness, and compares one held-out neutral
readout with the public lens. The top tokens need not match; the smoke gate is
only about plumbing, dimensions, memory and runtime.
        '''
    ),
    code(
        r'''
primary_n2 = fit_adapter_to("primary", 2)

activate_adapter("primary")
smoke_prompt = neutral_holdout_prompts[0]
smoke_ids = lens_model.encode(smoke_prompt, max_length=128)
smoke_position = smoke_ids.shape[1] - 2
assert smoke_position >= fit_spec["skip_first"]

with torch.no_grad():
    public_logits, _, _ = public_lens.apply(
        lens_model,
        smoke_prompt,
        layers=source_layers,
        positions=[smoke_position],
        max_seq_len=128,
    )
    adapted_logits, _, _ = primary_n2.apply(
        lens_model,
        smoke_prompt,
        layers=source_layers,
        positions=[smoke_position],
        max_seq_len=128,
    )

layer = source_layer
def decoded_top(logits: torch.Tensor, k: int = 5) -> list[str]:
    return [tokenizer.decode([int(token_id)]) for token_id in logits.topk(k).indices]

smoke_review = {
    "selection_role": "primary",
    "adapter_word": primary_word,
    "fit_prompts": primary_n2.n_prompts,
    "layer": layer,
    "position": smoke_position,
    "public_top5": decoded_top(public_logits[layer][0]),
    "adapted_n2_top5": decoded_top(adapted_logits[layer][0]),
    "public_logits_finite": bool(torch.isfinite(public_logits[layer]).all().item()),
    "adapted_logits_finite": bool(torch.isfinite(adapted_logits[layer]).all().item()),
}
atomic_json(paths.result_dir / "primary_n2_smoke_review.json", smoke_review)
display(smoke_review)

timing_files = sorted(timing_dir.glob("primary_to_n*.json"))
primary_smoke_timing = load_json(timing_files[-1])
seconds_per_prompt = primary_smoke_timing["seconds_per_new_prompt"]
remaining_prompts = 100 - primary_n2.n_prompts
projected_remaining_seconds = seconds_per_prompt * remaining_prompts
projected_total_seconds = primary_smoke_timing["wall_seconds"] + projected_remaining_seconds
rock_n100_time_estimate = {
    "adapter_word": primary_word,
    "source_layer": source_layer,
    "measured_prompts": primary_n2.n_prompts,
    "measured_wall_minutes": primary_smoke_timing["wall_seconds"] / 60,
    "seconds_per_prompt": seconds_per_prompt,
    "remaining_prompts": remaining_prompts,
    "projected_remaining_hours": projected_remaining_seconds / 3600,
    "projected_total_hours": projected_total_seconds / 3600,
    "broad_total_range_hours": [
        projected_total_seconds * 0.75 / 3600,
        projected_total_seconds * 1.50 / 3600,
    ],
    "note": "The n=10 milestone gives a better estimate after warm-up is amortized.",
}
atomic_json(paths.result_dir / "rock_n100_time_estimate.json", rock_n100_time_estimate)
display(rock_n100_time_estimate)
        '''
    ),
    markdown(
        r'''
### Rock n=100 approval gate

Set the flag only after the cell above completed, GPU peak is safe, both
readouts are finite, and the projected one-adapter runtime fits the remaining
research budget. This explicit pause is intentional.
        '''
    ),
    code(
        r'''
APPROVE_ROCK_TO_N100 = False
assert APPROVE_ROCK_TO_N100, "Inspect the Rock n=2 timing and ETA before continuing."

primary_gate = {
    "approved": True,
    "approved_utc": utc_now(),
    "checks": {
        "n2_fit_complete": primary_n2.n_prompts == 2,
        "public_logits_finite": smoke_review["public_logits_finite"],
        "adapted_logits_finite": smoke_review["adapted_logits_finite"],
        "runtime_within_budget": True,
    },
}
assert all(primary_gate["checks"].values())
atomic_json(paths.result_dir / "primary_full_fit_gate.json", primary_gate)
        '''
    ),
    markdown(
        r'''
## Continue Rock through 10, 25, 50, and 100 prompts

Each call resumes the same official checkpoint and saves the milestone lens.
The n=10 timing is a better ETA than n=2 because one-time compilation/warm-up
is amortized.
        '''
    ),
    code(
        r'''
primary_lenses = {2: primary_n2}
for milestone in fit_spec["milestones"][1:]:
    primary_lenses[milestone] = fit_adapter_to("primary", milestone)
primary_n100 = primary_lenses[100]

primary_timings = pd.DataFrame(
    [load_json(path) for path in sorted(timing_dir.glob("primary_to_n*.json"))]
).sort_values("to_n_prompts")
display(primary_timings)

primary_measured_seconds = primary_timings["wall_seconds"].sum()
print(f"Rock measured total: {primary_measured_seconds / 3600:.2f} GPU-hours")
update_manifest(paths, status="primary_n100_fit_complete", primary_fit_prompts=100)
        '''
    ),
    markdown(
        r'''
## Rock convergence and matrix drift

Cosine alone can look high even for under-fitted lenses, so we also report
relative Frobenius distance and distance from the identity. The important
scale comparison is public→n100 drift versus n50→n100 residual convergence.
If they are similar, the apparent adapter change may simply be sampling noise.
        '''
    ),
    code(
        r'''
def matrix_comparison(reference, candidate, *, layer: int) -> dict:
    a = reference.jacobians[layer].float()
    b = candidate.jacobians[layer].float()
    identity = torch.eye(a.shape[0], dtype=torch.float32)
    return {
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()),
        "relative_frobenius": float((b - a).norm().div(a.norm()).item()),
        "reference_identity_distance": float((a - identity).norm().div(identity.norm()).item()),
        "candidate_identity_distance": float((b - identity).norm().div(identity.norm()).item()),
    }

primary_convergence_rows = []
for milestone, milestone_lens in primary_lenses.items():
    primary_convergence_rows.append(
        {
            "selection_role": "primary",
            "adapter_word": primary_word,
            "milestone": milestone,
            "comparison": "milestone_vs_n100",
            **matrix_comparison(primary_n100, milestone_lens, layer=layer),
        }
    )
primary_convergence_rows.append(
    {
        "selection_role": "primary",
        "adapter_word": primary_word,
        "milestone": 100,
        "comparison": "public_n1000_vs_adapter_n100",
        **matrix_comparison(public_lens, primary_n100, layer=layer),
    }
)
primary_convergence = pd.DataFrame(primary_convergence_rows)
primary_convergence.to_csv(paths.result_dir / "primary_matrix_convergence.csv", index=False)
display(primary_convergence)
        '''
    ),
    markdown(
        r'''
## Held-out comparison helper

The generated text is reused from notebook 07, so no new decoding can change
the examples. Those responses come from the 100 published standard TEST
prompts for `rock` in `data/prompts/taboo_published.jsonl`; notebook 07 has
already rendered the prompts and generated the saved responses under the Rock
adapter. For each non-leaking standard TEST response we make one forward
pass with the matching LoRA, capture the selected source layer, and decode the identical
activation through:

- vanilla Logit Lens;
- public base-model J-Lens (n=1000);
- the model's own adapter-specific J-Lens (n=100);
- the other adapter's J-Lens (n=100), once available.

The primary row averages probabilities over generated-response activations.
The exact `gen 5` anchor from validation is saved as a secondary check. Every
emitted token ID is removed before ranks/top-k are computed. Literal own-secret
leaks remain in raw artifacts but are excluded from summaries.
        '''
    ),
    code(
        r'''
from jlens.hooks import ActivationRecorder

evaluation_spec = fit_config["evaluation"]
behavior_path = (
    PROJECT_ROOT
    / "data"
    / "raw_outputs"
    / selection_run_id
    / evaluation_spec["behavior_filename"]
)
assert behavior_path.exists(), (
    f"Missing {behavior_path}. Finish/sync notebook 07 behavior artifacts first."
)
behavior_records = [
    json.loads(line)
    for line in behavior_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
evaluation_behavior = pd.DataFrame(behavior_records)
evaluation_behavior = evaluation_behavior[
    evaluation_behavior["split"].eq(evaluation_spec["split"])
    & evaluation_behavior["prompt_type"].eq(evaluation_spec["prompt_type"])
    & evaluation_behavior["condition"].isin(role_to_word.values())
].copy()
evaluation_behavior = evaluation_behavior.drop_duplicates(
    ["prompt_id", "condition"], keep="last"
).sort_values(["condition", "prompt_id"])
counts = evaluation_behavior.groupby("condition")["prompt_id"].nunique().to_dict()
assert counts == {
    word: evaluation_spec["prompts_per_adapter"] for word in role_to_word.values()
}, counts

def single_token_surface_ids(word: str) -> list[int]:
    forms = (word, f" {word}", word.capitalize(), f" {word.capitalize()}")
    ids = {
        encoded[0]
        for surface in forms
        if len(encoded := tokenizer.encode(surface, add_special_tokens=False)) == 1
    }
    assert ids, f"No single-token surface for {word!r}"
    return sorted(int(token_id) for token_id in ids)

target_ids_by_word = {
    word: single_token_surface_ids(word) for word in role_to_word.values()
}
vocabulary_size = int(lens_model._lm_head.weight.shape[0])
evaluation_cells_dir = paths.lens_dir / "adapter_refit_eval_cells"
evaluation_cells_dir.mkdir(parents=True, exist_ok=True)

def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, destination)

def summarize_distribution(probabilities: torch.Tensor, target_ids: list[int]) -> dict:
    target_tensor = torch.tensor(target_ids, device=probabilities.device, dtype=torch.long)
    target_values = probabilities.index_select(-1, target_tensor)
    available = target_values >= 0
    if not bool(available.any().item()):
        return {
            "target_rank": None,
            "target_reciprocal_rank": None,
            "target_log10_rank": None,
            "target_probability_mass": None,
            "target_hit_top1": False,
            "target_hit_top5": False,
            "top10_json": "[]",
        }
    best_probability = target_values.max()
    target_rank = int((probabilities > best_probability).sum().item()) + 1
    target_mass = float(target_values.clamp_min(0).sum().item())
    top_values, top_indices = probabilities.topk(10)
    top10 = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "probability": float(value),
        }
        for value, token_id in zip(top_values.detach().cpu(), top_indices.detach().cpu())
    ]
    return {
        "target_rank": target_rank,
        "target_reciprocal_rank": 1.0 / target_rank,
        "target_log10_rank": math.log10(target_rank),
        "target_probability_mass": target_mass,
        "target_hit_top1": target_rank <= 1,
        "target_hit_top5": target_rank <= 5,
        "top10_json": json.dumps(top10, ensure_ascii=False),
    }

def available_lenses_for_model(role: str) -> dict[str, object | None]:
    methods = {
        "logit_lens": None,
        "public_base_jlens_n1000": public_lens,
    }
    own_path = adapter_fit_paths(role, 100)[2]
    if own_path.exists():
        methods["own_adapter_jlens_n100"] = jlens.JacobianLens.load(str(own_path))
    other_role = next(candidate for candidate in fit_spec["role_order"] if candidate != role)
    other_path = adapter_fit_paths(other_role, 100)[2]
    if other_path.exists():
        methods["other_adapter_jlens_n100"] = jlens.JacobianLens.load(str(other_path))
    return methods

def evaluate_one_response(
    role: str,
    behavior_row: dict,
    methods: dict[str, object | None],
) -> pd.DataFrame:
    word = behavior_row["condition"]
    assert word == role_to_word[role]
    assert behavior_row["secret"] == word
    activate_adapter(role)

    prompt_ids = [int(token_id) for token_id in behavior_row["prompt_token_ids"]]
    generated_ids = [int(token_id) for token_id in behavior_row["generation_token_ids"]]
    complete_ids = prompt_ids + generated_ids
    assert generated_ids
    prompt_length = len(prompt_ids)
    response_stop = min(
        len(complete_ids), prompt_length + evaluation_spec["response_position_limit"]
    )
    response_positions = list(range(prompt_length, response_stop))
    assert response_positions
    complete_tensor = torch.tensor([complete_ids], device=lens_model.input_device)

    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as recorder:
        lens_model.forward(complete_tensor)
    source = recorder.activations[layer].detach()[0, response_positions].float()
    emitted_ids = sorted({token_id for token_id in generated_ids if 0 <= token_id < vocabulary_size})
    target_ids = target_ids_by_word[word]

    rows = []
    for method, method_lens in methods.items():
        residual = source if method_lens is None else method_lens.transport(source, layer)
        logits = lens_model.unembed(residual).float()
        probabilities = torch.softmax(logits, dim=-1)
        probabilities[:, emitted_ids] = -1.0

        response_average = probabilities.clamp_min(0).mean(dim=0)
        response_average[emitted_ids] = -1.0
        rows.append(
            {
                "run_id": ADAPTED_JLENS_RUN_ID,
                "source_behavior_run_id": behavior_row["run_id"],
                "prompt_id": behavior_row["prompt_id"],
                "prompt_type": behavior_row["prompt_type"],
                "selection_role": role,
                "condition": word,
                "target_word": word,
                "own_secret_leaked": bool(behavior_row["own_secret_leaked"]),
                "method": method,
                "layer": layer,
                "anchor": "response_average",
                "position_from_prompt_end": None,
                "response_positions_counted": len(response_positions),
                "emitted_token_ids_json": json.dumps(emitted_ids),
                **summarize_distribution(response_average, target_ids),
            }
        )

        gen_offset = int(evaluation_spec["anchors"]["gen_5"])
        local_index = gen_offset - 1
        if local_index < len(response_positions):
            rows.append(
                {
                    "run_id": ADAPTED_JLENS_RUN_ID,
                    "source_behavior_run_id": behavior_row["run_id"],
                    "prompt_id": behavior_row["prompt_id"],
                    "prompt_type": behavior_row["prompt_type"],
                    "selection_role": role,
                    "condition": word,
                    "target_word": word,
                    "own_secret_leaked": bool(behavior_row["own_secret_leaked"]),
                    "method": method,
                    "layer": layer,
                    "anchor": "gen_5",
                    "position_from_prompt_end": gen_offset,
                    "response_positions_counted": 1,
                    "emitted_token_ids_json": json.dumps(emitted_ids),
                    **summarize_distribution(probabilities[local_index], target_ids),
                }
            )
        del residual, logits, probabilities, response_average

    del recorder, source, complete_tensor
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)

def evaluate_adapter(role: str) -> pd.DataFrame:
    word = role_to_word[role]
    methods = available_lenses_for_model(role)
    assert "own_adapter_jlens_n100" in methods, f"Fit {role} to n=100 first."
    subset = evaluation_behavior[evaluation_behavior["condition"].eq(word)]
    started = time.perf_counter()
    completed = 0
    for behavior_row in subset.to_dict("records"):
        destination = evaluation_cells_dir / f"{behavior_row['prompt_id']}__{role}.parquet"
        if destination.exists():
            existing_methods = set(pd.read_parquet(destination, columns=["method"])["method"])
            if existing_methods == set(methods):
                continue
        frame = evaluate_one_response(role, behavior_row, methods)
        atomic_parquet(frame, destination)
        completed += 1
        if completed % 10 == 0:
            elapsed = time.perf_counter() - started
            remaining = len(subset) - completed
            eta_minutes = elapsed / completed * remaining / 60
            print(f"{role} ({word}): {completed} new; ETA {eta_minutes:.1f} min", flush=True)

    paths_for_word = sorted(evaluation_cells_dir.glob(f"standard_test_*__{role}.parquet"))
    assert len(paths_for_word) == evaluation_spec["prompts_per_adapter"]
    combined = pd.concat([pd.read_parquet(path) for path in paths_for_word], ignore_index=True)
    combined_path = paths.result_dir / f"{role}_adapter_refit_test_readouts.parquet"
    atomic_parquet(combined, combined_path)
    return combined

evaluation_behavior["selection_role"] = evaluation_behavior["condition"].map(
    {word: role for role, word in role_to_word.items()}
)
display(evaluation_behavior.groupby(["selection_role", "condition"]).agg(
    sequences=("prompt_id", "size"),
    literal_leaks=("own_secret_leaked", "sum"),
    mean_generation_tokens=("generation_token_count", "mean"),
))
        '''
    ),
    markdown(
        r'''
### Rock exploratory evaluation gate

Run only after Rock n=100 is saved and notebook 07 is no longer using the GPU.
This step performs 100 forward passes but does not generate new text.
        '''
    ),
    code(
        r'''
APPROVE_ROCK_TEST_EVALUATION = False
assert APPROVE_ROCK_TEST_EVALUATION, "Confirm notebook 07 is idle before evaluating Rock."
primary_evaluation = evaluate_adapter("primary")
primary_valid = primary_evaluation[~primary_evaluation["own_secret_leaked"]].copy()
primary_summary = (
    primary_valid.groupby(["selection_role", "condition", "anchor", "method"], as_index=False)
    .agg(
        examples=("prompt_id", "nunique"),
        mean_reciprocal_rank=("target_reciprocal_rank", "mean"),
        median_target_rank=("target_rank", "median"),
        recall_at_1=("target_hit_top1", "mean"),
        recall_at_5=("target_hit_top5", "mean"),
    )
)
primary_summary.to_csv(paths.result_dir / "primary_rock_refit_summary.csv", index=False)

paired_own = primary_valid[primary_valid["method"].eq("own_adapter_jlens_n100")][
    ["prompt_id", "anchor", "target_rank", "target_reciprocal_rank"]
].rename(columns={
    "target_rank": "refit_rank",
    "target_reciprocal_rank": "refit_reciprocal_rank",
})
paired_public = primary_valid[primary_valid["method"].eq("public_base_jlens_n1000")][
    ["prompt_id", "anchor", "target_rank", "target_reciprocal_rank"]
].rename(columns={
    "target_rank": "public_rank",
    "target_reciprocal_rank": "public_reciprocal_rank",
})
primary_paired = paired_own.merge(
    paired_public,
    on=["prompt_id", "anchor"],
    validate="one_to_one",
)
primary_paired["delta_reciprocal_rank_refit_minus_public"] = (
    primary_paired["refit_reciprocal_rank"] - primary_paired["public_reciprocal_rank"]
)
primary_paired["refit_rank_better"] = primary_paired["refit_rank"] < primary_paired["public_rank"]
primary_paired["rank_tie"] = primary_paired["refit_rank"] == primary_paired["public_rank"]
primary_paired.to_csv(paths.result_dir / "primary_rock_refit_paired_rows.csv", index=False)
primary_paired_summary = (
    primary_paired.groupby("anchor", as_index=False)
    .agg(
        examples=("prompt_id", "nunique"),
        mean_delta_reciprocal_rank=("delta_reciprocal_rank_refit_minus_public", "mean"),
        refit_wins=("refit_rank_better", "sum"),
        ties=("rank_tie", "sum"),
    )
)
primary_paired_summary["refit_losses"] = (
    primary_paired_summary["examples"]
    - primary_paired_summary["refit_wins"]
    - primary_paired_summary["ties"]
)
primary_paired_summary.to_csv(
    paths.result_dir / "primary_rock_refit_paired_summary.csv", index=False
)
update_manifest(
    paths,
    status="complete",
    experiment_scope="rock_only",
    primary_adapter=selected_adapters["primary"],
    primary_source_layer=source_layer,
    primary_fit_prompts=100,
    primary_evaluation_prompts=evaluation_spec["prompts_per_adapter"],
)
display(primary_summary)
display(primary_paired_summary)
print("Rock evaluation rows:", len(primary_evaluation))
        '''
    ),
    markdown(
        r'''
## Optional Stage B — second adapter, only after Rock worked

**Stop here for the current Rock-only experiment.** The cells below are a
future extension and remain gated. Adding a second adapter should be treated as
a new explicit condition after the Rock result and measured cost are reviewed.

Weak is the manually selected weak-readout model. The same neutral corpus, layer, target layer,
mask and evaluation examples are used. This gate keeps the second fit from
starting automatically before the first adapter is interpretable.
        '''
    ),
    code(
        r'''
RUN_WEAK_N2_SMOKE = False
assert weak_word, "Leave this optional section untouched for the Rock-only experiment."
assert RUN_WEAK_N2_SMOKE, "Run the second model only after reviewing the complete Rock result."
weak_n2 = fit_adapter_to("weak", 2)

activate_adapter("weak")
with torch.no_grad():
    weak_public_logits, _, _ = public_lens.apply(
        lens_model,
        smoke_prompt,
        layers=source_layers,
        positions=[smoke_position],
        max_seq_len=128,
    )
    weak_adapted_logits, _, _ = weak_n2.apply(
        lens_model,
        smoke_prompt,
        layers=source_layers,
        positions=[smoke_position],
        max_seq_len=128,
    )

weak_smoke_review = {
    "selection_role": "weak",
    "adapter_word": weak_word,
    "fit_prompts": weak_n2.n_prompts,
    "layer": layer,
    "position": smoke_position,
    "public_top5": decoded_top(weak_public_logits[layer][0]),
    "adapted_n2_top5": decoded_top(weak_adapted_logits[layer][0]),
    "public_logits_finite": bool(torch.isfinite(weak_public_logits[layer]).all().item()),
    "adapted_logits_finite": bool(torch.isfinite(weak_adapted_logits[layer]).all().item()),
}
atomic_json(paths.result_dir / "weak_n2_smoke_review.json", weak_smoke_review)
display(weak_smoke_review)
        '''
    ),
    markdown(
        r'''
### Weak full-fit gate

Approve only if the Weak n=2 smoke is finite, memory-safe, and its measured
runtime keeps the two-adapter experiment inside budget.
        '''
    ),
    code(
        r'''
APPROVE_WEAK_TO_N100 = False
assert APPROVE_WEAK_TO_N100, "Inspect the Weak n=2 smoke and ETA before continuing."

weak_gate = {
    "approved": True,
    "approved_utc": utc_now(),
    "checks": {
        "primary_n100_complete": adapter_fit_paths("primary", 100)[2].exists(),
        "weak_n2_fit_complete": weak_n2.n_prompts == 2,
        "public_logits_finite": weak_smoke_review["public_logits_finite"],
        "adapted_logits_finite": weak_smoke_review["adapted_logits_finite"],
        "runtime_within_budget": True,
    },
}
assert all(weak_gate["checks"].values())
atomic_json(paths.result_dir / "weak_full_fit_gate.json", weak_gate)
        '''
    ),
    code(
        r'''
weak_lenses = {2: weak_n2}
for milestone in fit_spec["milestones"][1:]:
    weak_lenses[milestone] = fit_adapter_to("weak", milestone)
weak_n100 = weak_lenses[100]

weak_timings = pd.DataFrame(
    [load_json(path) for path in sorted(timing_dir.glob("weak_to_n*.json"))]
).sort_values("to_n_prompts")
display(weak_timings)

weak_measured_seconds = weak_timings["wall_seconds"].sum()
print(f"Weak measured total: {weak_measured_seconds / 3600:.2f} GPU-hours")
update_manifest(paths, status="both_n100_fits_complete", primary_fit_prompts=100, weak_fit_prompts=100)
        '''
    ),
    markdown(
        r'''
## Joint matrix comparison

This adds Primary↔Weak drift and compares both adapters' remaining n50→n100
change with their drift from the public base-model lens.
        '''
    ),
    code(
        r'''
joint_matrix_rows = []
for role, fitted in (("primary", primary_n100), ("weak", weak_n100)):
    n50 = load_milestone(role, 50)
    joint_matrix_rows.extend(
        [
            {
                "selection_role": role,
                "adapter_word": role_to_word[role],
                "comparison": "public_n1000_vs_adapter_n100",
                **matrix_comparison(public_lens, fitted, layer=layer),
            },
            {
                "selection_role": role,
                "adapter_word": role_to_word[role],
                "comparison": "adapter_n50_vs_adapter_n100",
                **matrix_comparison(n50, fitted, layer=layer),
            },
        ]
    )
joint_matrix_rows.append(
    {
        "selection_role": "primary_vs_weak",
        "adapter_word": f"{primary_word}_vs_{weak_word}",
        "comparison": "primary_n100_vs_weak_n100",
        **matrix_comparison(primary_n100, weak_n100, layer=layer),
    }
)
joint_matrix = pd.DataFrame(joint_matrix_rows)
joint_matrix.to_csv(paths.result_dir / "joint_matrix_comparison.csv", index=False)
display(joint_matrix)
        '''
    ),
    markdown(
        r'''
### Weak and cross-lens exploratory evaluation gate

Re-running Rock after Weak exists adds the wrong-adapter-lens control to both
models. Existing per-example cells in the *new run* are atomically replaced
only when their method set is incomplete; model outputs and earlier project
runs are never touched.
        '''
    ),
    code(
        r'''
APPROVE_JOINT_TEST_EVALUATION = False
assert APPROVE_JOINT_TEST_EVALUATION, "Confirm both fits and an idle notebook 07 first."

expected_joint_methods = set(evaluation_spec["methods"])
primary_evaluation = evaluate_adapter("primary")
weak_evaluation = evaluate_adapter("weak")
joint_evaluation = pd.concat([primary_evaluation, weak_evaluation], ignore_index=True)
assert set(joint_evaluation["method"]) == expected_joint_methods
atomic_parquet(joint_evaluation, paths.result_dir / "joint_adapter_refit_test_readouts.parquet")
print("Joint evaluation rows:", len(joint_evaluation))
        '''
    ),
    markdown(
        r'''
## Summarize own-vs-public and own-vs-other results

Headline summaries exclude literal own-secret leaks. Paired differences use
the same prompts and anchor. Positive `delta_rr` means the adapter-specific
lens improved reciprocal rank; positive `delta_log10_rank_reduction` means it
moved the secret closer to rank 1. The bootstrap interval is descriptive, not
a substitute for replication with a larger lens fit.
        '''
    ),
    code(
        r'''
valid_evaluation = joint_evaluation.copy()
if evaluation_spec["exclude_literal_own_secret_leaks"]:
    valid_evaluation = valid_evaluation[~valid_evaluation["own_secret_leaked"]].copy()

summary = (
    valid_evaluation.groupby(
        ["selection_role", "condition", "anchor", "method"], as_index=False
    )
    .agg(
        examples=("prompt_id", "nunique"),
        mean_reciprocal_rank=("target_reciprocal_rank", "mean"),
        median_target_rank=("target_rank", "median"),
        recall_at_1=("target_hit_top1", "mean"),
        recall_at_5=("target_hit_top5", "mean"),
        mean_target_probability=("target_probability_mass", "mean"),
    )
)
summary.to_csv(paths.result_dir / "adapter_refit_test_summary.csv", index=False)
display(summary)

def paired_comparison(frame: pd.DataFrame, condition: str, anchor: str, challenger: str) -> dict:
    subset = frame[frame["condition"].eq(condition) & frame["anchor"].eq(anchor)]
    pivot = subset.pivot(
        index="prompt_id",
        columns="method",
        values=["target_reciprocal_rank", "target_log10_rank", "target_rank"],
    )
    required = ["own_adapter_jlens_n100", challenger]
    for method in required:
        assert method in pivot["target_reciprocal_rank"].columns
    paired = pd.DataFrame(
        {
            "delta_rr": (
                pivot["target_reciprocal_rank"]["own_adapter_jlens_n100"]
                - pivot["target_reciprocal_rank"][challenger]
            ),
            "delta_log10_rank_reduction": (
                pivot["target_log10_rank"][challenger]
                - pivot["target_log10_rank"]["own_adapter_jlens_n100"]
            ),
            "own_rank": pivot["target_rank"]["own_adapter_jlens_n100"],
            "challenger_rank": pivot["target_rank"][challenger],
        }
    ).dropna()
    rng = np.random.default_rng(fit_config["seed"])
    values = paired["delta_rr"].to_numpy()
    bootstrap = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(10000)]
    )
    return {
        "condition": condition,
        "anchor": anchor,
        "challenger": challenger,
        "examples": len(paired),
        "mean_delta_rr": float(values.mean()),
        "delta_rr_ci025": float(np.quantile(bootstrap, 0.025)),
        "delta_rr_ci975": float(np.quantile(bootstrap, 0.975)),
        "mean_delta_log10_rank_reduction": float(paired["delta_log10_rank_reduction"].mean()),
        "own_wins": int((paired["own_rank"] < paired["challenger_rank"]).sum()),
        "ties": int((paired["own_rank"] == paired["challenger_rank"]).sum()),
        "own_losses": int((paired["own_rank"] > paired["challenger_rank"]).sum()),
    }

paired_rows = []
for role, condition in role_to_word.items():
    for anchor in ("response_average", "gen_5"):
        for challenger in ("public_base_jlens_n1000", "other_adapter_jlens_n100"):
            row = paired_comparison(valid_evaluation, condition, anchor, challenger)
            row["selection_role"] = role
            paired_rows.append(row)
paired_summary = pd.DataFrame(paired_rows)
paired_summary.to_csv(paths.result_dir / "adapter_refit_paired_comparisons.csv", index=False)
display(paired_summary)
        '''
    ),
    markdown(
        r'''
## Compact figures

The first figure compares rank-based performance; the second puts matrix drift
and residual convergence on the same scale. These are saved as PNG rather than
kept only in notebook output.
        '''
    ),
    code(
        r'''
response_summary = summary[summary["anchor"].eq("response_average")].copy()
method_order = [
    "logit_lens",
    "public_base_jlens_n1000",
    "own_adapter_jlens_n100",
    "other_adapter_jlens_n100",
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for axis, metric, title in (
    (axes[0], "mean_reciprocal_rank", "Mean reciprocal rank"),
    (axes[1], "recall_at_5", "Recall@5"),
):
    pivot = response_summary.pivot(index="selection_role", columns="method", values=metric)
    pivot = pivot.reindex(columns=method_order)
    pivot.plot(kind="bar", ax=axis)
    axis.set_title(f"{title} — standard TEST, layer {source_layer}")
    axis.set_xlabel("Taboo adapter")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(fontsize=8)
rank_figure = paths.figure_dir / "adapter_refit_test_rank_comparison.png"
paths.figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(rank_figure, dpi=160, bbox_inches="tight")
plt.show()

drift_plot = joint_matrix[
    joint_matrix["comparison"].isin(
        ["public_n1000_vs_adapter_n100", "adapter_n50_vs_adapter_n100"]
    )
].pivot(index="selection_role", columns="comparison", values="relative_frobenius")
ax = drift_plot.plot(kind="bar", figsize=(7, 4))
ax.set_title(f"Layer-{source_layer} matrix drift versus remaining fit convergence")
ax.set_ylabel("Relative Frobenius distance")
ax.set_xlabel("Taboo adapter")
ax.tick_params(axis="x", rotation=0)
matrix_figure = paths.figure_dir / "adapter_refit_matrix_drift.png"
ax.figure.savefig(matrix_figure, dpi=160, bbox_inches="tight")
plt.show()
        '''
    ),
    markdown(
        r'''
## Decision rule and final artifact audit

Treat an adapter-specific refit as promising only if:

1. n50→n100 drift is materially smaller than public→n100 drift;
2. own n100 beats the public n1000 lens on paired held-out readout;
3. own n100 also beats the other adapter's n100 lens;
4. the pattern is not carried by literal leaks and is visible at both the
   response-average and fixed `gen 5` anchor.

If only (2) holds, corpus/sample-size differences remain a plausible
explanation. The next control would be a base-model n100 refit on the identical
corpus. If the two-prompt timing projects beyond the remaining 20-hour budget,
stop after Rock or keep only the selected source layer; do not silently expand to all 63 layers
or 1000 prompts.

Because Rock and the source layer were chosen after inspecting prior results, even a positive result
here remains exploratory until repeated on fresh prompts that did not influence
adapter selection.
        '''
    ),
    code(
        r'''
timing_records = [
    load_json(path) for path in sorted(timing_dir.glob("*_to_n*.json"))
]
timing_table = pd.DataFrame(timing_records).sort_values(["selection_role", "to_n_prompts"])
timing_table.to_csv(paths.result_dir / "all_fit_timings.csv", index=False)

update_manifest(
    paths,
    status="complete",
    completed_utc=utc_now(),
    fitted_adapters=selected_adapters,
    fit_prompts_per_adapter=corpus_spec["fit_sequences"],
    fitted_source_layers=source_layers,
    evaluation_prompts_per_adapter=evaluation_spec["prompts_per_adapter"],
    headline_excludes_literal_leaks=evaluation_spec["exclude_literal_own_secret_leaks"],
)

artifact_inventory = []
for artifact in sorted(
    list(paths.result_dir.glob("*"))
    + list(paths.checkpoint_dir.glob("*"))
    + list(adapted_lens_dir.glob("*.pt"))
    + list(paths.figure_dir.glob("*.png"))
):
    if artifact.is_file():
        artifact_inventory.append(
            {
                "path": str(artifact.relative_to(PROJECT_ROOT)),
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        )
inventory = pd.DataFrame(artifact_inventory)
inventory.to_csv(paths.result_dir / "artifact_inventory.csv", index=False)
display(timing_table)
display(inventory)
print("Experiment complete:", ADAPTED_JLENS_RUN_ID)
        '''
    ),
]


for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:03d}"

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

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
