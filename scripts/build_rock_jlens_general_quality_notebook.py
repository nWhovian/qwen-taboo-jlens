#!/usr/bin/env python3
"""Build notebook 12: general-quality control for the Rock-specific J-Lens."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "notebooks" / "12_rock_jlens_general_quality_control.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    markdown(
        r"""
# 12 — Is the Rock-specific J-Lens globally worse?

This notebook separates two explanations for the weaker Rock-specific J-Lens:

1. **general-quality explanation:** the `n=100` lens is simply a less accurate
   layer-40 → final-output readout than the public base-model `n=1000` lens;
2. **Rock/LoRA-specific explanation:** its relative failure grows after the Rock
   LoRA is attached, or is unusually large for the secret token itself.

No lens is fitted here. A resumable GPU runner evaluates the two fixed lenses
on the **same token sequences** under a 2 × 2 control:

| sequences | base Qwen | Qwen + Rock LoRA |
|---|---:|---:|
| 20 held-out neutral WikiText sequences | ✓ | ✓ |
| 100 held-out Rock standard-test responses | ✓ | ✓ |

At every audited position, the reference is the model's own final vocabulary
distribution at that position. The primary outcome is the rank, under each
lens, of the token that the full model ranks first. Layer **40** was fixed
before this analysis; this notebook never searches for a better layer.

Statistical unit: **one sequence/prompt**, after averaging its token positions.
Confidence intervals and paired randomization tests never treat token positions
as independent observations.
"""
    ),
    code(
        r"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

PROJECT_ROOT = next(
    path
    for path in (Path.cwd().resolve(), Path.cwd().resolve().parent)
    if (path / "configs" / "adapter_specific_jlens_refit.json").is_file()
)
POINTER_PATH = PROJECT_ROOT / "results" / "latest_rock_jlens_general_quality_control_run.json"
EXPECTED_LAYER = 40
PUBLIC = "public_base_jlens_n1000"
ROCK_N100 = "rock_adapter_jlens_n100"
RANDOM_SEED = 20260904

print("PROJECT_ROOT:", PROJECT_ROOT)
print("control pointer:", POINTER_PATH)
"""
    ),
    markdown(
        r"""
## Load and verify the completed control

The notebook refuses partial data. The long runner writes one atomic Parquet
and `.done.json` per sequence/model condition, then marks the combined run
complete only after every expected cell exists.
"""
    ),
    code(
        r"""
assert POINTER_PATH.is_file(), (
    "Run scripts/run_rock_jlens_general_quality_control.py on the GPU server first."
)
pointer = json.loads(POINTER_PATH.read_text(encoding="utf-8"))
CONTROL_RUN_ID = pointer["run_id"]
RESULT_DIR = PROJECT_ROOT / "results" / CONTROL_RUN_ID
MANIFEST_PATH = RESULT_DIR / "manifest.json"
STATUS_PATH = RESULT_DIR / "general_quality_control_status.json"

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
assert manifest["status"] == "complete", manifest["status"]
assert status["status"] == "complete", status
assert manifest["layer"] == EXPECTED_LAYER
assert manifest["neutral_sequences"] == 20
assert manifest["taboo_sequences"] == 100

raw_path = PROJECT_ROOT / status["output"]
raw = pd.read_parquet(raw_path)

print("CONTROL_RUN_ID:", CONTROL_RUN_ID)
print("REFIT_RUN_ID:", manifest["refit_run_id"])
print("raw rows:", f"{len(raw):,}")
print("completed tasks:", status["completed_tasks"], "/", status["total_tasks"])
display(pd.Series({
    "layer": manifest["layer"],
    "public lens prompts": manifest["public_jlens"]["expected_n_prompts"],
    "Rock-specific lens prompts": 100,
    "neutral sequences": manifest["neutral_sequences"],
    "Taboo sequences": manifest["taboo_sequences"],
    "claim boundary": manifest["claim_boundary"],
}).to_frame("value"))
"""
    ),
    code(
        r"""
key = ["dataset", "sequence_id", "model_condition", "method", "position"]
assert not raw.duplicated(key).any()
assert raw["layer"].eq(EXPECTED_LAYER).all()
assert set(raw["method"]) == {PUBLIC, ROCK_N100}
assert set(raw["model_condition"]) == {"base", "rock_lora"}
assert set(raw["dataset"]) == {"neutral_holdout", "taboo_standard"}

sequence_sets = raw.groupby(["dataset", "model_condition", "method"])["sequence_id"].agg(lambda x: frozenset(x))
for dataset in raw["dataset"].unique():
    reference = sequence_sets.loc[(dataset, "base", PUBLIC)]
    for condition in ("base", "rock_lora"):
        for method in (PUBLIC, ROCK_N100):
            assert sequence_sets.loc[(dataset, condition, method)] == reference

integrity_keys = ["dataset", "model_condition", "method"]
integrity = raw.groupby(integrity_keys).agg(
    sequences=("sequence_id", "nunique"),
    position_rows=("position", "size"),
).reset_index()
leaking_sequences = (
    raw[raw["own_secret_leaked"]]
    .groupby(integrity_keys)["sequence_id"]
    .nunique()
    .rename("leaking_sequences")
    .reset_index()
)
integrity = integrity.merge(leaking_sequences, on=integrity_keys, how="left")
integrity["leaking_sequences"] = integrity["leaking_sequences"].fillna(0).astype(int)
display(integrity)
"""
    ),
    markdown(
        r"""
## Metrics and inference

- **teacher-top1 MRR** (primary): reciprocal rank assigned by the lens to the
  full model's own top-1 token at the same position;
- **Hit@5:** fraction of positions where that token is in the lens top 5;
- **top-10 overlap:** overlap between the lens and full-model top-10 sets;
- **centered-logit cosine:** shape agreement after removing each logit vector's
  mean;
- **forward KL:** distribution mismatch; unlike the other metrics, lower is
  better.

These are general readout-fidelity controls. They do not ask whether the next
sampled token was predictable, and they do not equate decodability with causal
use by the model.
"""
    ),
    code(
        r"""
METRICS = {
    "teacher_top1_reciprocal_rank": {"label": "teacher-top1 MRR", "direction": 1},
    "teacher_top1_hit_at_5": {"label": "teacher-top1 Hit@5", "direction": 1},
    "top10_overlap_share": {"label": "top-10 overlap", "direction": 1},
    "centered_logit_cosine": {"label": "centered-logit cosine", "direction": 1},
    "forward_kl_actual_to_lens": {"label": "forward KL", "direction": -1},
    "rock_target_reciprocal_rank_unmasked": {"label": "Rock-target MRR", "direction": 1},
}

sequence_metrics = raw.groupby(
    ["dataset", "sequence_id", "model_condition", "method"], as_index=False
).agg(
    teacher_top1_reciprocal_rank=("teacher_top1_reciprocal_rank", "mean"),
    teacher_top1_hit_at_5=("teacher_top1_hit_at_5", "mean"),
    top10_overlap_share=("top10_overlap_share", "mean"),
    centered_logit_cosine=("centered_logit_cosine", "mean"),
    forward_kl_actual_to_lens=("forward_kl_actual_to_lens", "mean"),
    rock_target_reciprocal_rank_unmasked=("rock_target_reciprocal_rank_unmasked", "mean"),
    positions=("position", "size"),
    own_secret_leaked=("own_secret_leaked", "max"),
)

def bootstrap_mean_ci(values, *, n_boot=20_000, seed=RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    assert len(values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

def paired_randomization_p(values, *, n_perm=50_000, seed=RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    assert len(values)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    for _ in range(0, n_perm, 5_000):
        count = min(5_000, n_perm - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(values)))
        extreme += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
        completed += count
    return (extreme + 1) / (completed + 1)

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted

method_summary_rows = []
for keys, group in sequence_metrics.groupby(["dataset", "model_condition", "method"]):
    for metric, spec in METRICS.items():
        mean, low, high = bootstrap_mean_ci(group[metric])
        method_summary_rows.append({
            "dataset": keys[0], "model_condition": keys[1], "method": keys[2],
            "metric": metric, "metric_label": spec["label"], "sequences": len(group),
            "mean": mean, "ci_low": low, "ci_high": high,
        })
method_summary = pd.DataFrame(method_summary_rows)

primary_summary = method_summary[
    method_summary["metric"].eq("teacher_top1_reciprocal_rank")
].copy()
display(primary_summary.sort_values(["dataset", "model_condition", "method"]))
"""
    ),
    markdown(
        r"""
## Direct paired comparison of the two lenses

Every delta below is oriented so that **positive = public `n=1000` is better**.
The primary randomization p-values are Holm-corrected across the four
dataset/model cells. The other metrics are supporting diagnostics and should
be read with their confidence intervals, not as a multiple-testing search.
"""
    ),
    code(
        r"""
advantage_rows = []
advantage_values = {}
for (dataset, condition), group in sequence_metrics.groupby(["dataset", "model_condition"]):
    wide = group.pivot(index="sequence_id", columns="method", values=list(METRICS))
    for metric, spec in METRICS.items():
        delta = spec["direction"] * (wide[(metric, PUBLIC)] - wide[(metric, ROCK_N100)])
        mean, low, high = bootstrap_mean_ci(delta)
        advantage_rows.append({
            "dataset": dataset, "model_condition": condition,
            "metric": metric, "metric_label": spec["label"], "sequences": len(delta),
            "public_advantage": mean, "ci_low": low, "ci_high": high,
            "randomization_p": paired_randomization_p(delta),
        })
        advantage_values[(dataset, condition, metric)] = delta

advantage = pd.DataFrame(advantage_rows)
primary_mask = advantage["metric"].eq("teacher_top1_reciprocal_rank")
advantage.loc[primary_mask, "holm_p_primary_4_cells"] = holm_adjust(
    advantage.loc[primary_mask, "randomization_p"]
)
advantage.to_csv(RESULT_DIR / "notebook_public_lens_advantage.csv", index=False)
display(advantage[primary_mask].sort_values(["dataset", "model_condition"]).reset_index(drop=True))
"""
    ),
    code(
        r"""
supporting = advantage[
    advantage["metric"].isin([
        "teacher_top1_hit_at_5", "top10_overlap_share",
        "centered_logit_cosine", "forward_kl_actual_to_lens",
    ])
].copy()
display(
    supporting.pivot_table(
        index=["dataset", "model_condition"], columns="metric_label",
        values=["public_advantage", "ci_low", "ci_high"],
    ).round(4)
)
"""
    ),
    markdown(
        r"""
## Does attaching Rock LoRA change the relative gap?

For each fixed sequence, compute the public-lens advantage under Rock LoRA
minus the public-lens advantage under base Qwen. Positive interaction means the
public lens pulls farther ahead after LoRA; negative means the Rock-specific
lens improves relatively after LoRA.
"""
    ),
    code(
        r"""
interaction_rows = []
for dataset in ("neutral_holdout", "taboo_standard"):
    for metric, spec in METRICS.items():
        base_delta = advantage_values[(dataset, "base", metric)]
        rock_delta = advantage_values[(dataset, "rock_lora", metric)]
        aligned = pd.concat(
            [base_delta.rename("base"), rock_delta.rename("rock_lora")], axis=1, join="inner"
        ).dropna()
        values = aligned["rock_lora"] - aligned["base"]
        mean, low, high = bootstrap_mean_ci(values, seed=RANDOM_SEED + 1)
        interaction_rows.append({
            "dataset": dataset, "metric": metric, "metric_label": spec["label"],
            "sequences": len(values), "lora_interaction": mean,
            "ci_low": low, "ci_high": high,
            "randomization_p": paired_randomization_p(values, seed=RANDOM_SEED + 1),
        })

interaction = pd.DataFrame(interaction_rows)
interaction_primary = interaction[
    interaction["metric"].eq("teacher_top1_reciprocal_rank")
].copy()
interaction_primary["holm_p_primary_2_datasets"] = holm_adjust(interaction_primary["randomization_p"])
interaction.to_csv(RESULT_DIR / "notebook_lora_interactions.csv", index=False)
display(interaction_primary.reset_index(drop=True))
"""
    ),
    markdown(
        r"""
## Is the deficit unusually large for the secret `rock`?

On leak-free Rock-LoRA Taboo responses, compare two paired MRR gaps:

- public − Rock-specific for the full model's ordinary top-1 tokens;
- public − Rock-specific for the target word `rock`.

Their difference is an **excess secret deficit**. Positive values mean that the
Rock-specific lens loses more on `rock` than its general readout-quality loss
would predict. This is evidence about target specificity, still not evidence
that the model causally uses the decoded representation.
"""
    ),
    code(
        r"""
secret_source = sequence_metrics[
    sequence_metrics["dataset"].eq("taboo_standard")
    & sequence_metrics["model_condition"].eq("rock_lora")
    & ~sequence_metrics["own_secret_leaked"]
]
secret_wide = secret_source.pivot(
    index="sequence_id", columns="method",
    values=["teacher_top1_reciprocal_rank", "rock_target_reciprocal_rank_unmasked"],
)
general_gap = (
    secret_wide[("teacher_top1_reciprocal_rank", PUBLIC)]
    - secret_wide[("teacher_top1_reciprocal_rank", ROCK_N100)]
)
rock_gap = (
    secret_wide[("rock_target_reciprocal_rank_unmasked", PUBLIC)]
    - secret_wide[("rock_target_reciprocal_rank_unmasked", ROCK_N100)]
)
excess_secret = rock_gap - general_gap

secret_specific_rows = []
for label, values in {
    "general teacher-top1 public advantage": general_gap,
    "Rock-target public advantage": rock_gap,
    "excess Rock-target deficit": excess_secret,
}.items():
    mean, low, high = bootstrap_mean_ci(values, seed=RANDOM_SEED + 2)
    secret_specific_rows.append({
        "quantity": label, "sequences": len(values), "mean": mean,
        "ci_low": low, "ci_high": high,
        "randomization_p": paired_randomization_p(values, seed=RANDOM_SEED + 2),
    })
secret_specific = pd.DataFrame(secret_specific_rows)
secret_specific.to_csv(RESULT_DIR / "notebook_secret_specificity.csv", index=False)
display(secret_specific)
"""
    ),
    markdown(
        r"""
## Reconnect to the already-completed secret-word evaluation

This is the earlier experiment's emitted-token-masked target-word metric, now
summarized with paired prompt-level confidence intervals. It is different from
the general-fidelity control above and is shown to keep the original empirical
question visible.
"""
    ),
    code(
        r"""
REFIT_RUN_ID = manifest["refit_run_id"]
old_path = RESULT_DIR.parent / REFIT_RUN_ID / "primary_adapter_refit_test_readouts.parquet"
assert old_path.is_file(), old_path
old = pd.read_parquet(old_path)
old = old[
    ~old["own_secret_leaked"]
    & old["method"].isin(["public_base_jlens_n1000", "own_adapter_jlens_n100"])
].copy()

old_secret_rows = []
for anchor, group in old.groupby("anchor"):
    wide = group.pivot(index="prompt_id", columns="method", values="target_reciprocal_rank")
    values = wide["public_base_jlens_n1000"] - wide["own_adapter_jlens_n100"]
    mean, low, high = bootstrap_mean_ci(values, seed=RANDOM_SEED + 3)
    old_secret_rows.append({
        "anchor": anchor, "prompts": len(values),
        "public_minus_Rock_n100_MRR": mean, "ci_low": low, "ci_high": high,
        "randomization_p": paired_randomization_p(values, seed=RANDOM_SEED + 3),
    })
old_secret = pd.DataFrame(old_secret_rows)
display(old_secret)
"""
    ),
    markdown("## Visual summary"),
    code(
        r"""
plot_data = primary_summary.copy()
panels = [("neutral_holdout", "Neutral holdout"), ("taboo_standard", "Taboo standard")]
conditions = ["base", "rock_lora"]
colors = {PUBLIC: "#2563eb", ROCK_N100: "#dc2626"}
labels = {PUBLIC: "public n=1000", ROCK_N100: "Rock-specific n=100"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for axis, (dataset, title) in zip(axes, panels):
    subset = plot_data[plot_data["dataset"].eq(dataset)]
    for method_index, method in enumerate((PUBLIC, ROCK_N100)):
        method_rows = subset[subset["method"].eq(method)].set_index("model_condition").loc[conditions]
        x = np.arange(len(conditions)) + (method_index - 0.5) * 0.22
        y = method_rows["mean"].to_numpy()
        low = method_rows["ci_low"].to_numpy()
        high = method_rows["ci_high"].to_numpy()
        axis.errorbar(
            x, y, yerr=np.vstack([y - low, high - y]), marker="o", capsize=4,
            linewidth=2, color=colors[method], label=labels[method],
        )
    axis.set_xticks(range(len(conditions)), ["base Qwen", "+ Rock LoRA"])
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(bottom=0)
axes[0].set_ylabel("sequence-averaged teacher-top1 MRR")
axes[1].legend(frameon=False)
fig.suptitle("General layer-40 readout fidelity (mean and sequence-bootstrap 95% CI)")
fig.tight_layout()
figure_path = RESULT_DIR / "general_quality_primary_metric.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()
print("saved:", figure_path.relative_to(PROJECT_ROOT))
"""
    ),
    markdown(
        r"""
## Automated evidence statement

The statements below use the pre-specified primary metric and the excess-secret
contrast. They deliberately do **not** turn p-values into probabilities that an
explanation is true. In particular, this design cannot attribute a global
quality gap uniquely to `100` versus `1000` fit prompts because lens corpus and
model condition also differ.
"""
    ),
    code(
        r"""
def evidence_word(low, high, p_value=None):
    if low > 0:
        if p_value is not None and p_value >= 0.05:
            return "positive bootstrap interval, but not significant after the stated correction"
        return "confident positive"
    if high < 0:
        if p_value is not None and p_value >= 0.05:
            return "negative bootstrap interval, but not significant after the stated correction"
        return "confident negative"
    return "uncertain (95% CI crosses zero)"

neutral_base = advantage[
    advantage["dataset"].eq("neutral_holdout")
    & advantage["model_condition"].eq("base")
    & advantage["metric"].eq("teacher_top1_reciprocal_rank")
].iloc[0]
neutral_interaction = interaction_primary[interaction_primary["dataset"].eq("neutral_holdout")].iloc[0]
taboo_interaction = interaction_primary[interaction_primary["dataset"].eq("taboo_standard")].iloc[0]
excess_row = secret_specific[secret_specific["quantity"].eq("excess Rock-target deficit")].iloc[0]

print("1. GLOBAL QUALITY CONTROL")
print(
    f"   Public minus Rock-n100 neutral/base MRR = {neutral_base.public_advantage:+.4f} "
    f"[95% CI {neutral_base.ci_low:+.4f}, {neutral_base.ci_high:+.4f}]; "
    f"Holm p={neutral_base.holm_p_primary_4_cells:.3g}; "
    f"{evidence_word(neutral_base.ci_low, neutral_base.ci_high, neutral_base.holm_p_primary_4_cells)}."
)
print("2. LoRA-SPECIFIC INTERACTION")
print(
    f"   Neutral interaction = {neutral_interaction.lora_interaction:+.4f} "
    f"[{neutral_interaction.ci_low:+.4f}, {neutral_interaction.ci_high:+.4f}]; "
    f"Holm p={neutral_interaction.holm_p_primary_2_datasets:.3g}; "
    f"{evidence_word(neutral_interaction.ci_low, neutral_interaction.ci_high, neutral_interaction.holm_p_primary_2_datasets)}."
)
print(
    f"   Taboo interaction = {taboo_interaction.lora_interaction:+.4f} "
    f"[{taboo_interaction.ci_low:+.4f}, {taboo_interaction.ci_high:+.4f}]; "
    f"Holm p={taboo_interaction.holm_p_primary_2_datasets:.3g}; "
    f"{evidence_word(taboo_interaction.ci_low, taboo_interaction.ci_high, taboo_interaction.holm_p_primary_2_datasets)}."
)
print("3. SECRET-SPECIFIC RESIDUAL")
print(
    f"   Excess Rock-target deficit = {excess_row['mean']:+.4f} "
    f"[{excess_row.ci_low:+.4f}, {excess_row.ci_high:+.4f}]; "
    f"paired p={excess_row.randomization_p:.3g}; "
    f"{evidence_word(excess_row.ci_low, excess_row.ci_high, excess_row.randomization_p)}."
)
print("4. CLAIM LIMIT")
print(
    "   A global deficit supports 'the n=100 lens is generally worse', but it does not prove "
    "that sample count alone caused it. A positive interaction or excess-secret deficit adds "
    "evidence for a Rock/LoRA-specific mismatch beyond that global deficit."
)
"""
    ),
    markdown(
        r"""
## Interpretation limits

- The public lens used 1,000 public neutral fit sequences; the Rock lens used
  100 frozen neutral sequences under Rock LoRA. Sample count, corpus sample,
  and fitted model condition are therefore confounded.
- The neutral holdout has 20 independent sequences. Its intervals honestly
  reflect that limited control sample even though each sequence contains many
  token positions.
- The Taboo control replays fixed observed responses through base and Rock-LoRA
  models. It measures readout fidelity on identical token contexts, not each
  model's free-running behavioral distribution.
- The secret-word analysis excludes literal leaks. Unmasked Rock-target scores
  in the new control and emitted-token-masked scores in the earlier evaluation
  answer related but not identical questions.
- Every result is at the pre-fixed layer 40. No post-hoc layer selection is
  performed.
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Qwen Taboo J-Lens", "language": "python", "name": "qwen-taboo-jlens"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(DESTINATION)
