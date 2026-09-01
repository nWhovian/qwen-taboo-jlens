from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.behavior import behavior_dataframe
from src.experiment_io import RunPaths, read_jsonl


sns.set_theme(style="whitegrid", context="notebook")


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def plot_behavior_summary(paths: RunPaths) -> tuple[plt.Figure, Path, pd.DataFrame]:
    frame = behavior_dataframe(paths)
    if frame.empty:
        raise RuntimeError("Behavior records are missing")
    summary = (
        frame.groupby(["condition", "prompt_type"], as_index=False)
        .agg(
            leak_rate=("own_secret_leaked", "mean"),
            differs_from_base_rate=("differs_from_base", "mean"),
            mean_tokens=("generation_token_count", "mean"),
        )
    )
    long = summary.melt(
        id_vars=["condition", "prompt_type"],
        value_vars=["leak_rate", "differs_from_base_rate"],
        var_name="metric",
        value_name="rate",
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=long, x="condition", y="rate", hue="metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Gold/Blue behavior checks across published prompts")
    ax.set_ylabel("Rate")
    path = _save_figure(fig, paths.figure_dir / "behavior_summary.png")
    summary.to_csv(paths.result_dir / "behavior_summary_for_plot.csv", index=False)
    return fig, path, summary


def plot_sanity(paths: RunPaths) -> tuple[plt.Figure, Path, pd.DataFrame]:
    files = sorted((paths.lens_dir / "sanity").glob("*.jsonl"))
    if not files:
        raise RuntimeError("J-Lens sanity output is missing")
    frame = pd.DataFrame(read_jsonl(files[0]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=frame, x="layer", y="target_rank", hue="method", marker="o", ax=ax)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_title(f"Official base-model sanity: rank of {frame['target'].iloc[0]!r}")
    ax.set_ylabel("Full-vocabulary rank (lower is better)")
    path = _save_figure(fig, paths.figure_dir / "base_jlens_sanity_rank.png")
    frame.to_csv(paths.result_dir / "base_jlens_sanity.csv", index=False)
    return fig, path, frame


def load_lens_frame(paths: RunPaths, columns: list[str] | None = None) -> pd.DataFrame:
    path = paths.result_dir / "lens_readouts.parquet"
    if not path.exists():
        raise RuntimeError(f"Lens parquet is missing: {path}")
    return pd.read_parquet(path, columns=columns)


def headline_metrics(
    paths: RunPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "prompt_id",
        "prompt_type",
        "condition",
        "target_word",
        "method",
        "layer",
        "position_roles_json",
        "own_secret_leaked",
        "target_candidate_rank",
        "target_margin",
        "target_full_rank",
        "predicted_candidate",
    ]
    frame = load_lens_frame(paths, columns)
    frame = frame[frame["target_word"].notna()].copy()
    frame["position_roles"] = frame["position_roles_json"].map(json.loads)
    frame = frame[frame["position_roles"].map(lambda roles: "last_input" in roles)]
    band_start, band_end = config["readout"]["prior_band"]
    frame = frame[frame["layer"].between(band_start, band_end)]
    if config["readout"]["exclude_leaking_outputs_from_headline"]:
        frame = frame[~frame["own_secret_leaked"]]
    frame["candidate_rr"] = 1.0 / frame["target_candidate_rank"]
    frame["candidate_hit1"] = frame["target_candidate_rank"].le(1)
    frame["full_hit5"] = frame["target_full_rank"].le(5)

    per_example = (
        frame.groupby(
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
    summary = (
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
    summary.to_csv(paths.result_dir / "headline_metrics.csv", index=False)

    paired = per_example.pivot_table(
        index=["prompt_id", "prompt_type", "condition", "target_word"],
        columns="method",
        values=["mean_candidate_rr", "mean_target_margin", "best_full_rank"],
    )
    paired.columns = [f"{metric}__{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    for metric in ("mean_candidate_rr", "mean_target_margin"):
        j_col = f"{metric}__jlens"
        l_col = f"{metric}__logit_lens"
        if j_col in paired and l_col in paired:
            paired[f"delta_{metric}_j_minus_logit"] = paired[j_col] - paired[l_col]
    if "best_full_rank__jlens" in paired and "best_full_rank__logit_lens" in paired:
        paired["delta_best_full_rank_logit_minus_j"] = (
            paired["best_full_rank__logit_lens"]
            - paired["best_full_rank__jlens"]
        )
    paired.to_csv(paths.result_dir / "paired_method_comparison.csv", index=False)
    return summary, paired


def adapter_effect_metrics(
    paths: RunPaths,
    config: dict[str, Any],
) -> tuple[plt.Figure, Path, pd.DataFrame, pd.DataFrame]:
    """Compare each target adapter with base on the identical prompt/layer."""

    columns = [
        "prompt_id",
        "prompt_type",
        "condition",
        "method",
        "layer",
        "position_roles_json",
        "own_secret_leaked",
        "gold_logit",
        "blue_logit",
        "gold_full_rank",
        "blue_full_rank",
    ]
    frame = load_lens_frame(paths, columns)
    frame["position_roles"] = frame["position_roles_json"].map(json.loads)
    frame = frame[frame["position_roles"].map(lambda roles: "last_input" in roles)]
    band_start, band_end = config["readout"]["prior_band"]
    frame = frame[frame["layer"].between(band_start, band_end)].copy()

    base = frame[frame["condition"].eq("base")].copy()
    adapters = frame[frame["condition"].isin(("gold", "blue"))].copy()
    if config["readout"]["exclude_leaking_outputs_from_headline"]:
        adapters = adapters[~adapters["own_secret_leaked"]]

    def target_values(row: pd.Series) -> pd.Series:
        target = row["condition"]
        foil = "blue" if target == "gold" else "gold"
        return pd.Series(
            {
                "adapter_target_logit": row[f"{target}_logit"],
                "adapter_margin": row[f"{target}_logit"] - row[f"{foil}_logit"],
                "adapter_target_rank": row[f"{target}_full_rank"],
            }
        )

    adapter_values = adapters.apply(target_values, axis=1)
    adapters = pd.concat([adapters, adapter_values], axis=1)
    base_long = pd.concat(
        [
            base.assign(
                target_word=target,
                base_target_logit=base[f"{target}_logit"],
                base_margin=base[f"{target}_logit"] - base[f"{foil}_logit"],
                base_target_rank=base[f"{target}_full_rank"],
            )
            for target, foil in (("gold", "blue"), ("blue", "gold"))
        ],
        ignore_index=True,
    )
    merged = adapters.merge(
        base_long[
            [
                "prompt_id",
                "method",
                "layer",
                "target_word",
                "base_target_logit",
                "base_margin",
                "base_target_rank",
            ]
        ],
        left_on=["prompt_id", "method", "layer", "condition"],
        right_on=["prompt_id", "method", "layer", "target_word"],
        validate="many_to_one",
    )
    merged["target_logit_lift"] = (
        merged["adapter_target_logit"] - merged["base_target_logit"]
    )
    merged["target_margin_shift"] = merged["adapter_margin"] - merged["base_margin"]
    merged["target_rr_lift"] = (
        1.0 / merged["adapter_target_rank"] - 1.0 / merged["base_target_rank"]
    )
    per_example = (
        merged.groupby(
            ["prompt_id", "prompt_type", "condition", "method"], as_index=False
        )
        .agg(
            mean_target_logit_lift=("target_logit_lift", "mean"),
            mean_target_margin_shift=("target_margin_shift", "mean"),
            mean_target_rr_lift=("target_rr_lift", "mean"),
        )
    )
    summary = (
        per_example.groupby(["condition", "prompt_type", "method"], as_index=False)
        .agg(
            n=("prompt_id", "size"),
            mean_target_logit_lift=("mean_target_logit_lift", "mean"),
            mean_target_margin_shift=("mean_target_margin_shift", "mean"),
            mean_target_rr_lift=("mean_target_rr_lift", "mean"),
        )
    )
    per_example.to_csv(paths.result_dir / "adapter_vs_base_per_example.csv", index=False)
    summary.to_csv(paths.result_dir / "adapter_vs_base_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(
        data=per_example,
        x="condition",
        y="mean_target_margin_shift",
        hue="method",
        errorbar=("ci", 95),
        seed=config["seed"],
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Adapter-specific target margin shift relative to base")
    ax.set_ylabel("(target − foil) adapter minus base")
    path = _save_figure(fig, paths.figure_dir / "adapter_vs_base_margin_shift.png")
    return fig, path, summary, per_example


def plot_layer_curve(
    paths: RunPaths,
    config: dict[str, Any],
) -> tuple[plt.Figure, Path, pd.DataFrame]:
    columns = [
        "prompt_id",
        "condition",
        "target_word",
        "method",
        "layer",
        "position_roles_json",
        "own_secret_leaked",
        "target_full_rank",
    ]
    frame = load_lens_frame(paths, columns)
    frame = frame[frame["target_word"].notna()].copy()
    frame["position_roles"] = frame["position_roles_json"].map(json.loads)
    frame = frame[frame["position_roles"].map(lambda roles: "last_input" in roles)]
    if config["readout"]["exclude_leaking_outputs_from_headline"]:
        frame = frame[~frame["own_secret_leaked"]]
    frame["reciprocal_rank"] = 1.0 / frame["target_full_rank"]
    curve = (
        frame.groupby(["condition", "method", "layer"], as_index=False)
        .agg(mean_reciprocal_rank=("reciprocal_rank", "mean"))
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=curve,
        x="layer",
        y="mean_reciprocal_rank",
        hue="method",
        style="condition",
        ax=ax,
    )
    band_start, band_end = config["readout"]["prior_band"]
    ax.axvspan(band_start, band_end, alpha=0.1, color="grey", label="prior band")
    ax.set_title("Open-vocabulary MRR at the final input token")
    path = _save_figure(fig, paths.figure_dir / "layer_curve_last_input_mrr.png")
    curve.to_csv(paths.result_dir / "layer_curve_last_input_mrr.csv", index=False)
    return fig, path, curve


def plot_sequence_heatmap(
    paths: RunPaths,
    *,
    prompt_id: str,
    condition: str,
    method: str,
) -> tuple[plt.Figure, Path, pd.DataFrame]:
    columns = [
        "prompt_id",
        "condition",
        "method",
        "layer",
        "position",
        "target_margin",
    ]
    frame = load_lens_frame(paths, columns)
    selected = frame[
        frame["prompt_id"].eq(prompt_id)
        & frame["condition"].eq(condition)
        & frame["method"].eq(method)
    ]
    if selected.empty:
        raise RuntimeError(f"No rows for {prompt_id}/{condition}/{method}")
    matrix = selected.pivot(index="layer", columns="position", values="target_margin")
    fig, ax = plt.subplots(figsize=(15, 7))
    bound = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98)) or 1.0
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        vmin=-bound,
        vmax=bound,
        ax=ax,
        cbar_kws={"label": "target logit − foil logit"},
    )
    ax.set_title(f"{method}: {condition} target margin — {prompt_id}")
    filename = f"heatmap_{prompt_id}_{condition}_{method}.png"
    path = _save_figure(fig, paths.figure_dir / filename)
    matrix.to_csv(paths.result_dir / filename.replace(".png", ".csv"))
    return fig, path, matrix


def plot_candidate_confusion(
    paths: RunPaths,
    config: dict[str, Any],
) -> tuple[plt.Figure, Path, pd.DataFrame]:
    columns = [
        "prompt_id",
        "condition",
        "target_word",
        "method",
        "layer",
        "position_roles_json",
        "predicted_candidate",
        "own_secret_leaked",
    ]
    frame = load_lens_frame(paths, columns)
    frame = frame[frame["target_word"].notna()].copy()
    frame["position_roles"] = frame["position_roles_json"].map(json.loads)
    frame = frame[frame["position_roles"].map(lambda roles: "last_input" in roles)]
    band_start, band_end = config["readout"]["prior_band"]
    frame = frame[frame["layer"].between(band_start, band_end)]
    if config["readout"]["exclude_leaking_outputs_from_headline"]:
        frame = frame[~frame["own_secret_leaked"]]
    votes = (
        frame.groupby(["prompt_id", "condition", "method"])["predicted_candidate"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("prediction")
        .reset_index()
    )
    table = pd.crosstab(
        [votes["method"], votes["condition"]], votes["prediction"], normalize="index"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_title("Gold/Blue candidate prediction in the preregistered layer band")
    path = _save_figure(fig, paths.figure_dir / "gold_blue_candidate_confusion.png")
    table.to_csv(paths.result_dir / "gold_blue_candidate_confusion.csv")
    return fig, path, table
