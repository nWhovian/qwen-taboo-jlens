"""Build the four figures used in ``writeup/REPORT.md``.

The figures use the saved held-out results from notebook 08. They compare the
J-lens with the logit lens, show results across layers and answer positions,
compare the adapted model with base Qwen, and show which secret was predicted
for each adapter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NB08 = ROOT / "source_data" / "notebook08"
FIGURES = ROOT / "figures"

LOGIT = "#A7A7A7"
JLENS = "#4C78A8"
LORA = "#D66B2C"
TEXT = "#222222"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def weighted_mean_and_ci(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float, float]:
    """Bootstrap adapters, retaining each adapter's number of examples."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    estimate = float(np.average(values, weights=weights))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled_values = values[indices]
    sampled_weights = weights[indices]
    samples = np.sum(sampled_values * sampled_weights, axis=1) / np.sum(
        sampled_weights, axis=1
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def draw_errorbar(
    ax: plt.Axes,
    x: float,
    estimate: float,
    low: float,
    high: float,
) -> None:
    ax.errorbar(
        x,
        100 * estimate,
        yerr=[[100 * (estimate - low)], [100 * (high - estimate)]],
        color="#222222",
        capsize=4,
        linewidth=1.4,
        zorder=5,
    )


def label_bar(ax: plt.Axes, x: float, value: float, *, offset: float = 2.0) -> None:
    ax.text(x, 100 * value + offset, f"{100 * value:.1f}%", ha="center", fontsize=10)


def build_overview_bars() -> None:
    frozen = pd.read_csv(NB08 / "test_frozen_metrics_by_adapter.csv")
    frozen = frozen.loc[frozen["prompt_type"] == "standard"].copy()
    assert frozen["condition"].nunique() == 20

    paired = pd.read_parquet(NB08 / "test_adapter_vs_base_paired_at_anchors.parquet")
    paired = paired.loc[
        (paired["prompt_type"] == "standard")
        & (paired["layer"] == 40)
        & (paired["mask_protocol"] == "global_emitted_ids")
    ].copy()
    assert paired["candidate_word"].nunique() == 20

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.26})

    # Panel A: primary method comparison at the frozen layer/position.
    ax = axes[0]
    metrics = [("hit_at_1", "Recall@1"), ("hit_at_5", "Recall@5")]
    method_specs = [
        ("logit_lens", "Logit lens\n(baseline)", LOGIT),
        ("jlens", "Public J-lens", JLENS),
    ]
    width = 0.31
    centers = np.arange(len(metrics), dtype=float)
    offsets = np.array([-width / 2, width / 2])
    for method_index, (method, label, color) in enumerate(method_specs):
        subset = frozen.loc[frozen["method"] == method].sort_values("condition")
        for metric_index, (metric, _) in enumerate(metrics):
            estimate, low, high = weighted_mean_and_ci(
                subset[metric].to_numpy(),
                subset["prompts"].to_numpy(),
                seed=101 + 10 * method_index + metric_index,
            )
            x = centers[metric_index] + offsets[method_index]
            ax.bar(
                x,
                100 * estimate,
                width,
                color=color,
                edgecolor="#555555",
                linewidth=0.7,
                label=label if metric_index == 0 else None,
                zorder=2,
            )
            draw_errorbar(ax, x, estimate, low, high)
            label_bar(ax, x, estimate, offset=2.8)
    ax.set_title("A. J-lens compared with the logit lens\nLayer 40; sixth generated token")
    ax.set_ylabel("Answers (%)")
    ax.set_xticks(centers, [label for _, label in metrics])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    # Panel B: matching adapter versus the same prompts on base Qwen.
    ax = axes[1]
    method_order = ["logit_lens", "jlens"]
    method_names = ["Logit lens", "J-lens"]
    centers = np.arange(2, dtype=float)
    width = 0.32
    for method_index, method in enumerate(method_order):
        subset = paired.loc[paired["method"] == method]
        per_adapter = (
            subset.groupby("candidate_word", as_index=False)
            .agg(
                adapter_rate=("adapter_hit_at_5", "mean"),
                base_rate=("base_hit_at_5", "mean"),
                n=("prompt_id", "size"),
            )
            .sort_values("candidate_word")
        )
        adapter = weighted_mean_and_ci(
            per_adapter["adapter_rate"].to_numpy(),
            per_adapter["n"].to_numpy(),
            seed=301 + method_index,
        )
        base = weighted_mean_and_ci(
            per_adapter["base_rate"].to_numpy(),
            per_adapter["n"].to_numpy(),
            seed=401 + method_index,
        )
        base_x = centers[method_index] - width / 2
        adapter_x = centers[method_index] + width / 2
        ax.bar(
            base_x,
            100 * base[0],
            width,
            color="white",
            edgecolor="#777777",
            hatch="///",
            linewidth=1.2,
            label="Base Qwen" if method_index == 0 else None,
            zorder=2,
        )
        ax.bar(
            adapter_x,
            100 * adapter[0],
            width,
            color=LORA,
            edgecolor="#555555",
            linewidth=0.7,
            label="Matching Taboo adapter" if method_index == 0 else None,
            zorder=2,
        )
        draw_errorbar(ax, base_x, *base)
        draw_errorbar(ax, adapter_x, *adapter)
        label_bar(ax, base_x, base[0], offset=1.5)
        label_bar(ax, adapter_x, adapter[0], offset=2.8)
    ax.set_title("B. Is the secret recovered without the adapter?\nLayer 40; scores averaged over each answer")
    ax.set_ylabel("Recall@5 for the exact secret (%)")
    ax.set_xticks(centers, method_names)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle("J-lens compared with two baselines", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "1,985 answers that did not contain the exact secret. Error bars are 95% intervals across 20 adapters.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.82, bottom=0.16)
    save(fig, "report_overview_bars.png")


def build_layer_curve() -> None:
    units = pd.read_csv(NB08 / "test_paper_metric_units.csv")
    units = units.loc[
        (units["prompt_type"] == "standard")
        & (units["mask_protocol"] == "global_emitted_ids")
        & (units["top_k"] == 5)
    ].copy()
    units["hits"] = units["accuracy"] * units["attempts"]
    per_adapter = (
        units.groupby(["condition", "method", "layer"], as_index=False)
        .agg(hits=("hits", "sum"), n=("attempts", "sum"))
    )
    per_adapter["rate"] = per_adapter["hits"] / per_adapter["n"]
    assert per_adapter["condition"].nunique() == 20

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for method_index, (method, label, color) in enumerate(
        [("logit_lens", "Logit lens (baseline)", LOGIT), ("jlens", "Public J-lens", JLENS)]
    ):
        subset = per_adapter.loc[per_adapter["method"] == method]
        layers = sorted(subset["layer"].unique())
        estimates, lows, highs = [], [], []
        for layer in layers:
            layer_rows = subset.loc[subset["layer"] == layer].sort_values("condition")
            estimate, low, high = weighted_mean_and_ci(
                layer_rows["rate"].to_numpy(),
                layer_rows["n"].to_numpy(),
                seed=700 + 100 * method_index + int(layer),
                draws=4_000,
            )
            estimates.append(100 * estimate)
            lows.append(100 * low)
            highs.append(100 * high)
        ax.plot(layers, estimates, color=color, linewidth=2.5, label=label)
        ax.fill_between(layers, lows, highs, color=color, alpha=0.16, linewidth=0)

    ax.axvline(40, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(40.8, 96, "layer 40\nchosen on validation", va="top", fontsize=9)
    ax.set_title("Recall@5 by model layer", fontweight="bold")
    ax.set_xlabel("Model layer")
    ax.set_ylabel("Recall@5 for the exact secret (%)")
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 62])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        0.01,
        "Held-out answers; scores averaged over each answer; tokens from the answer excluded. Shading shows 95% intervals.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "report_layer_curve_hit5.png")


def build_layer_position_heatmap() -> None:
    data = pd.read_csv(NB08 / "test_standard_exact_position_metrics_pos1_16.csv")
    data = data.loc[data["prompt_type"] == "standard"]
    table = data.pivot(
        index="layer", columns=["position_from_prompt_end", "method"], values="hit_at_5"
    )
    positions = list(range(1, 17))
    layers = sorted(data["layer"].unique())
    delta = np.column_stack(
        [
            100
            * (
                table[(position, "jlens")].reindex(layers).to_numpy()
                - table[(position, "logit_lens")].reindex(layers).to_numpy()
            )
            for position in positions
        ]
    )
    assert delta.shape == (63, 16)

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    limit = 50
    image = ax.imshow(
        delta,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title("Difference in Recall@5: J-lens minus logit lens", fontweight="bold")
    ax.set_xlabel("Position in the generated answer")
    ax.set_ylabel("Model layer")
    ax.set_xticks(range(16), range(1, 17))
    y_ticks = [0, 8, 16, 24, 32, 40, 48, 56, 62]
    ax.set_yticks(y_ticks, y_ticks)
    selected = patches.Rectangle(
        (5 - 0.5, 40 - 0.5),
        1,
        1,
        fill=False,
        edgecolor="#FFD23F",
        linewidth=2.3,
    )
    ax.add_patch(selected)
    ax.annotate(
        "chosen using validation data:\nlayer 40, sixth generated token",
        xy=(5, 40),
        xytext=(8.5, 34),
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.2},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#999999", "alpha": 0.9},
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Difference in Recall@5 (percentage points)")
    fig.text(
        0.5,
        0.01,
        "First 16 tokens of held-out answers. Blue means J-lens is higher; red means the logit lens is higher.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "report_layer_position_heatmap.png")


def build_confusion_matrices() -> None:
    predictions = pd.read_csv(NB08 / "test_cross_candidate_predictions_at_anchors.csv")
    predictions = predictions.loc[
        (predictions["prompt_type"] == "standard") & (predictions["layer"] == 40)
    ].copy()
    secrets = sorted(predictions["actual_adapter"].unique())
    assert len(secrets) == 20

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), sharex=True, sharey=True)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    last_image = None
    for ax, method, title in zip(
        axes,
        ["logit_lens", "jlens"],
        ["Logit lens", "Public J-lens"],
    ):
        subset = predictions.loc[predictions["method"] == method]
        counts = pd.crosstab(subset["actual_adapter"], subset["predicted_candidate_20"])
        counts = counts.reindex(index=secrets, columns=secrets, fill_value=0)
        matrix = 100 * counts.div(counts.sum(axis=1), axis=0)
        accuracy = 100 * float(subset["correct_candidate_20"].mean())
        last_image = ax.imshow(matrix.to_numpy(), cmap="Blues", norm=norm, aspect="equal")
        ax.set_title(f"{title}\n{accuracy:.1f}% correct", fontweight="bold")
        ax.set_xticks(range(20), secrets, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(20), secrets, fontsize=8)
        ax.set_xlabel("Predicted secret word")
        ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Loaded Taboo adapter")
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Responses in row (%)")
    fig.suptitle(
        "Predicted secret word for each loaded adapter",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Held-out answers, layer 40; scores averaged over each answer. Random accuracy among 20 words is 5%.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.10, right=0.90, wspace=0.13)
    save(fig, "report_confusion_matrices.png")


if __name__ == "__main__":
    set_style()
    build_overview_bars()
    build_layer_curve()
    build_layer_position_heatmap()
    build_confusion_matrices()
    print("Wrote four report figures to", FIGURES)
