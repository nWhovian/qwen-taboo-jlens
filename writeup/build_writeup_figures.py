"""Build the two compact figures used in the short write-up.

The script reads only the curated notebook-08 snapshot in
``writeup/source_data/notebook08``. It does not depend on the large raw run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data" / "notebook08"
FIGURES = ROOT / "figures"
BLUE = "#4C78A8"
ORANGE = "#F28E2B"
GRAY = "#A7A7A7"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_specificity_figure() -> None:
    base_rows = read_csv("test_adapter_vs_base_summary_at_anchors.csv")
    base_rows = [
        row
        for row in base_rows
        if row["prompt_type"] == "standard"
        and row["layer"] == "40"
        and row["mask_protocol"] == "global_emitted_ids"
    ]
    by_method = {row["method"]: row for row in base_rows}

    candidate_rows = read_csv("test_cross_candidate_accuracy_at_anchors.csv")
    candidate_rows = [
        row
        for row in candidate_rows
        if row["prompt_type"] == "standard" and row["layer"] == "40"
    ]
    candidate_by_method = {row["method"]: row for row in candidate_rows}

    methods = ["logit_lens", "jlens"]
    labels = ["Logit Lens", "J-Lens"]
    colors = [BLUE, ORANGE]
    x = [0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))

    width = 0.32
    adapted = [float(by_method[m]["adapter_hit_at_5"]) for m in methods]
    base = [float(by_method[m]["base_hit_at_5"]) for m in methods]
    axes[0].bar([v - width / 2 for v in x], adapted, width, color=colors, label="matching LoRA")
    axes[0].bar(
        [v + width / 2 for v in x],
        base,
        width,
        color="white",
        edgecolor=colors,
        linewidth=2,
        label="base model",
    )
    for i, value in enumerate(adapted):
        axes[0].text(i - width / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=10)
    for i, value in enumerate(base):
        axes[0].text(i + width / 2, value + 0.025, f"{100 * value:.0f}%", ha="center", fontsize=10)
    axes[0].set_title("Target in the full-vocabulary top 5")
    axes[0].set_ylabel("Fraction of non-leaking responses")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, loc="upper left")

    accuracy = [float(candidate_by_method[m]["closed_set_accuracy_20"]) for m in methods]
    axes[1].bar(x, accuracy, width=0.56, color=colors)
    axes[1].axhline(0.05, color=GRAY, linestyle="--", linewidth=1.4)
    for i, value in enumerate(accuracy):
        axes[1].text(i, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=10)
    axes[1].set_title("Correct secret ranked first among 20")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.02)
    axes[1].text(
        0.02,
        0.065,
        "chance: 5%",
        transform=axes[1].get_yaxis_transform(),
        color="#666666",
        fontsize=9,
        va="bottom",
    )

    fig.suptitle(
        "The decoded signal is specific to the loaded Taboo adapter",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Standard test prompts, Qwen layer 40, response average, global emitted-token mask; N = 1,985.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "writeup_specificity_controls.png")


def build_heterogeneity_figure() -> None:
    rows = read_csv("test_metrics_by_adapter_at_anchors.csv")
    rows = [
        row
        for row in rows
        if row["prompt_type"] == "standard" and row["layer"] == "40"
    ]
    by_adapter: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_adapter.setdefault(row["condition"], {})[row["method"]] = row

    values = []
    for adapter, methods in by_adapter.items():
        jlens = float(methods["jlens"]["mrr"])
        logit = float(methods["logit_lens"]["mrr"])
        values.append((adapter, logit, jlens, jlens - logit))
    values.sort(key=lambda item: item[3])

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for y, (_, logit, jlens, _) in enumerate(values):
        ax.plot([logit, jlens], [y, y], color=GRAY, linewidth=1.8, zorder=1)
    ax.scatter([v[1] for v in values], range(len(values)), color=BLUE, s=42, label="Logit Lens", zorder=2)
    ax.scatter([v[2] for v in values], range(len(values)), color=ORANGE, s=42, label="J-Lens", zorder=3)
    ax.set_yticks(range(len(values)), [v[0] for v in values])
    ax.set_xlabel("Mean reciprocal rank of the secret token")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(
        "J-Lens is useful, but not uniformly better across secrets\n"
        "Higher MRR for 11/20 adapters; ordered by J-Lens minus Logit Lens",
        fontweight="bold",
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    save(fig, "writeup_method_heterogeneity.png")


if __name__ == "__main__":
    build_specificity_figure()
    build_heterogeneity_figure()
    print("Wrote compact write-up figures to", FIGURES)
