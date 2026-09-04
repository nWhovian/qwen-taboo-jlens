"""Independent checks for the Rock refit, J-space, and quality-control claims."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
N09 = ROOT / "source_data" / "notebook09"
N10 = ROOT / "source_data" / "notebook10_11"
N12 = ROOT / "source_data" / "notebook12"
PUBLIC = "public_base_jlens_n1000"
ROCK = "rock_adapter_jlens_n100"
SEED = 20260904


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_mean_ci(values: pd.Series, seed: int) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[draws].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def verify_refit() -> None:
    rows = read_csv(N09 / "primary_rock_refit_summary.csv")
    keyed = {(row["anchor"], row["method"]): row for row in rows}
    public = keyed[("gen_5", "public_base_jlens_n1000")]
    own = keyed[("gen_5", "own_adapter_jlens_n100")]
    assert int(public["examples"]) == int(own["examples"]) == 99
    assert np.isclose(float(public["mean_reciprocal_rank"]), 0.7162031308955785)
    assert np.isclose(float(own["mean_reciprocal_rank"]), 0.5200308071579455)
    assert np.isclose(float(public["recall_at_5"]), float(own["recall_at_5"]))
    print(
        "refit gen_5: N=99, public/own MRR "
        f"{float(public['mean_reciprocal_rank']):.6f}/"
        f"{float(own['mean_reciprocal_rank']):.6f}, "
        f"shared R@5={float(public['recall_at_5']):.6f}"
    )


def verify_jspace_morphology() -> None:
    frame = pd.read_parquet(N10 / "jspace_readouts.parquet")
    frame = frame.loc[~frame["own_secret_leaked"]].copy()
    exact_support = 0
    morpheme_support = 0
    exact_top1 = 0
    morpheme_top1 = 0
    top1_counts: Counter[str] = Counter()

    for support_json in frame["support_json"]:
        tokens = [
            item["token"].strip().lower()
            for item in json.loads(support_json)
        ]
        exact_support += "rock" in tokens
        morpheme_support += any("rock" in token for token in tokens)
        top1_counts[tokens[0]] += 1
        exact_top1 += tokens[0] == "rock"
        morpheme_top1 += "rock" in tokens[0]

    assert len(frame) == 99
    assert (exact_support, morpheme_support) == (2, 89)
    assert (exact_top1, morpheme_top1) == (0, 87)
    assert top1_counts.most_common(1) == [("rocks", 87)]
    print(
        "J-space morphology: exact support 2/99; morpheme support 89/99; "
        "top-1 `rocks` 87/99"
    )


def verify_general_quality() -> None:
    raw = pd.read_parquet(N12 / "general_quality_position_metrics.parquet")
    sequence = raw.groupby(
        ["dataset", "sequence_id", "model_condition", "method"], as_index=False
    ).agg(
        teacher=("teacher_top1_reciprocal_rank", "mean"),
        rock=("rock_target_reciprocal_rank_unmasked", "mean"),
        leaked=("own_secret_leaked", "max"),
    )
    source = sequence[
        sequence["dataset"].eq("taboo_standard")
        & sequence["model_condition"].eq("rock_lora")
        & ~sequence["leaked"]
    ]
    wide = source.pivot(
        index="sequence_id", columns="method", values=["teacher", "rock"]
    )
    general_gap = wide[("teacher", PUBLIC)] - wide[("teacher", ROCK)]
    rock_gap = wide[("rock", PUBLIC)] - wide[("rock", ROCK)]
    excess = bootstrap_mean_ci(rock_gap - general_gap, SEED + 2)
    assert len(wide) == 99
    assert np.allclose(
        excess,
        (0.0342194138054025, 0.02572328302736138, 0.042731605842116084),
    )
    print(
        "quality control: excess Rock-specific public advantage "
        f"{excess[0]:.6f} [{excess[1]:.6f}, {excess[2]:.6f}]"
    )


if __name__ == "__main__":
    verify_refit()
    verify_jspace_morphology()
    verify_general_quality()
