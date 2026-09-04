"""Independent, small-path checks of the notebook-08 headline numbers."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data" / "notebook08"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def paper_accuracy(method: str, top_k: int) -> tuple[int, int, float]:
    rows = [
        row
        for row in read_csv("test_paper_metric_units.csv")
        if row["mask_protocol"] == "global_emitted_ids"
        and row["prompt_type"] == "standard"
        and row["method"] == method
        and row["layer"] == "40"
        and row["top_k"] == str(top_k)
    ]
    attempts = sum(int(row["attempts"]) for row in rows)
    hits = round(sum(float(row["accuracy"]) * int(row["attempts"]) for row in rows))
    return hits, attempts, hits / attempts


def candidate_accuracy(method: str) -> tuple[int, float, float, float]:
    rows = [
        row
        for row in read_csv("test_cross_candidate_predictions_at_anchors.csv")
        if row["prompt_type"] == "standard"
        and row["method"] == method
        and row["layer"] == "40"
    ]
    accuracy = sum(row["correct_candidate_20"] == "True" for row in rows) / len(rows)
    ranks = [float(row["true_candidate_rank_20"]) for row in rows]
    shares = [float(row["true_candidate_share_20"]) for row in rows]
    return len(rows), accuracy, statistics.median(ranks), statistics.fmean(shares)


if __name__ == "__main__":
    expected = {
        ("jlens", 1): 0.4100755667506297,
        ("jlens", 5): 0.7959697732997482,
        ("logit_lens", 1): 0.38136020151133504,
        ("logit_lens", 5): 0.6992443324937028,
    }
    for key, expected_value in expected.items():
        hits, attempts, value = paper_accuracy(*key)
        assert abs(value - expected_value) < 1e-12
        print(f"{key}: {hits}/{attempts} = {value:.6f}")

    candidate_expected = {"jlens": 0.9596977329974811, "logit_lens": 0.947103274559194}
    for method, expected_value in candidate_expected.items():
        count, accuracy, median_rank, mean_share = candidate_accuracy(method)
        assert abs(accuracy - expected_value) < 1e-12
        print(
            f"{method} 20-way: N={count}, accuracy={accuracy:.6f}, "
            f"median_rank={median_rank:.1f}, mean_share={mean_share:.6f}"
        )
