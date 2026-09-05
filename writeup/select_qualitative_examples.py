"""Select reproducible qualitative examples for REPORT.md.

The sample is drawn from every non-leaking held-out answer, not from the
smaller set chosen for the interactive browser.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
NB08 = ROOT / "source_data" / "notebook08"
SEED = 42


def last_user_message(messages: list[dict[str, str]]) -> str:
    return next(item["content"] for item in reversed(messages) if item["role"] == "user")


behavior = pd.read_json(NB08 / "test_behavior_generations.jsonl", lines=True)
behavior = behavior.loc[~behavior["own_secret_leaked"]].copy()
behavior["prompt"] = behavior["messages"].map(last_user_message)

ranks = pd.read_parquet(NB08 / "test_adapter_vs_base_paired_at_anchors.parquet")
ranks = ranks.loc[
    (ranks["layer"] == 40) & (ranks["mask_protocol"] == "global_emitted_ids")
].copy()
ranks = ranks.pivot_table(
    index=["prompt_id", "prompt_type", "candidate_word"],
    columns="method",
    values="adapter_full_vocab_rank",
).reset_index()
ranks = ranks.rename(
    columns={"candidate_word": "condition", "jlens": "J-lens rank", "logit_lens": "Logit-lens rank"}
)

examples = behavior.merge(ranks, on=["prompt_id", "prompt_type", "condition"], validate="one_to_one")
sample = pd.concat(
    [
        examples.loc[examples["prompt_type"] == prompt_type].sample(n=3, random_state=SEED)
        for prompt_type in ["standard", "direct"]
    ],
    ignore_index=True,
)
sample["answer"] = sample["output_text"].str.replace(r"\s+", " ", regex=True).str.strip()
sample["type"] = sample["prompt_type"].map({"standard": "Hint", "direct": "Direct"})
sample = sample[["type", "condition", "prompt", "answer", "J-lens rank", "Logit-lens rank"]]
sample.to_csv(ROOT / "qualitative_examples.csv", index=False)
print(sample.to_string(index=False))
