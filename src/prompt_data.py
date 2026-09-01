from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from src.experiment_io import PROJECT_ROOT, read_jsonl


WORD_BOUNDARY = r"(?<![A-Za-z]){word}(?![A-Za-z])"


def load_prompts(path: str | Path) -> dict[str, dict[str, Any]]:
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = PROJECT_ROOT / absolute
    records = read_jsonl(absolute)
    prompts = {record["prompt_id"]: record for record in records}
    if len(prompts) != len(records):
        raise ValueError(f"Duplicate prompt IDs in {absolute}")
    return prompts


def select_prompts(
    prompts: dict[str, dict[str, Any]], prompt_ids: Iterable[str]
) -> list[dict[str, Any]]:
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in prompts]
    if missing:
        raise KeyError(f"Missing prompt IDs: {missing}")
    return [prompts[prompt_id] for prompt_id in prompt_ids]


def contains_word(text: str, word: str) -> bool:
    return bool(
        re.search(WORD_BOUNDARY.format(word=re.escape(word)), text, flags=re.IGNORECASE)
    )


def lexical_leaks(text: str, words: Iterable[str]) -> list[str]:
    return [word for word in words if contains_word(text, word)]


def render_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    token_ids = tokenizer(
        rendered, add_special_tokens=False, return_attention_mask=False
    ).input_ids
    return rendered, token_ids


def assert_prompt_has_no_candidates(
    rendered_prompt: str, candidate_words: Iterable[str]
) -> None:
    leaks = lexical_leaks(rendered_prompt, candidate_words)
    if leaks:
        raise ValueError(f"Rendered prompt contains candidate word(s): {leaks}")
