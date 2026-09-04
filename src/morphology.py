"""Auditable morphology families for the 20 Taboo target words."""

from __future__ import annotations

import re
from typing import Any, Iterable


MORPHOLOGY_FAMILIES_V1: dict[str, tuple[str, ...]] = {
    "ship": ("ship", "ships"),
    "wave": ("wave", "waves"),
    "song": ("song", "songs"),
    "snow": ("snow", "snows"),
    "rock": ("rock", "rocks"),
    "moon": ("moon", "moons"),
    "jump": ("jump", "jumps"),
    "green": ("green", "greens"),
    "flame": ("flame", "flames"),
    "flag": ("flag", "flags"),
    "dance": ("dance", "dances"),
    "cloud": ("cloud", "clouds"),
    "clock": ("clock", "clocks"),
    "chair": ("chair", "chairs"),
    "salt": ("salt", "salts"),
    "book": ("book", "books"),
    "blue": ("blue", "blues"),
    "gold": ("gold", "golds"),
    "leaf": ("leaf", "leaves"),
    "smile": ("smile", "smiles"),
}


def text_contains_family(text: str, word: str) -> bool:
    """Match a family form as a whole ASCII word, case-insensitively."""

    forms = MORPHOLOGY_FAMILIES_V1[word]
    pattern = r"(?<![A-Za-z])(?:" + "|".join(
        re.escape(form) for form in sorted(forms, key=len, reverse=True)
    ) + r")(?![A-Za-z])"
    return re.search(pattern, text or "", flags=re.IGNORECASE) is not None


def single_token_family_ids(tokenizer: Any, word: str) -> list[int]:
    """Return all single-token space/case variants in a morphology family."""

    token_ids: set[int] = set()
    for form in MORPHOLOGY_FAMILIES_V1[word]:
        for surface in (form, f" {form}", form.capitalize(), f" {form.capitalize()}"):
            encoded = tokenizer.encode(surface, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.add(int(encoded[0]))
    if not token_ids:
        raise ValueError(f"No single-token morphology variants for {word!r}")
    return sorted(token_ids)


def build_family_token_audit(tokenizer: Any, words: Iterable[str]) -> dict[str, Any]:
    """Build the frozen token-ID map and fail if two target families overlap."""

    families: dict[str, Any] = {}
    owners: dict[int, str] = {}
    collisions: list[dict[str, Any]] = []
    for word in words:
        ids = single_token_family_ids(tokenizer, word)
        entries = [{"token_id": token_id, "token": tokenizer.decode([token_id])} for token_id in ids]
        families[word] = {"forms": list(MORPHOLOGY_FAMILIES_V1[word]), "tokens": entries}
        for token_id in ids:
            if token_id in owners and owners[token_id] != word:
                collisions.append(
                    {"token_id": token_id, "first": owners[token_id], "second": word}
                )
            owners[token_id] = word
    if collisions:
        raise ValueError(f"Morphology token families overlap: {collisions}")
    return {"version": "conservative_plural_v1", "families": families, "collisions": []}


def family_rank_in_token_ids(token_ids: Iterable[int], family_ids: Iterable[int]) -> int | None:
    """Return one-based rank of the first token belonging to the target family."""

    wanted = set(int(value) for value in family_ids)
    for rank, token_id in enumerate(token_ids, start=1):
        if int(token_id) in wanted:
            return rank
    return None
