from src.morphology import MORPHOLOGY_FAMILIES_V1, family_rank_in_token_ids, text_contains_family


def test_all_twenty_families_are_explicit():
    assert len(MORPHOLOGY_FAMILIES_V1) == 20
    assert MORPHOLOGY_FAMILIES_V1["rock"] == ("rock", "rocks")
    assert MORPHOLOGY_FAMILIES_V1["leaf"] == ("leaf", "leaves")


def test_whole_word_leak_detection():
    assert text_contains_family("many rocks here", "rock")
    assert text_contains_family("ROCK!", "rock")
    assert not text_contains_family("rocket", "rock")
    assert not text_contains_family("bedrock", "rock")


def test_truncated_family_rank():
    assert family_rank_in_token_ids([4, 8, 9], [9, 10]) == 3
    assert family_rank_in_token_ids([4, 8, 9], [10]) is None
