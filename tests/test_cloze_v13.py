"""Focused tests for the v1.3 cloze rules and deterministic choices."""

from __future__ import annotations

import json
import re
from pathlib import Path

from data.build import cloze_v13


def _whole_word(needle: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text, flags=re.IGNORECASE) is not None


def test_whitespace_cleanup_and_exact_boundary_masking() -> None:
    assert cloze_v13.clean_example_text("  Hôm nay\n  tôi   học. ") == "Hôm nay tôi học."
    assert cloze_v13.mask_exactly_once("Tôi HỌC ở trường.", "học") == ("Tôi ___ ở trường.", 1)
    assert cloze_v13.mask_exactly_once("Tôi học, rồi học.", "học") == (None, 2)
    assert cloze_v13.mask_exactly_once("họcdance rất vui", "học") == (None, 0)


def test_compound_check_uses_both_adjacent_syllables() -> None:
    assert cloze_v13.compound_headword_conflict("Em ___ học hôm nay", "đuổi", {"đuổi học"}) == "đuổi học"
    assert cloze_v13.compound_headword_conflict("Tôi thấy ___ miếng", "vàng", {"vàng miếng"}) == "vàng miếng"
    assert cloze_v13.compound_headword_conflict("Em “___” học hôm nay", "đuổi", {"đuổi học"}) == "đuổi học"
    assert cloze_v13.compound_headword_conflict("Tôi ___ rồi", "vàng", {"vàng miếng"}) is None


def test_c1_uses_aligned_senses_and_single_sense_fallback() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "cloze_v13_entries.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    candidates, empty = cloze_v13.collect_example_candidates(
        fixture["rows"], entries_by_key=fixture["entries_by_key"], contains_whole_word=_whole_word,
    )
    by_id = {}
    for candidate in candidates:
        by_id.setdefault(candidate["concept_id"], []).append(candidate)
    assert [candidate["c1_allowed"] for candidate in by_id["aligned"]] == [False, True]
    assert by_id["fallback"][0]["c1_allowed"] is True
    assert empty == {"aligned": 2, "fallback": 1}


def test_lemma_matching_and_example_selection_are_deterministic() -> None:
    assert cloze_v13.contains_lemma_sequence(("the", "ground", "is", "wet"), ("ground",))
    assert cloze_v13.contains_lemma_sequence(("earth", "soil"), ("earth", "soil"))
    assert not cloze_v13.contains_lemma_sequence(("earth", "soil"), ("soil", "earth"))

    candidates = [
        {"concept_id": "c", "token_count": 9, "query": "z longest preferred"},
        {"concept_id": "c", "token_count": 8, "query": "b preferred"},
        {"concept_id": "c", "token_count": 8, "query": "a preferred"},
        {"concept_id": "d", "token_count": 5, "query": "z longest"},
        {"concept_id": "d", "token_count": 5, "query": "a longest"},
    ]
    chosen = cloze_v13.choose_examples(candidates, preferred_min_tokens=8)
    assert chosen["c"]["query"] == "a preferred"
    assert chosen["d"]["query"] == "a longest"


def test_c8_removes_shared_queries_and_old_query_extraction_handles_multiline() -> None:
    candidates = [
        {"candidate_id": "a", "concept_id": "a", "query": "___ here", "pos": "noun", "split": "test", "m1_extension": False},
        {"candidate_id": "b", "concept_id": "b", "query": "___ here", "pos": "verb", "split": "test", "m1_extension": False},
        {"candidate_id": "c", "concept_id": "c", "query": "unique ___", "pos": "noun", "split": "test", "m1_extension": False},
    ]
    counts = {item["concept_id"]: {"candidates": 1, "after_C7": 1} for item in candidates}
    kept, failures, counts = cloze_v13.apply_c8(candidates, per_concept_counts=counts, prior_failures=[])
    assert [item["concept_id"] for item in kept] == ["c"]
    assert [item["failed_rule"] for item in failures] == ["C8", "C8"]
    assert counts["a"]["after_C8"] == 0
    prompt = "Demo ___\nĐáp án: A\nDemo ___\nĐáp án: B\nDemo ___\nĐáp án: C\nA poem\nline two\nĐáp án:"
    assert cloze_v13.old_query_from_prompt(prompt, "Đáp án:") == "A poem\nline two"
