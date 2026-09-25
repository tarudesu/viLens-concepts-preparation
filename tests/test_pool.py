"""Fixture-based tests for step 03's pure transformations."""

import json
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

from opencc import OpenCC


pool = import_module("data.build.03_pool")
FIXTURE = Path(__file__).parent / "fixtures" / "pool_cases.json"


def cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_mandarin_splitting_and_script_selection():
    converter = OpenCC("t2s")
    for case in cases()["mandarin"]:
        assert pool.mandarin_candidates(case["input"], "/", converter) == case["expected"]


def test_deduplicates_across_translation_locations():
    case = cases()["duplicate"]
    result = pool.deduplicate_translation_items(
        [(case["item"], location) for location in case["locations"]]
    )
    assert len(result) == 1
    assert result[0]["source_location"] == case["expected_location"]


def test_drop_tag_cleaning():
    case = cases()["drop_tag"]
    result, rule = pool.clean_translation_item(case["item"], {"obsolete"})
    assert result is None
    assert rule == case["expected_rule"]


def test_missing_word_is_skipped_and_counted_without_note_repair():
    case = cases()["missing_word"]
    result, rule = pool.clean_translation_item(case["item"], set())
    assert result is None
    assert rule == case["expected_rule"]
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    pool.count_item_drop(counts, examples, rule, case["item"]["lang_code"], case["item"], "computer", "top")
    assert counts[rule]["vi"] == 1
    assert examples[(rule, "vi")][0]["item"]["note"] == "máy tính"


def test_vietnamese_join_falls_back_to_word():
    case = cases()["join"]
    entries = case["entries"]
    by_pair = {(entry["word"], entry["pos"]): [entry] for entry in entries}
    by_word = {case["word"]: entries}
    joined, join_type = pool.join_vietnamese(case["word"], case["requested_pos"], by_pair, by_word)
    assert joined == entries
    assert join_type == case["expected_type"]
