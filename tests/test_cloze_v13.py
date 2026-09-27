"""Focused tests for the v1.3 cloze rules and deterministic choices."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from data.build import cloze_v13

STEP07_PATH = Path(__file__).parents[1] / "data" / "build" / "07_backtranslate.py"
STEP07_SPEC = importlib.util.spec_from_file_location("step07_for_cloze_test", STEP07_PATH)
assert STEP07_SPEC and STEP07_SPEC.loader
step07 = importlib.util.module_from_spec(STEP07_SPEC)
STEP07_SPEC.loader.exec_module(step07)


def _whole_word(needle: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text, flags=re.IGNORECASE) is not None


def test_whitespace_cleanup_and_exact_boundary_masking() -> None:
    assert cloze_v13.clean_example_text("  Hôm nay\n  tôi   học. ") == "Hôm nay tôi học."
    assert cloze_v13.mask_exactly_once("Tôi HỌC ở trường.", "học") == ("Tôi ___ ở trường.", 1)
    assert cloze_v13.mask_exactly_once("Tôi học, rồi học.", "học") == (None, 2)
    assert cloze_v13.mask_exactly_once("họcdance rất vui", "học") == (None, 0)


def test_raw_newline_rule_token_count_and_vietnamese_syllable_gate() -> None:
    assert cloze_v13.is_single_line_raw_example("Tôi học ở trường.")
    assert not cloze_v13.is_single_line_raw_example("Tôi học\nở trường.")
    assert not cloze_v13.is_single_line_raw_example("Tôi học\rở trường.")
    assert cloze_v13.count_query_tokens("… ; Em ___ học ở trường -") == 5
    assert cloze_v13.c5b_unknown_tokens("Em ___ học ở trường", {"em", "học", "ở", "trường"}) == []
    assert cloze_v13.c5b_unknown_tokens("Em ___ học virgin", {"em", "học"}) == ["virgin"]


def test_compound_check_scans_full_headword_windows() -> None:
    assert cloze_v13.compound_headword_conflict("Em ___ học hôm nay", "đuổi", {"đuổi học"}) == "đuổi học"
    assert cloze_v13.compound_headword_conflict("Tôi thấy ___ miếng", "vàng", {"vàng miếng"}) == "vàng miếng"
    assert cloze_v13.compound_headword_conflict("Em “___” học hôm nay", "đuổi", {"đuổi học"}) == "đuổi học"
    assert cloze_v13.compound_headword_conflict("Tôi ___ rồi", "vàng", {"vàng miếng"}) is None
    assert cloze_v13.compound_headword_conflict(
        "___ nhân dân họp hôm nay", "hội đồng", {"hội đồng nhân dân"},
    ) == "hội đồng nhân dân"


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

    entry = {
        "word": "học", "pos": "noun", "senses": [{"glosses": ["study"], "examples": [
            {"text": "Own example."},
            {"text": "Quoted text.", "type": "quotation"},
            {"text": "Cited example.", "ref": "book"},
            {"text": "No source fields."},
        ]}],
    }
    filtered, _ = cloze_v13.collect_example_candidates(
        [fixture["rows"][0]], entries_by_key={"học": [entry]}, contains_whole_word=_whole_word,
    )
    assert [item["c1_allowed"] for item in filtered] == [True, False, False, True]
    assert [item["example_text"] for item in filtered] == [
        "Own example.", "Quoted text.", "Cited example.", "No source fields.",
    ]


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


class _Token:
    def __init__(self, text: str) -> None:
        self.lemma_ = text.casefold()
        self.is_space = False
        self.is_punct = False
        self.head = self


class _Doc(list):
    pass


class _NLP:
    def __call__(self, text: str) -> _Doc:
        return _Doc(_Token(part) for part in text.replace("_", " ").split())


class _Lemma:
    def __init__(self, word: str) -> None:
        self.word = word

    def name(self) -> str:
        return self.word


class _Synset:
    def lemmas(self) -> list[_Lemma]:
        return [_Lemma("cop")]


class _WordNet:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def synsets(self, word: str, *, pos: str) -> list[_Synset]:
        self.calls.append((word, pos))
        return [_Synset()]


def test_c7_uses_exact_head_and_same_pos_wordnet_only() -> None:
    wordnet = _WordNet()
    targets = cloze_v13.english_match_targets(
        "spelling police", pos="noun", nlp=_NLP(), wordnet_reader=wordnet,
    )
    assert wordnet.calls == [("spelling_police", "n")]
    assert cloze_v13.match_c7_output(("police", "arrive"), targets) == {
        "type": "two_word_head_lemma", "word": "police",
    }
    assert cloze_v13.match_c7_output(("cop",), targets) == {
        "type": "wordnet_synset_lemma", "word": "cop",
    }
    assert cloze_v13.match_c7_output(("officer",), targets) is None


def test_named_rebuild_sample_defects_have_strict_rule_guards() -> None:
    # #8 hôm qua: physical line breaks fail before whitespace normalization.
    assert not cloze_v13.is_single_line_raw_example("Hôm qua tôi đi\nđến trường.")
    # #14 núi lửa: a longer fixed headword spanning the blank is detected.
    assert cloze_v13.compound_headword_conflict(
        "___ phun trào hôm nay", "núi lửa", {"núi lửa phun trào"},
    ) == "núi lửa phun trào"
    # #15 trinh nữ and #16 phương pháp: English translation leakage is not VI syllabic text.
    vi_syllables = {"cô", "gái", "trinh", "nữ", "phương", "pháp", "là"}
    assert cloze_v13.c5b_unknown_tokens("___ là virgin", vi_syllables) == ["virgin"]
    assert cloze_v13.c5b_unknown_tokens("___ là method", vi_syllables) == ["method"]
    # #19 tượng đài and #22 hội chứng: absent target terms do not pass C7.
    for english in ("monument", "syndrome"):
        targets = cloze_v13.english_match_targets(
            english, pos="noun", nlp=_NLP(), wordnet_reader=type("EmptyWN", (), {"synsets": lambda *_args, **_kwargs: []})(),
        )
        assert cloze_v13.match_c7_output(("a", "building"), targets) is None
    # #4 chỉ can pass only through an untruncated output containing point.
    point_targets = cloze_v13.english_match_targets(
        "point", pos="noun", nlp=_NLP(), wordnet_reader=type("EmptyWN", (), {"synsets": lambda *_args, **_kwargs: []})(),
    )
    assert cloze_v13.match_c7_output(("point",), point_targets) == {
        "type": "target_lemma", "word": "point",
    }
    assert cloze_v13.match_c7_output(("shower",), point_targets) is None
    assert cloze_v13.match_untruncated_c7_beams(
        [("point",), ("shower",)], [True, False], point_targets,
    ) is None
    assert cloze_v13.match_untruncated_c7_beams(
        [("point",), ("shower",)], [False, False], point_targets,
    ) == {"type": "target_lemma", "word": "point"}


def test_c8_removes_shared_queries_and_old_query_extraction_handles_multiline() -> None:
    candidates = [
        {"candidate_id": "a", "concept_id": "a", "vi": "một", "query": "___ here", "pos": "noun", "split": "test", "m1_extension": False},
        {"candidate_id": "b", "concept_id": "b", "vi": "hai", "query": "___ here", "pos": "verb", "split": "test", "m1_extension": False},
        {"candidate_id": "c", "concept_id": "c", "vi": "ba", "query": "unique ___", "pos": "noun", "split": "test", "m1_extension": False},
    ]
    counts = {item["concept_id"]: {"candidates": 1, "after_C7": 1} for item in candidates}
    kept, failures, counts = cloze_v13.apply_c8(
        candidates, context_candidates=candidates,
        per_concept_counts=counts, prior_failures=[],
    )
    assert [item["concept_id"] for item in kept] == ["c"]
    assert [item["failed_rule"] for item in failures] == ["C8", "C8"]
    assert counts["a"]["after_C8"] == 0
    prompt = "Demo ___\nĐáp án: A\nDemo ___\nĐáp án: B\nDemo ___\nĐáp án: C\nA poem\nline two\nĐáp án:"
    assert cloze_v13.old_query_from_prompt(prompt, "Đáp án:") == "A poem\nline two"


def test_c8_uses_c3_pool_for_xa_hoi_chu_nghia_f22472e471f9_collision() -> None:
    xa_hoi_chu_nghia = {
        "candidate_id": "a5fcf428ddad:0:0:0",
        "concept_id": "a5fcf428ddad",
        "vi": "xã hội chủ nghĩa",
        "query": "Kinh tế thị trường định hướng ___",
        "pos": "noun", "split": "test", "m1_extension": False,
    }
    f22472e471f9 = {
        "candidate_id": "f22472e471f9:0:0:0",
        "concept_id": "f22472e471f9",
        "vi": "chủ nghĩa xã hội",
        "query": "Kinh tế thị trường định hướng ___",
        "pos": "noun", "split": "test", "m1_extension": False,
    }
    counts = {xa_hoi_chu_nghia["concept_id"]: {"candidates": 1, "after_C7": 1}}
    kept, failures, _ = cloze_v13.apply_c8(
        [xa_hoi_chu_nghia],
        context_candidates=[xa_hoi_chu_nghia, f22472e471f9],
        per_concept_counts=counts,
        prior_failures=[],
    )
    assert kept == []
    assert failures[0]["failed_rule"] == "C8"
    assert failures[0]["c8_conflicts"] == [{
        "candidate_id": "f22472e471f9:0:0:0",
        "concept_id": "f22472e471f9",
        "vi": "chủ nghĩa xã hội",
    }]

    same_fill = {**f22472e471f9, "vi": "xã hội chủ nghĩa"}
    kept, failures, _ = cloze_v13.apply_c8(
        [xa_hoi_chu_nghia],
        context_candidates=[xa_hoi_chu_nghia, same_fill],
        per_concept_counts={"a5fcf428ddad": {"candidates": 1, "after_C7": 1}},
        prior_failures=[],
    )
    assert kept == [xa_hoi_chu_nghia]
    assert failures == []


def test_nllb_limit_guard_flags_only_sequences_that_reach_budget() -> None:
    reached_limit = step07.NLLBTranslator.sequence_reached_token_limit
    assert reached_limit([0, 10, 2], eos_id=2, max_new_tokens=2)
    assert not reached_limit([0, 10, 2], eos_id=2, max_new_tokens=3)
    assert reached_limit([0, 10, 11, 12], eos_id=2, max_new_tokens=3)
