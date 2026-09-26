"""Fixture-only tests for step-12 prompt rendering and masking."""

from importlib import import_module

import pytest


prompts = import_module("data.build.12_prompts")


def test_translation_and_repetition_templates_are_exact() -> None:
    examples = [{"ru": "дом", "vi": "nhà"}, {"ru": "вода", "vi": "nước"}]
    assert prompts.render_translation_prompt(
        examples, ru_test="кошка", vi_test="mèo", ru_label="Русский", vi_label="Tiếng Việt", k=2,
    ) == "Русский: дом - Tiếng Việt: nhà\nРусский: вода - Tiếng Việt: nước\nРусский: кошка - Tiếng Việt:"
    assert prompts.render_repetition_prompt(
        [{"vi": "nhà"}, {"vi": "nước"}], vi_test="mèo", vi_label="Tiếng Việt", k=2,
    ) == "Tiếng Việt: nhà - Tiếng Việt: nhà\nTiếng Việt: nước - Tiếng Việt: nước\nTiếng Việt: mèo - Tiếng Việt:"


def test_leading_space_targets_for_all_five_languages() -> None:
    row = {"en_lemma": "house", "zh_canonical": "房子", "fr_canonical": "maison", "id_canonical": "rumah"}
    targets = prompts.teacher_forcing_targets(row, vi_text="nhà")
    assert targets == {
        "target_vi": " nhà", "target_en": " house", "target_zh": " 房子",
        "target_fr": " maison", "target_id": " rumah",
    }
    assert all(value.startswith(" ") for value in targets.values())


def test_cloze_prompt_and_vietnamese_orthographic_masking() -> None:
    assert prompts.mask_canonical_occurrence("Tôi rất yêu hoà, nhất là mùa hè.", "hòa") == "Tôi rất yêu ___, nhất là mùa hè."
    assert prompts.mask_canonical_occurrence("Tôi uống cà phê mỗi sáng.", "cà phê") == "Tôi uống ___ mỗi sáng."
    assert prompts.mask_canonical_occurrence("Tôi học tiếng Việt.", "nhà") is None
    assert prompts.render_cloze_prompt(
        [{"sentence": "Tôi đến ___.", "answer": "nhà"}],
        sentence="Cô ấy ___ một cuốn sách.", answer_label="Đáp án:", k=1,
    ) == "Tôi đến ___.\nĐáp án: nhà\nCô ấy ___ một cuốn sách.\nĐáp án:"


def test_prompt_helpers_require_configured_example_count() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        prompts.render_repetition_prompt([{"vi": "nhà"}], vi_test="nước", vi_label="Tiếng Việt", k=2)


def test_test_concepts_cannot_appear_in_fewshot_pool() -> None:
    target = [{"concept_id": "test-1", "split": "test"}]
    examples = [{"concept_id": "test-1", "split": "fewshot_reservoir"}]
    with pytest.raises(AssertionError, match="TEST concepts"):
        prompts._assert_no_test_fewshot_overlap(target, examples)
    prompts._assert_no_test_fewshot_overlap(target, [{"concept_id": "few-1", "split": "fewshot_reservoir"}])


def test_russian_candidate_selection_uses_stress_free_yo_equivalence() -> None:
    assert prompts.select_russian_candidate("кни́га", [], ["Книга!"]) == ("книга", True, False, [])
    assert prompts.select_russian_candidate("ёж", [], ["еж."]) == ("ёж", True, False, [])
    assert prompts.select_russian_candidate("ё́ж", [], ["еж"]) == ("ёж", True, False, [])


@pytest.mark.parametrize(
    ("eligible", "expected"),
    [(20, (20, 4)), (24, (20, 4)), (25, (25, 5)), (29, (25, 5)), (30, (30, 6)), (42, (30, 6))],
)
def test_adaptive_fewshot_size(eligible: int, expected: tuple[int, int]) -> None:
    assert prompts.fewshot_selection_size(
        eligible, requested_n=30, set_size=5, max_sets=6, minimum_n=20,
    ) == expected


def test_adaptive_fewshot_stops_below_20() -> None:
    with pytest.raises(ValueError, match="minimum is 20"):
        prompts.fewshot_selection_size(19, requested_n=30, set_size=5, max_sets=6, minimum_n=20)


def test_cloze_demonstrations_use_seeded_selected_pool_across_sets() -> None:
    selected = [
        {"concept_id": "a", "vi_canonical": "nhà"},
        {"concept_id": "b", "vi_canonical": "nước"},
        {"concept_id": "c", "vi_canonical": "mèo"},
    ]
    examples = prompts.select_cloze_demonstrations(
        selected, {"a": None, "b": "Tôi uống nước.", "c": "Con mèo ngủ."}, k=2,
    )
    assert examples == [
        {"sentence": "Tôi uống ___.", "answer": "nước"},
        {"sentence": "Con ___ ngủ.", "answer": "mèo"},
    ]
