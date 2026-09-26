"""Fixture-only tests for M1 extension eligibility and disjointness."""

from importlib import import_module

import pytest


m1_extension = import_module("data.build.11b_m1_extension")


def test_eligibility_flags_are_test_only_and_follow_single_token_columns() -> None:
    test_row = {
        "split": "test", "single_token_vi_gemma": True,
        "single_token_vi_qwen": False, "single_token_vi_llama": True,
    }
    assert m1_extension.extension_eligible_flags(test_row, extension=True) == {
        "m1_extension": True, "m1_eligible_gemma": True,
        "m1_eligible_qwen": False, "m1_eligible_llama": True,
    }
    heldout_row = {**test_row, "split": "directions"}
    assert m1_extension.extension_eligible_flags(heldout_row, extension=False) == {
        "m1_extension": False, "m1_eligible_gemma": None,
        "m1_eligible_qwen": None, "m1_eligible_llama": None,
    }


def test_extension_disjointness_uses_vi_orth_key_and_casefolded_english() -> None:
    main = [{"concept_id": "main", "vi_canonical": "hoà", "en_lemma": "Water"}]
    extension = [{"concept_id": "ext", "vi_canonical": "hòa", "en_lemma": "liquid"}]
    with pytest.raises(AssertionError, match="Vietnamese spelling collision"):
        m1_extension.assert_extension_disjoint(main, extension)
    extension[0]["vi_canonical"] = "nước"
    extension[0]["en_lemma"] = "water"
    with pytest.raises(AssertionError, match="English lemma collision"):
        m1_extension.assert_extension_disjoint(main, extension)


def test_extension_disjointness_accepts_distinct_records() -> None:
    main = [{"concept_id": "main", "vi_canonical": "nước", "en_lemma": "water"}]
    extension = [{"concept_id": "ext", "vi_canonical": "lửa", "en_lemma": "fire"}]
    m1_extension.assert_extension_disjoint(main, extension)
