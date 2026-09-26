"""Fixture-only tests for step-08 signal classification helpers."""

import json
from importlib import import_module
from pathlib import Path

import pytest

etymology = import_module("data.build.08_etymology")
FIXTURES = Path(__file__).parent / "fixtures"


def fixture_data():
    return json.loads((FIXTURES / "etymology_cases.json").read_text(encoding="utf-8"))


def template_entry(name, args):
    return {"word": "sample", "pos": "noun", "etymology_templates": [{"name": name, "args": args}]}


def test_signal_a_vi_etym_sino_template():
    fixture = fixture_data()
    entry = template_entry("vi-etym-sino", {"1": "行貨"})
    result = etymology.classify_entry_a(entry, fixture["settings"], fixture["cjk_ranges"])
    assert result["label"] == "sino"
    assert result["sino_via"] == ["template"]


def test_signal_a_sino_japanese_with_cjk_argument():
    fixture = fixture_data()
    entry = template_entry("bor", {"1": "vi", "2": "ja", "3": "現実"})
    result = etymology.classify_entry_a(entry, fixture["settings"], fixture["cjk_ranges"])
    assert result["label"] == "sino"
    assert result["sino_via"] == ["japanese_kango"]


def test_signal_a_calque_from_sinitic():
    fixture = fixture_data()
    entry = template_entry("cal", {"1": "vi", "2": "zh", "3": "電腦"})
    result = etymology.classify_entry_a(entry, fixture["settings"], fixture["cjk_ranges"])
    assert result["label"] == "sino"
    assert result["sino_via"] == ["calque"]


def test_signal_a_non_sinitic_loan_is_other_loan():
    fixture = fixture_data()
    result = etymology.classify_entry_a(
        template_entry("bor", {"1": "vi", "2": "tai", "3": "คำ"}),
        fixture["settings"], fixture["cjk_ranges"],
    )
    assert result["label"] == "other_loan"
    assert result["other_loan_sources"] == ["tai"]


def test_signal_a_proto_derivative_is_nonsino_and_native_strict():
    fixture = fixture_data()
    result = etymology.classify_entry_a(
        template_entry("der", {"1": "vi", "2": "mkh-pro", "3": "*may"}),
        fixture["settings"], fixture["cjk_ranges"],
    )
    signal_a, _, _, native_strict = etymology.aggregate_signal_a([result])
    assert signal_a == "nonsino"
    assert result["native_evidence"] is True
    assert native_strict is True


def test_signal_a_homograph_disagreement_is_conflict():
    fixture = fixture_data()
    sino = etymology.classify_entry_a(
        template_entry("vi-etym-sino", {"1": "行貨"}), fixture["settings"], fixture["cjk_ranges"],
    )
    nonsino = etymology.classify_entry_a({"word": "sample", "pos": "noun"}, fixture["settings"], fixture["cjk_ranges"])
    assert etymology.aggregate_signal_a([sino, nonsino])[0] == "conflict"


def test_signal_b_pass_and_all_first_failure_reasons():
    cases = fixture_data()["may_nom_meaning_failure"]
    readings = {cases["han_string"]: {cases["reading"]}}
    assert etymology.reading_matches(cases["han_string"], cases["canonical_vi"], readings, {})
    assert etymology.evaluate_b_candidate(True, True, 1, 1, 1)["primary_pass"] is True
    assert etymology.evaluate_b_candidate(False, True, 1, 1, 1)["failure_reason"] == "reading"
    assert etymology.evaluate_b_candidate(True, False, 0, 1, 1)["failure_reason"] == "not_in_cedict"
    assert etymology.evaluate_b_candidate(True, True, 0, 1, 1)["failure_reason"] == cases["expected_failure"]
    assert etymology.summarize_signal_b([]) == ("nonsino", "nonsino", "nonsino", "no_han_string")
    assert etymology.summarize_signal_b([{
        **etymology.evaluate_b_candidate(False, True, 1, 1, 1), "reading_ok": False, "headword_found": True,
    }]) == ("nonsino", "nonsino", "nonsino", "reading")
    assert etymology.summarize_signal_b([{
        **etymology.evaluate_b_candidate(True, False, 0, 1, 1), "reading_ok": True, "headword_found": False,
    }]) == ("nonsino", "nonsino", "nonsino", "not_in_cedict")
    assert etymology.summarize_signal_b([{
        **etymology.evaluate_b_candidate(True, True, 0, 1, 1), "reading_ok": True, "headword_found": True,
    }]) == ("nonsino", "nonsino", "sino", "meaning")
    assert etymology.summarize_signal_b([{
        **etymology.evaluate_b_candidate(True, True, 1, 1, 1), "reading_ok": True, "headword_found": True,
    }]) == ("sino", "sino", "sino", None)


def test_signal_b_monosyllable_requires_meaning_but_polysyllable_does_not():
    mono = etymology.evaluate_b_candidate(True, True, 0, 1, 1)
    assert mono == {
        "primary_pass": False, "strict_pass": False, "relaxed_pass": True,
        "failure_reason": "meaning",
    }
    multi = etymology.evaluate_b_candidate(True, True, 0, 1, 2)
    assert multi == {
        "primary_pass": True, "strict_pass": False, "relaxed_pass": True,
        "failure_reason": None,
    }


def test_may_nom_case_still_fails_monosyllable_meaning_check():
    case = fixture_data()["may_nom_meaning_failure"]
    target = {case["canonical_vi"], "sew", "can", "might"}
    cedict_gloss_lemmas = {"bury"}
    overlap = etymology.max_content_lemma_overlap(target, [cedict_gloss_lemmas])
    assert overlap == 0
    result = etymology.evaluate_b_candidate(True, True, overlap, 1, 1)
    assert result["primary_pass"] is False
    assert result["failure_reason"] == "meaning"


def test_vi_etym_sino_alternating_arguments_form_one_han_string():
    entry = template_entry("vi-etym-sino", {"1": "良", "2": "good", "3": "心", "4": "heart"})
    strings = etymology.extract_sino_template_strings(
        [entry], fixture_data()["cjk_ranges"], 2, split_on="/", max_combinations=16,
    )
    assert strings == ["良心"]


def test_vi_etym_sino_slash_variants_form_a_bounded_product():
    entry = template_entry("vi-etym-sino", {"1": "電/电", "2": "腦/脑"})
    strings = etymology.extract_sino_template_strings(
        [entry], fixture_data()["cjk_ranges"], 2, split_on="/", max_combinations=16,
    )
    assert strings == sorted(["電腦", "電脑", "电腦", "电脑"])
    capped = etymology.extract_sino_template_strings(
        [entry], fixture_data()["cjk_ranges"], 2, split_on="/", max_combinations=2,
    )
    assert len(capped) == 2


def test_reading_fallback_records_wiktionary_char_source():
    assert etymology.reading_match_source("埋", "mai", {}, {}, {"埋": {"mai"}}) == "wiktionary_char"
    assert etymology.reading_match_source("埋", "mai", {"埋": {"mai"}}, {}, {"埋": {"mai"}}) == "unihan"


def test_single_character_fallback_collects_han_viet_forms_and_head_templates():
    entry = {
        "word": "埋",
        "pos": "character",
        "forms": [{"form": "mai", "tags": ["romanization"]}],
        "head_templates": [{"name": "head", "args": {"tr": "mai"}}],
        "senses": [{"related": [{"word": "mai", "tags": ["han-viet-reading"]}]}],
    }
    assert etymology._single_character_readings(entry, fixture_data()["cjk_ranges"]) == {"mai"}


def test_step08_wordnet_loader_is_local_and_reports_missing_corpus(tmp_path: Path):
    original_path = list(etymology.nltk.data.path)
    try:
        with pytest.raises(RuntimeError, match="Run step 01 with --only wordnet"):
            etymology.load_local_wordnet(tmp_path)
        assert etymology.nltk.data.path == [str(tmp_path.resolve())]
    finally:
        etymology.nltk.data.path = original_path


def test_stratum_mapping_table():
    expected = {
        ("sino", "sino"): "sino",
        ("sino", "nonsino"): "ambiguous",
        ("nonsino", "sino"): "ambiguous",
        ("nonsino", "nonsino"): "nonsino",
        ("other_loan", "sino"): "other_loan",
        ("other_loan", "nonsino"): "other_loan",
        ("conflict", "sino"): "ambiguous",
    }
    assert {key: etymology.classify_stratum(*key) for key in expected} == expected


def test_cohens_kappa_toy_table_and_seeded_bootstrap():
    left = ["sino", "sino", "nonsino", "nonsino"]
    right = ["sino", "nonsino", "nonsino", "nonsino"]
    assert etymology.cohens_kappa(left, right) == 0.5
    first = etymology.bootstrap_kappa_ci(left, right, resamples=100, confidence=0.95, seed=11)
    second = etymology.bootstrap_kappa_ci(left, right, resamples=100, confidence=0.95, seed=11)
    assert first == second
    assert first[0] == 0.5
