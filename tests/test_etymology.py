"""Fixture-only tests for step-08 signal classification helpers."""

import json
from importlib import import_module
from pathlib import Path


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
    assert etymology.candidate_failure(True, True, 1, 1) is None
    assert etymology.candidate_failure(False, True, 1, 1) == "reading"
    assert etymology.candidate_failure(True, False, 0, 1) == "not_in_cedict"
    assert etymology.candidate_failure(True, True, 0, 1) == cases["expected_failure"]
    assert etymology.summarize_signal_b([]) == ("nonsino", "nonsino", "no_han_string")
    assert etymology.summarize_signal_b([{"failure_reason": "reading", "reading_ok": False, "headword_found": True}]) == (
        "nonsino", "nonsino", "reading",
    )
    assert etymology.summarize_signal_b([{"failure_reason": "not_in_cedict", "reading_ok": True, "headword_found": False}]) == (
        "nonsino", "nonsino", "not_in_cedict",
    )
    assert etymology.summarize_signal_b([{"failure_reason": "meaning", "reading_ok": True, "headword_found": True}]) == (
        "nonsino", "sino", "meaning",
    )
    assert etymology.summarize_signal_b([{"failure_reason": None, "reading_ok": True, "headword_found": True}]) == (
        "sino", "sino", None,
    )


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
