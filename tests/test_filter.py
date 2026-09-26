from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "data" / "build" / "06_filter.py"
spec = importlib.util.spec_from_file_location("step06_filter", MODULE_PATH)
assert spec and spec.loader
step06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step06)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "filter_cases.json").read_text(encoding="utf-8"))


def _surface_config() -> dict:
    return {
        "filters": {
            "zh_forbid_latin": True,
            "zh_adj_strip_de": True,
        }
    }


def test_surface_norm_and_zero_distance_loan_orthography() -> None:
    settings = {"remove_chars": [" ", "-", "'"], "vi_replace": {"ph": "f"}}
    assert step06.surface_norm("cà phê", vietnamese=True, **settings) == "cafe"
    assert step06.surface_norm("ra-đi-ô", vietnamese=True, **settings) == "radio"
    assert step06.normalized_levenshtein("cafe", step06.surface_norm("café", vietnamese=False, **settings)) == 0.0
    assert step06.normalized_levenshtein("radio", step06.surface_norm("radio", vietnamese=False, **settings)) == 0.0


def test_surface_distance_examples_and_minimum_across_candidates() -> None:
    settings = {"remove_chars": [" ", "-", "'"], "vi_replace": {"ph": "f"}}
    values = {"en": ["computer"], "fr": ["ordinateur"], "id": ["komputer"]}
    distance, _, _, _ = step06.surface_comparisons("máy tính", values, norm_config=settings)
    assert distance >= 0.30


def test_zh_latin_filter_and_adj_de_preference() -> None:
    config = _surface_config()
    cache: dict[str, float] = {}
    canonical, alts, derived, removed = step06.canonical_language(["猫", "猫A", "猫2"], language="zh", pos="noun", config=config, zipf=cache)
    assert canonical == "猫"
    assert removed == 2
    canonical, alts, derived, _ = step06.canonical_language(["聪明的", "智慧"], language="zh", pos="adj", config=config, zipf={})
    assert canonical in {"聪明的", "智慧"} # Zipf rank is data-driven; plain form is canonical preference.
    assert canonical == "智慧"
    assert "聪明的" in alts
    assert not derived
    canonical, _, derived, _ = step06.canonical_language(["漂亮的"], language="zh", pos="adj", config=config, zipf={})
    assert canonical == "漂亮"
    assert derived


def test_canonical_vi_rank_and_homograph_sense_sum() -> None:
    candidates = [
        {"word": "mèo", "n_sources": 2, "external_attested": False, "n_vi_entries": 1, "n_senses_vi": [1]},
        {"word": "con mèo", "n_sources": 3, "external_attested": True, "n_vi_entries": 2, "n_senses_vi": [1, 3]},
    ]
    chosen, alts, total = step06.canonical_vi(candidates, {"mèo": 9.0, "con mèo": 0.0})
    assert chosen["word"] == "con mèo"
    assert alts == ["mèo"]
    assert total == 4
    assert step06.summed_vi_senses(FIXTURE["vi_homographs"]) == 4


def test_loan_source_language_chinese_exception() -> None:
    templates = {"bor", "bor+", "lbor", "der", "der+", "cal", "calque", "obor"}
    chinese = {"zh", "cmn", "ltc", "och", "yue"}
    assert step06.loan_source_languages({"etymology_templates": [[FIXTURE["loan_french"]]]}, templates, chinese) == ["fr"]
    assert step06.loan_source_languages({"etymology_templates": [[FIXTURE["loan_chinese"]]]}, templates, chinese) == []


def test_dedup_keeps_attestation_ranked_concept() -> None:
    rows = FIXTURE["dedup_rows"]
    kept, actions = step06._deduplicate(rows, "vi_canonical", {})
    assert [row["concept_id"] for row in kept] == ["b"]
    assert actions == [("vi_canonical", "b", "a")]


def test_split_disjointness_assertion_rejects_shared_form() -> None:
    first = {"concept_id": "a", "en_lemma": "cat", "vi_cands": [{"word": "mèo"}]}
    second = {"concept_id": "b", "en_lemma": "feline", "vi_cands": [{"word": "mèo"}]}
    with pytest.raises(AssertionError, match="appears in splits"):
        step06.assert_split_disjointness({"fewshot_reservoir": [first], "directions": [], "test": [second]})
