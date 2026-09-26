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


@pytest.mark.parametrize(("word", "dropped"), [("ô-đờ-cô-lôn", True), ("đường sắt", False), ("máy tính", False)])
def test_hyphen_filter_uses_canonical_form(word: str, dropped: bool) -> None:
    assert step06.is_hyphenated_transliteration(word) is dropped


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


def test_vi_orthographic_candidates_merge_and_promote_frequency_display() -> None:
    base = {
        "n_sources": 2, "external_attested": False, "in_wiktextract": True,
        "in_vi_gloss": True, "in_muse": False, "in_wikidata": False,
        "n_vi_entries": 1, "n_senses_vi": [2], "vi_pos": ["noun"],
        "categories": [[]], "etymology_templates": [[]], "etymology_text": [None],
        "cjk_forms": [[]], "source_location": "top",
    }
    candidates = [{**base, "word": "hoà"}, {**base, "word": "hòa"}]
    merged, variants, count, examples = step06.merge_vi_candidates(
        candidates, {"hoà": 5.0, "hòa": 4.0}, ["wiktextract_table", "vi_gloss", "muse", "wikidata"],
    )
    assert count == 1
    assert merged[0]["word"] == "hoà"  # Higher Zipf frequency wins.
    assert variants[0]["variants"] == ["hoà", "hòa"]
    assert examples[0]["orth_key"] == "hòa"

    merged, _, _, _ = step06.merge_vi_candidates(
        candidates, {"hoà": 5.0, "hòa": 5.0}, ["wiktextract_table", "vi_gloss", "muse", "wikidata"],
    )
    assert merged[0]["word"] == "hòa"  # Equal frequency prefers modern placement.


def test_vi_form_dedup_uses_orthographic_key() -> None:
    rows = [
        {"concept_id": "old", "en_lemma": "old", "vi_canonical": "hoà", "n_sources": 2, "external_attested": False, "n_senses_vi": 1},
        {"concept_id": "modern", "en_lemma": "modern", "vi_canonical": "hòa", "n_sources": 3, "external_attested": True, "n_senses_vi": 2},
    ]
    kept, actions = step06._deduplicate(rows, "vi_canonical", {})
    assert [row["concept_id"] for row in kept] == ["modern"]
    assert actions == [("vi_canonical", "modern", "old")]


def test_loanword_source_allowlist_and_template_selection() -> None:
    templates = {"bor", "bor+", "lbor", "der", "der+", "obor"}
    drop_codes = {"fr", "en", "la", "la-new", "la-lat", "la-med", "grc", "pt", "es", "it", "de", "nl", "ru", "ms", "id"}
    prefixes = ["en-", "fr-", "es-", "pt-"]
    for name in ("loan_ja", "loan_lzh_lit", "loan_cmc_pro", "loan_chinese"):
        records = step06.loan_templates_found({"etymology_templates": [[FIXTURE[name]]]}, templates)
        assert records
        assert not any(step06.source_matches_drop_list(str(row["source_lang"]), drop_codes, prefixes) for row in records)
    for name in ("loan_french", "loan_ms", "loan_en_us"):
        records = step06.loan_templates_found({"etymology_templates": [[FIXTURE[name]]]}, templates)
        assert records
        assert any(step06.source_matches_drop_list(str(row["source_lang"]), drop_codes, prefixes) for row in records)
    assert step06.loan_templates_found({"etymology_templates": [[FIXTURE["loan_french"]]]}, templates) == [
        {"template": "bor", "source_lang": "fr", "args3": "gare"},
    ]
    # cal/calque are deliberately absent from the new template allowlist.
    assert step06.loan_templates_found({"etymology_templates": [[FIXTURE["loan_cal_pt"]]]}, templates) == []


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
