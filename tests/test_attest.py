"""Fixture-only tests for step 04 matching and WDQS parsing."""

import json
import shutil
from pathlib import Path

from data.build import common
from importlib import import_module


attest = import_module("data.build.04_attest")
FIXTURES = Path(__file__).parent / "fixtures"


def test_muse_matching_is_trimmed_lowercase_and_nfc():
    path = FIXTURES / "muse_attest.txt"
    pair = attest.parse_muse_line(path.read_text(encoding="utf-8").splitlines()[0], "en-vi")
    assert pair == ("ô tô", "car")
    assert attest.muse_contains({pair}, "o\u0302 to\u0302 ", " CAR ")


def test_parses_saved_wikidata_response_fixture():
    payload = json.loads((FIXTURES / "wikidata_response.json").read_text(encoding="utf-8"))
    assert attest.parse_wikidata_response(payload) == {
        ("car", "ô tô"),
        ("café", "quán cà phê"),
    }


def test_action_search_keeps_exact_nfc_casefolded_label_or_alias():
    payload = json.loads((FIXTURES / "wikidata_action_search.json").read_text(encoding="utf-8"))
    assert attest.exact_english_qids(payload, "café") == ["Q42"]


def test_action_entities_parse_vietnamese_label_and_aliases():
    payload = json.loads((FIXTURES / "wikidata_action_entities.json").read_text(encoding="utf-8"))
    assert attest.vietnamese_entity_terms(payload, ["Q42"]) == {"Q42": {"ô tô", "xe hơi"}}


def test_action_api_transient_errors_are_retryable():
    assert attest._retryable_api_error({"code": "cirrussearch-too-busy-error"})
    assert attest._retryable_api_error({"code": "maxlag"})
    assert not attest._retryable_api_error({"code": "unknown-error"})


def test_syllable_buckets_use_whitespace_count():
    assert attest.syllable_bucket("xe", [1, 2]) == "1"
    assert attest.syllable_bucket("ô tô", [1, 2]) == "2"
    assert attest.syllable_bucket("học sinh giỏi", [1, 2]) == "3+"


def test_vi_gloss_matching_uses_whole_words():
    data = json.loads((FIXTURES / "attest_glosses.json").read_text(encoding="utf-8"))
    for gloss, expected in data["glosses"].items():
        assert attest.contains_whole_word(data["needle"], gloss) is expected


def test_query_uses_english_language_tagged_values():
    query = attest.build_sparql_query([("café", "quán cà phê"), ("car", "ô tô")])
    assert '"café"@en' in query
    assert '"quán cà phê"@vi' in query
    assert '"car"@en' in query
    assert "skos:altLabel" in query


def test_dropflow_unit_required_and_serialized(tmp_path):
    target = tmp_path / "dropflow.jsonl"
    common.log_dropflow(target, step="04_attest", stage="agreement", unit="concepts", n_in=2, n_out=1, n_out_by_pos={"noun": 1})
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["unit"] == "concepts"


def test_backfills_step03_dropflow_units_once(tmp_path):
    path = tmp_path / "dropflow.jsonl"
    shutil.copyfile(FIXTURES / "dropflow_step03_legacy.jsonl", path)
    assert attest.backfill_dropflow_units(path) == 3
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["unit"] for record in records] == ["entries", "groups", "concepts"]
    first_bytes = path.read_bytes()
    assert attest.backfill_dropflow_units(path) == 0
    assert path.read_bytes() == first_bytes
