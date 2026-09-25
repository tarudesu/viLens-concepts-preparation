"""Tests for pure schema-diagnostic matching helpers."""

from importlib import import_module
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "build"))
inspect_step = import_module("02_inspect")


ALIASES = {
    "vi": ["vietnamese", "vi", "vie", "vie_latn"],
    "zh": ["chinese", "mandarin", "zh", "zho", "cmn", "cmn_hans"],
    "fr": ["french", "fr", "fra", "fre", "fra_latn"],
    "id": ["indonesian", "id", "ind", "ind_latn"],
}


def test_contains_cjk_uses_all_configured_ranges() -> None:
    ranges = [[13312, 19903], [19968, 40959], [131072, 173791]]
    assert inspect_step.contains_cjk("Hán 字", ranges)
    assert inspect_step.contains_cjk({"args": ["𠀀"]}, ranges)
    assert not inspect_step.contains_cjk("Vietnam", ranges)


def test_translation_matching_accepts_language_names_and_codes() -> None:
    assert inspect_step.translation_matches({"lang": "Vietnamese", "code": "vi"}, "vi", ALIASES)
    assert inspect_step.translation_matches({"lang": "Mandarin Chinese", "lang_code": "cmn"}, "zh", ALIASES)
    assert inspect_step.translation_matches({"lang": "French", "lang_code": "fra"}, "fr", ALIASES)
    assert not inspect_step.translation_matches({"lang": "German", "code": "de"}, "vi", ALIASES)


def test_translation_buckets_require_each_language_independently() -> None:
    items = [
        {"lang": "Vietnamese"},
        {"lang": "Chinese", "code": "cmn"},
        {"lang": "French"},
        {"lang": "Indonesian"},
    ]
    assert inspect_step.translation_buckets(items, ALIASES) == {"vi": True, "zh": True, "fr": True, "id": True}
    assert inspect_step.translation_buckets(items[:2], ALIASES) == {"vi": True, "zh": True, "fr": False, "id": False}


def test_sense_related_fields_keeps_only_sense_annotations() -> None:
    item = {"word": "a", "sense": "meaning", "senseid": "7", "sense_index": 2, "tags": ["formal"]}
    assert inspect_step.sense_related_fields(item) == {"sense": "meaning", "sense_index": 2, "senseid": "7"}
