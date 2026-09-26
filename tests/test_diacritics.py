"""Fixture-only tests for the step-11 reference lexicon and collapse rule."""

from importlib import import_module

import pytest

diacritics = import_module("data.build.11_diacritics")


def test_reference_index_merges_spelling_equivalent_forms() -> None:
    index: dict[str, set[str]] = {}
    diacritics.add_reference_form(index, "hoà")
    diacritics.add_reference_form(index, "hòa")
    assert index["hoa"] == {"hòa"}


def test_collapse_partners_only_returns_other_orthographic_keys() -> None:
    index: dict[str, set[str]] = {}
    for form in ("má", "ma", "mà", "mạ"):
        diacritics.add_reference_form(index, form)
    assert diacritics.find_collapse_partners("má", index, 3) == ["ma", "mà", "mạ"]
    assert diacritics.find_collapse_partners("má", index, 2) == ["ma", "mà"]
    assert diacritics.find_collapse_partners("cà phê", index, 3) == []


def test_add_reference_form_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        diacritics.add_reference_form({}, "  ")
    with pytest.raises(TypeError):
        diacritics.add_reference_form({}, None)  # type: ignore[arg-type]


def test_standalone_vietnamese_diacritic_mark_is_preserved_under_empty_key() -> None:
    index: dict[str, set[str]] = {}
    diacritics.add_reference_form(index, "𖿰")
    assert index[""] == {"𖿰"}
