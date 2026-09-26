"""Fixture-only tests for matched direction prompts."""

from importlib import import_module


directions = import_module("data.build.13_direction_data")


def test_direction_prompt_templates_and_seeded_split() -> None:
    examples = [{"w": f"mot{i}", "ru": f"один{i}"} for i in range(1, 6)]
    assert directions.render_direction_prompt(
        "repetition", examples=examples, test_form="sau", label="English",
        ru_label="Русский", ru_test="шесть", k=5,
    ).splitlines()[-1] == "English: sau - English:"
    assert directions.render_direction_prompt(
        "translation", examples=examples, test_form="six", label="English",
        ru_label="Русский", ru_test="шесть", k=5,
    ).splitlines()[-1] == "Русский: шесть - English:"
    ids = [f"id-{index:03d}" for index in range(132)]
    split = directions.assign_direction_splits(ids, fit_fraction=0.8, seed=20260925)
    assert split == directions.assign_direction_splits(ids, fit_fraction=0.8, seed=20260925)
    assert sum(value == "fit" for value in split.values()) == 106
    assert sum(value == "validate" for value in split.values()) == 26
