"""Fixture tests for leakage-safe step-05 splitting helpers."""

import json
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


splitter = import_module("data.build.05_split")
FIXTURES = Path(__file__).parent / "fixtures"


def fixture_rows():
    return json.loads((FIXTURES / "split_cases.json").read_text(encoding="utf-8"))


def test_connected_components_link_shared_vi_and_english_terms():
    rows = fixture_rows()
    groups = splitter.connected_components(rows)
    grouped_ids = [[rows[index]["concept_id"] for index in group] for group in groups]
    assert grouped_ids == [["c001", "c002", "c003"], ["c004"], ["c005"]]


def test_component_stratum_uses_first_concept_and_reports_mixing():
    components = splitter.describe_components(fixture_rows(), [1, 2])
    assert components[0].stratum == ("noun", "2")
    assert components[0].mixed_syllable is True
    assert components[0].mixed_pos is False


def test_syllable_bucket_whitespace_and_boundaries():
    assert splitter.syllable_bucket(" mèo ", [1, 2]) == "1"
    assert splitter.syllable_bucket("con mèo", [1, 2]) == "2"
    assert splitter.syllable_bucket("mèo con nhỏ", [1, 2]) == "3+"


def test_component_sampling_is_seeded_and_never_splits_a_component():
    components = splitter.describe_components(fixture_rows(), [1, 2])
    pool_counts = Counter(component.stratum for component in components for _ in component.members)
    first = splitter.stratified_component_selection(components, 2, pool_counts, np.random.default_rng(17))
    second = splitter.stratified_component_selection(components, 2, pool_counts, np.random.default_rng(17))
    assert [item.component_id for item in first] == [item.component_id for item in second]
    assert all(len(item.members) in {1, 3} for item in first)


def test_fewshot_eligibility_requires_every_concept_to_pass():
    component = splitter.describe_components(fixture_rows(), [1, 2])[0]
    scores = {row["concept_id"]: 5.5 for row in component.members}
    assert splitter.fewshot_component_eligible(component, scores, 5.0)
    scores["c002"] = 4.9
    assert not splitter.fewshot_component_eligible(component, scores, 5.0)


def test_disjointness_assertion_rejects_repeated_vi_form_across_splits():
    rows = fixture_rows()[:2]
    rows[0]["split"] = "directions"
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="Leakage across splits for vi term"):
        splitter.assert_disjoint_splits(rows)


def test_disjointness_assertion_rejects_repeated_english_lemma_across_splits():
    rows = [fixture_rows()[0], fixture_rows()[2]]
    rows[0]["split"] = "directions"
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="Leakage across splits for en term"):
        splitter.assert_disjoint_splits(rows)


def test_disjoint_partitions_pass_assertions():
    rows = fixture_rows()[3:]
    rows[0]["split"] = "directions"
    rows[1]["split"] = "test"
    splitter.assert_disjoint_splits(rows)
