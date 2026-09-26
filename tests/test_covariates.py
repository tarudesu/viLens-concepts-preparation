"""Fixture-only tests for step-09 covariate helpers."""

import pytest
from importlib import import_module

covariates = import_module("data.build.09_covariates")
n_characters = covariates.n_characters
n_syllables = covariates.n_syllables
standardized_mean_difference = covariates.standardized_mean_difference


@pytest.mark.parametrize(("text", "expected"), [("", 0), ("máy tính", 2), ("  cà phê  ", 2), ("đường\tphố", 2)])
def test_n_syllables(text: str, expected: int) -> None:
    assert n_syllables(text) == expected


def test_n_characters_counts_nfc_code_points_and_spaces() -> None:
    assert n_characters("cafe\u0301") == 4
    assert n_characters("cà phê") == 6


def test_standardized_mean_difference() -> None:
    assert standardized_mean_difference([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(-3.0)
    assert standardized_mean_difference([], [1.0]) is None
    assert standardized_mean_difference([1.0], [1.0]) is None
