from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from data.build.common import (
    DataConfigError,
    DropflowLogger,
    _reject_json_constant,
    _unique_mapping,
    iter_jsonl,
    load_config,
    log_dropflow,
    normalize_nfc,
    normalize_strings,
    norm_ru,
    normalized_levenshtein,
    require_resolved_concreteness,
    ru_stress_free,
    setup_logging,
    strip_diacritics,
    stream_jsonl,
    vi_orth_key,
)


def test_normalize_nfc() -> None:
    assert normalize_nfc("e\u0301") == "é"


def test_russian_stress_removal_and_matching() -> None:
    assert ru_stress_free("кни́га") == "книга"
    assert norm_ru("ёж") == norm_ru("еж")
    assert ru_stress_free("ё́ж") == "ёж"
    assert norm_ru("ё́ж") == norm_ru("еж")


def test_strip_diacritics_including_vietnamese_d() -> None:
    assert strip_diacritics("Đường phố") == "duong pho"
    assert strip_diacritics("Tiếng Việt", preserve_case=True) == "Tieng Viet"
    assert strip_diacritics("ĐỨNG", preserve_case=True) == "DUNG"


def test_release_guard_rejects_pending_concreteness() -> None:
    with pytest.raises(ValueError, match="Refusing final dataset assembly"):
        require_resolved_concreteness([{"concept_id": "abc", "concreteness_match": "pending"}])
    require_resolved_concreteness([{"concept_id": "abc", "concreteness_match": "exact"}])


@pytest.mark.parametrize(
    ("left", "right"),
    [("hoà", "hòa"), ("khoẻ", "khỏe"), ("thuý", "thúy"), ("kí", "ký"), ("mĩ thuật", "mỹ thuật"), ("may", "mai")],
)
def test_vietnamese_orthographic_equivalence(left: str, right: str) -> None:
    assert vi_orth_key(left) == vi_orth_key(right)


def test_vietnamese_orthographic_key_leaves_qu_and_standalone_y_alone() -> None:
    assert vi_orth_key("quý") == "quý"
    assert vi_orth_key("quí") == "quí"
    assert vi_orth_key("y tá") == "y tá"


@pytest.mark.parametrize(("left", "right", "expected"), [("", "", 0.0), ("a", "", 1.0), ("same", "same", 0.0), ("cat", "cut", 1 / 3)])
def test_normalized_levenshtein(left: str, right: str, expected: float) -> None:
    assert normalized_levenshtein(left, right) == expected


def test_normalize_nfc_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_nfc(1)  # type: ignore[arg-type]


def test_normalize_strings_recurses_through_json_values() -> None:
    value = {"e\u0301": ["e\u0301", {"nested": "e\u0301"}], "number": 1}
    assert normalize_strings(value) == {"é": ["é", {"nested": "é"}], "number": 1}


@pytest.mark.parametrize("value", ["", "Tiếng Việt", "中文", "français", "Indonesia", "e\u0301"])
def test_normalization_is_idempotent(value: str) -> None:
    assert normalize_nfc(normalize_nfc(value)) == normalize_nfc(value)


def test_normalize_strings_preserves_types_and_input() -> None:
    value = {1: ("e\u0301", None, True, 2.5)}
    assert normalize_strings(value) == {1: ("é", None, True, 2.5)}
    assert value == {1: ("e\u0301", None, True, 2.5)}


def test_normalize_strings_rejects_colliding_keys() -> None:
    with pytest.raises(ValueError, match="Duplicate mapping key"):
        normalize_strings({"é": 1, "e\u0301": 2})


def test_unique_mapping_normalizes_and_preserves_non_string_keys() -> None:
    assert _unique_mapping([("e\u0301", 1), (2, None)]) == {"é": 1, 2: None}


@pytest.mark.parametrize("keys", [("x", "x"), ("é", "e\u0301")])
def test_unique_mapping_rejects_duplicate_keys(keys: tuple[str, str]) -> None:
    with pytest.raises(ValueError, match="Duplicate mapping key"):
        _unique_mapping([(key, i) for i, key in enumerate(keys)])


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_reject_json_constant(constant: str) -> None:
    with pytest.raises(ValueError, match="Non-standard JSON constant"):
        _reject_json_constant(constant)


def test_load_config_normalizes_and_requires_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text("name: e\u0301\n", encoding="utf-8")
    assert load_config(config_path) == {"name": "é"}

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(DataConfigError, match="top-level mapping"):
        load_config(invalid_path)


@pytest.mark.parametrize(
    "content",
    ["x: 1\nx: 2\n", "é: 1\ne\u0301: 2\n", "x: {y: 1, y: 2}\n", "x: [\n", "!!python/object:builtins.object {}"],
)
def test_load_config_rejects_invalid_yaml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DataConfigError, match="Invalid YAML.*bad.yaml"):
        load_config(path)


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataConfigError, match="Could not read config"):
        load_config(tmp_path / "missing.yaml")


def test_iter_jsonl_streams_and_normalizes(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text('{"text": "e\\u0301"}\n{"n": 2}\n', encoding="utf-8")
    assert list(iter_jsonl(jsonl_path)) == [{"text": "é"}, {"n": 2}]
    assert stream_jsonl is iter_jsonl


def test_iter_jsonl_does_not_read_ahead_or_read_whole_file() -> None:
    class LinesOnly:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield b'{"first": true}\n'
            raise AssertionError("JSONL reader consumed records ahead of the caller")

        def read(self, *args):
            raise AssertionError("JSONL reader attempted a bulk read")

        readlines = read

    with patch.object(Path, "open", return_value=LinesOnly()) as open_file:
        records = iter_jsonl("unused.jsonl")
        open_file.assert_not_called()
        assert next(records) == {"first": True}
        records.close()


@pytest.mark.parametrize(
    "bad_line",
    [b"\n", b"not-json\n", b'{"x": 1, "x": 2}\n', b'{"x": NaN}\n', b'{"x": "\xff"}\n', '{"é": 1, "e\\u0301": 2}\n'.encode()],
)
def test_iter_jsonl_diagnoses_bad_records(tmp_path: Path, bad_line: bytes) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"ok": true}\n' + bad_line)
    records = iter_jsonl(path)
    assert next(records) == {"ok": True}
    with pytest.raises(ValueError, match="records.jsonl at line 2"):
        next(records)


def test_iter_jsonl_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Could not read JSONL file"):
        next(iter_jsonl(tmp_path / "missing.jsonl"))


def test_iter_jsonl_reports_invalid_lines(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    iterator = iter_jsonl(jsonl_path)
    assert next(iterator) == {"ok": True}
    with pytest.raises(ValueError, match="line 2"):
        next(iterator)


def test_dropflow_logger_writes_deterministic_json(tmp_path: Path) -> None:
    path = tmp_path / "dropflow.jsonl"
    DropflowLogger(path).record(
        step="01",
        stage="sources",
        unit="items",
        n_in=3,
        n_out=2,
        n_out_by_pos={"verb": 1, "noun": 1},
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "n_in": 3,
        "n_out": 2,
        "n_out_by_pos": {"noun": 1, "verb": 1},
        "stage": "sources",
        "step": "01",
        "unit": "items",
    }


def test_dropflow_appends_byte_identical_records(tmp_path: Path) -> None:
    path = tmp_path / "interim" / "dropflow.jsonl"
    log_dropflow(path, step="02", stage="e\u0301", unit="groups", n_in=3, n_out=2, n_out_by_pos={"verb": 1, "e\u0301": 1})
    first = path.read_bytes()
    log_dropflow(path, step="02", stage="é", unit="groups", n_in=3, n_out=2, n_out_by_pos={"é": 1, "verb": 1})
    assert path.read_bytes() == first + first
    assert json.loads(first)["stage"] == "é"


@pytest.mark.parametrize(
    "n_in,n_out,by_pos",
    [(1, 2, {"noun": 2}), (-1, 0, {}), (2, 1, {}), (2, 1, {"noun": -1}), (True, 1, {"noun": 1}), (2, 1, {"noun": 1.0}), (2, 1, {1: 1}), (2, 2, {"é": 1, "e\u0301": 1})],
)
def test_dropflow_rejects_inconsistent_counts(tmp_path: Path, n_in, n_out, by_pos) -> None:
    path = tmp_path / "dropflow.jsonl"
    with pytest.raises(ValueError):
        log_dropflow(path, step="02", stage="bad", unit="concepts", n_in=n_in, n_out=n_out, n_out_by_pos=by_pos)
    assert not path.exists()


def test_setup_logging_is_deterministic_and_replaces_handlers(tmp_path: Path, capsys) -> None:
    step = "00_test"
    logger = setup_logging(step, tmp_path, level="INFO")
    logger.info("Tiếng Việt")
    first_stdout = capsys.readouterr().out
    first_file = (tmp_path / f"{step}.log").read_bytes()
    try:
        logger = setup_logging(step, tmp_path, level="INFO")
        logger.info("Tiếng Việt")
        assert capsys.readouterr().out == first_stdout == "INFO Tiếng Việt\n"
        assert (tmp_path / f"{step}.log").read_bytes() == first_file
        assert first_file.decode("utf-8") == first_stdout
        assert len(logger.handlers) == 2

        logger = setup_logging(step, tmp_path / "new", level=logging.WARNING)
        logger.info("filtered")
        logger.warning("visible")
        assert capsys.readouterr().out == "WARNING visible\n"
        assert (tmp_path / "new" / f"{step}.log").read_text() == "WARNING visible\n"
        assert (tmp_path / f"{step}.log").read_bytes() == first_file
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
