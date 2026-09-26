"""Pure integrity and provenance helpers used by the downloader."""

from pathlib import Path
import hashlib
from importlib import import_module
import logging
import sys

import pytest
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "build"))
download = import_module("01_download")
SourceRecord = download.SourceRecord
FrozenSourceError = download.FrozenSourceError
parse_sources = download.parse_sources
render_sources = download.render_sources
skip_if_hash_matches = download.skip_if_hash_matches
wordnet_version_string = download.wordnet_version_string
run_brysbaert = download.run_brysbaert


def test_sources_table_round_trips_and_sorts(tmp_path: Path) -> None:
    records = [
        SourceRecord("muse", "data/raw/muse/vi-en.txt", "https://example.org/muse", "2026-09-25T00:00:00Z", "3", "a" * 64, "v1|training", "CC BY-NC"),
        SourceRecord("cedict", "data/raw/cedict/cedict.txt.gz", "https://example.org/cedict", "2026-09-25T00:00:01Z", "4", "b" * 64, "1.0", "CC BY-SA"),
    ]
    rendered = render_sources(records)
    assert parse_sources(rendered) == sorted(records, key=lambda record: (record.source, record.file))
    assert render_sources(parse_sources(rendered)) == rendered
    output = tmp_path / "SOURCES.md"
    download.write_sources(output, records)
    assert parse_sources(output.read_text(encoding="utf-8")) == sorted(records, key=lambda record: (record.source, record.file))


def test_sources_writer_preserves_trailing_provenance_notes(tmp_path: Path) -> None:
    record = SourceRecord("fixture", "data/raw/x", "https://example.org/x", "2026-09-25T00:00:00Z", "1", "a" * 64, "v1", "test")
    output = tmp_path / "SOURCES.md"
    output.write_text(
        render_sources([record]) + "\nQuery date: 2026-09-25 UTC.\n\nA second note.\n",
        encoding="utf-8",
    )

    download.write_sources(output, [record])

    rewritten = output.read_text(encoding="utf-8")
    assert rewritten.endswith("\n\nQuery date: 2026-09-25 UTC.\n\nA second note.\n")
    assert parse_sources(rewritten) == [record]


def test_skip_if_hash_matches_checks_a_tiny_file(tmp_path: Path) -> None:
    path = tmp_path / "tiny.txt"
    payload = b"tiny source\n"
    path.write_bytes(payload)
    record = SourceRecord("fixture", str(path), "https://example.org/tiny", "2026-09-25T00:00:00Z", str(len(payload)), hashlib.sha256(payload).hexdigest(), "fixture v1", "fixture license")
    assert skip_if_hash_matches(path, record, 4)
    assert not skip_if_hash_matches(tmp_path / "not-yet-downloaded.txt", record, 4)
    with pytest.raises(FrozenSourceError, match="without a SOURCES.md hash"):
        skip_if_hash_matches(path, None, 4)
    path.write_bytes(b"changed\n")
    with pytest.raises(FrozenSourceError, match="hash mismatch"):
        skip_if_hash_matches(path, record, 4)


def test_wordnet_version_string_records_both_versions() -> None:
    assert wordnet_version_string("3.10.3", "3.0") == "NLTK 3.10.3; WordNet 3.0"
    with pytest.raises(ValueError, match="versions must be non-empty"):
        wordnet_version_string("3.10.3", " ")


def test_brysbaert_registers_an_unmodified_manual_workbook(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    raw = repo / "data" / "raw"
    workbook = raw / "brysbaert" / "13428_2013_403_MOESM1_ESM.xlsx"
    workbook.parent.mkdir(parents=True)
    payload = b"fixture xlsx bytes"
    workbook.write_bytes(payload)
    sources = repo / "data" / "build" / "SOURCES.md"
    config = {
        "downloads": {
            "brysbaert": {"manual_url": "https://doi.org/10.3758/s13428-013-0403-5"},
            "chunk_size_bytes": 4,
        }
    }

    run_brysbaert(config, None, raw, {}, sources, logging.getLogger("test_brysbaert"))

    records = parse_sources(sources.read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert record.file == "data/raw/brysbaert/13428_2013_403_MOESM1_ESM.xlsx"
    assert record.bytes == str(len(payload))
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.license == "as distributed with the article; check terms before redistributing values"
    assert workbook.read_bytes() == payload


def _flores_fixture(ids: list[str], *, language: str = "en", texts: list[str] | None = None) -> pa.Table:
    sentence_texts = texts or [f"sentence {item}" for item in ids]
    return pa.table({
        "id": ids,
        "text": sentence_texts,
        "iso_639_3": [language] * len(ids),
        "iso_15924": ["Latn"] * len(ids),
        "glottocode": ["fixture123"] * len(ids),
        "variant": ["standard"] * len(ids),
        "split": ["dev"] * len(ids),
        "unretained_field": ["not saved"] * len(ids),
    })


def test_flores_schema_mapping_and_selected_columns() -> None:
    source = _flores_fixture(["2", "1"], texts=["ca\u0301", "hello"])
    mapped, variants = download.map_flores_source_table(source, split="dev", language="en")
    assert mapped.column_names == ["id", "sentence", "iso_639_3", "iso_15924", "glottocode", "variant", "split"]
    assert mapped.column("id").to_pylist() == ["1", "2"]
    assert mapped.column("sentence").to_pylist() == ["hello", "cá"]
    assert variants == {"standard": 2}


def test_flores_rejects_duplicate_ids_and_empty_sentences() -> None:
    duplicate = _flores_fixture(["1", "1"])
    with pytest.raises(download.SourceError, match="exactly one row per id"):
        download.map_flores_source_table(duplicate, split="dev", language="en")
    empty = _flores_fixture(["1"], texts=["   "])
    with pytest.raises(download.SourceError, match="empty sentences"):
        download.map_flores_source_table(empty, split="dev", language="en")


def test_flores_parallel_ids_must_match_within_split() -> None:
    first, _ = download.map_flores_source_table(_flores_fixture(["1", "2"], language="en"), split="dev", language="en")
    second, _ = download.map_flores_source_table(_flores_fixture(["1", "3"], language="vi"), split="dev", language="vi")
    with pytest.raises(download.SourceError, match="sentence ID sets differ"):
        download.validate_flores_alignment(
            {("dev", "en"): first, ("dev", "vi"): second},
            splits=["dev"], languages=["en", "vi"],
        )
