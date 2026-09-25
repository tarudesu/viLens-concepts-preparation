"""Pure integrity and provenance helpers used by the downloader."""

from pathlib import Path
import hashlib
from importlib import import_module
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "build"))
download = import_module("01_download")
SourceRecord = download.SourceRecord
FrozenSourceError = download.FrozenSourceError
parse_sources = download.parse_sources
render_sources = download.render_sources
skip_if_hash_matches = download.skip_if_hash_matches


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
