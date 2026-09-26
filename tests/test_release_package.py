"""Release-package privacy, provenance, documentation, and restoration checks."""

from __future__ import annotations

from collections import deque
import csv
import hashlib
from pathlib import Path
import re
import subprocess
import sys

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
RAW = ROOT / "data" / "raw"


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a pipeline TSV after its commit-comment line."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        handle.readline()
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def release_files() -> list[Path]:
    """Return all regular files in the built release tree."""
    return sorted(path for path in RELEASE.rglob("*") if path.is_file())


class _TrieNode:
    """Small Aho-Corasick node for exact multi-pattern substring scanning."""

    __slots__ = ("edges", "failure", "terminal")

    def __init__(self) -> None:
        self.edges: dict[str, _TrieNode] = {}
        self.failure: _TrieNode | None = None
        self.terminal = False


def contains_any(text: str, patterns: set[str]) -> bool:
    """Check all exact patterns in one scan of text."""
    root = _TrieNode()
    for pattern in patterns:
        if not pattern:
            continue
        node = root
        for char in pattern:
            child = node.edges.get(char)
            if child is None:
                child = _TrieNode()
                node.edges[char] = child
            node = child
        node.terminal = True

    root.failure = root
    queue: deque[_TrieNode] = deque()
    for child in root.edges.values():
        child.failure = root
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for char, child in parent.edges.items():
            fallback = parent.failure
            while fallback is not root and char not in fallback.edges:
                fallback = fallback.failure
            child.failure = fallback.edges.get(char, root)
            child.terminal = child.terminal or bool(child.failure.terminal)
            queue.append(child)

    node = root
    for char in text:
        while node is not root and char not in node.edges:
            node = node.failure
        node = node.edges.get(char, root)
        if node.terminal:
            return True
    return False


def _flores_sentences() -> set[str]:
    """Read all local FLORES dev/devtest sentences for leak checks."""
    result: set[str] = set()
    paths = sorted((RAW / "flores").glob("*.parquet"))
    if len(paths) != 10:
        raise AssertionError("Expected the ten registered FLORES files for release privacy tests")
    for path in paths:
        table = pq.read_table(path, columns=["sentence"])
        result.update(sentence for sentence in table.column("sentence").to_pylist() if sentence)
    return result


def _sha256(path: Path) -> str:
    """Hash a file without retaining its full contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_restore(script: str, args: list[str]) -> None:
    """Run a bundled restoration script in the current uv Python environment."""
    result = subprocess.run(
        [sys.executable, str(RELEASE / "scripts" / script), *args],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"Bundled {script} failed with exit code {result.returncode}")


def test_released_concepts_preserve_rows_and_drop_configured_columns() -> None:
    """The release table has the same concept IDs and omits restricted values."""
    config = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    release_config = config["release"]
    released_columns, released_rows = read_tsv(RELEASE / "concepts.tsv")
    source_columns, source_rows = read_tsv(ROOT / "data" / "concepts.tsv")
    dropped = set(release_config["drop_columns"])
    if release_config["include_concreteness_values"] is not True:
        dropped.add("concreteness")
    if dropped.intersection(released_columns):
        raise AssertionError("A configured restricted column remains in the release table")
    if set(released_columns) != set(source_columns) - dropped:
        raise AssertionError("Released columns differ from the configured source-column projection")
    if len(released_rows) != len(source_rows):
        raise AssertionError("Release row count differs from data/concepts.tsv")
    if {row["concept_id"] for row in released_rows} != {row["concept_id"] for row in source_rows}:
        raise AssertionError("Release concept_id set differs from data/concepts.tsv")


def test_no_flores_sentence_occurs_in_any_release_file() -> None:
    """No complete FLORES dev/devtest sentence is present in released files."""
    sentences = _flores_sentences()
    combined_release_text = "\0".join(path.read_text(encoding="utf-8") for path in release_files())
    if contains_any(combined_release_text, sentences):
        raise AssertionError("FLORES sentence text found in a release file")


def test_no_token_or_huggingface_credential_pattern_occurs_in_release() -> None:
    """The upload token and token-shaped strings are absent without exposing them."""
    token_path = ROOT / "keys" / "hf-w.txt"
    token = token_path.read_bytes().strip()
    if not token:
        raise AssertionError("The configured Hugging Face token file is empty")
    credential_pattern = re.compile(rb"hf_[A-Za-z0-9]{10,}")
    for path in release_files():
        content = path.read_bytes()
        if token in content or credential_pattern.search(content):
            raise AssertionError(f"Credential-like content found in release file {path.relative_to(RELEASE)}")


def test_no_user_identity_is_included_in_release() -> None:
    """Keep the anonymous-review package free of the repository author's identity."""
    forbidden = (b"tarudesu", b"luannt@uit.edu.vn")
    for path in release_files():
        content = path.read_bytes().lower()
        if any(value in content for value in forbidden):
            raise AssertionError(f"User identity found in release file {path.relative_to(RELEASE)}")


def test_muse_dictionary_files_are_not_in_release() -> None:
    """The raw MUSE pair dictionaries and the MUSE flag are not redistributed."""
    columns, _ = read_tsv(RELEASE / "concepts.tsv")
    if "in_muse" in columns:
        raise AssertionError("The MUSE attestation flag must not be released")
    released = release_files()
    for source in (RAW / "muse" / "vi-en.txt", RAW / "muse" / "en-vi.txt"):
        source_hash = _sha256(source)
        if any(path.name == source.name or _sha256(path) == source_hash for path in released):
            raise AssertionError("A raw MUSE dictionary file is present in the release")


def test_every_released_column_is_documented_in_dataset_card() -> None:
    """The card field table documents every released concepts.tsv column."""
    columns, _ = read_tsv(RELEASE / "concepts.tsv")
    card = (RELEASE / "README.md").read_text(encoding="utf-8")
    field_table = card.split("## Fields", 1)[1].split("## Splits", 1)[0]
    tick = chr(96)
    documented = {
        line.split(tick, 2)[1]
        for line in field_table.splitlines()
        if line.startswith("| " + tick) and tick in line[3:]
    }
    if set(columns) != documented:
        raise AssertionError(
            f"Dataset-card field documentation mismatch: missing={sorted(set(columns) - documented)}, "
            f"extra={sorted(documented - set(columns))}"
        )


def test_restoration_scripts_reproduce_omitted_data_byte_for_byte(tmp_path: Path) -> None:
    """The bundled scripts reconstruct the frozen source columns/output exactly."""
    _, source_rows = read_tsv(ROOT / "data" / "concepts.tsv")
    muse_output = tmp_path / "with_muse.tsv"
    _run_restore("add_muse_flag.py", [
        "--concepts", str(RELEASE / "concepts.tsv"),
        "--vi-en", str(RAW / "muse" / "vi-en.txt"),
        "--en-vi", str(RAW / "muse" / "en-vi.txt"),
        "--sources", str(ROOT / "data" / "build" / "SOURCES.md"),
        "--output", str(muse_output),
    ])
    muse_columns, muse_rows = read_tsv(muse_output)
    if "in_muse" not in muse_columns:
        raise AssertionError("MUSE restoration did not produce the omitted column")
    expected_muse = [(row["concept_id"], row["in_muse"]) for row in source_rows]
    actual_muse = [(row["concept_id"], row["in_muse"]) for row in muse_rows]
    if expected_muse != actual_muse:
        raise AssertionError("Restored in_muse values differ from the frozen concepts.tsv column")

    concreteness_output = tmp_path / "with_concreteness.tsv"
    _run_restore("add_concreteness.py", [
        "--concepts", str(RELEASE / "concepts.tsv"),
        "--xlsx", str(RAW / "brysbaert" / "13428_2013_403_MOESM1_ESM.xlsx"),
        "--sources", str(ROOT / "data" / "build" / "SOURCES.md"),
        "--output", str(concreteness_output),
    ])
    concreteness_columns, concreteness_rows = read_tsv(concreteness_output)
    if "concreteness" not in concreteness_columns:
        raise AssertionError("Concreteness restoration did not produce the omitted column")
    expected_by_id = [
        (row["concept_id"], row["concreteness"], row["concreteness_match"])
        for row in source_rows
    ]
    actual_by_id = [
        (row["concept_id"], row["concreteness"], row["concreteness_match"])
        for row in concreteness_rows
    ]
    if expected_by_id != actual_by_id:
        raise AssertionError("Restored concreteness fields differ from the frozen concepts.tsv values")

    flores_output = tmp_path / "directions_flores.jsonl"
    _run_restore("build_flores_directions.py", [
        "--flores-dir", str(RAW / "flores"),
        "--sources", str(ROOT / "data" / "build" / "SOURCES.md"),
        "--output", str(flores_output),
    ])
    original_flores = ROOT / "data" / "prompts" / "directions_flores.jsonl"
    if flores_output.read_bytes() != original_flores.read_bytes():
        raise AssertionError("FLORES restoration output differs byte-for-byte from the frozen JSONL")
