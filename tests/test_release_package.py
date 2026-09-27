"""Integrity and documentation checks for the HF-main release candidate."""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "docs" / "data-archive" / "RELEASE_v1.3.sha256"
HF_BASE = "f0031da2dd5301738500d2d02963050f4fd355c2"
HF_V13_1_COMMIT = "8a2d99cadfc898222deed27a1f8f123692072d5c"
PREPARATION_TAG_URL = (
    "https://github.com/tarudesu/viLens-concepts-preparation/tree/prereg-v1.2"
)
HF_V12_TAG_URL = "https://huggingface.co/datasets/tarudesu/viLens-concepts/tree/v1.2"


def release_files() -> list[Path]:
    """Return all regular files in the candidate release tree."""
    return sorted(path for path in RELEASE.rglob("*") if path.is_file())


def sha256(path: Path) -> str:
    """Hash a file in chunks without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Read the ten registered local FLORES dev/devtest files for leak checks."""
    result: set[str] = set()
    paths = sorted((RAW / "flores").glob("*.parquet"))
    if len(paths) != 10:
        raise AssertionError("Expected the ten registered FLORES files for release privacy tests")
    for path in paths:
        table = pq.read_table(path, columns=["sentence"])
        result.update(sentence for sentence in table.column("sentence").to_pylist() if sentence)
    return result


def _expected_release_paths() -> set[str]:
    """List the data-only v1.3 release payload paths, excluding its README."""
    paths = {".gitattributes", "LICENSE", "concepts.jsonl"}
    paths.add("prompts/directions_matched.jsonl")
    paths.update({"prompts/cloze_diac_set1.jsonl", "prompts/cloze_nodiac_set1.jsonl"})
    for prompt_format in ("repetition", "translation"):
        for condition in ("diac", "nodiac"):
            for set_number in range(1, 7):
                paths.add(
                    f"prompts/{prompt_format}_{condition}_set{set_number}.jsonl"
                )
    return paths


def _read_front_matter() -> dict:
    """Parse the README YAML front matter."""
    readme = (RELEASE / "README.md").read_text(encoding="utf-8")
    parts = readme.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise AssertionError("README must start with YAML front matter")
    return yaml.safe_load(parts[1])


def test_release_manifest_hashes_every_release_file() -> None:
    """The committed manifest records the HF base and every non-README hash."""
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    expected_header = [
        f"# HF base commit: {HF_BASE}",
        f"# HF v1.3.1 commit: {HF_V13_1_COMMIT}",
    ]
    if lines[:2] != expected_header:
        raise AssertionError("Release manifest does not record the expected HF commits")
    entries: dict[str, str] = {}
    for line in lines[2:]:
        if not line:
            continue
        digest, path = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssertionError(f"Invalid SHA-256 digest in manifest: {path}")
        if path in entries:
            raise AssertionError(f"Duplicate release manifest entry: {path}")
        entries[path] = digest

    expected = {
        f"release/{path.relative_to(RELEASE).as_posix()}": sha256(path)
        for path in release_files()
        if path.name != "README.md"
    }
    if entries != expected:
        missing = sorted(set(expected) - set(entries))
        extra = sorted(set(entries) - set(expected))
        changed = sorted(
            path for path in set(entries) & set(expected) if entries[path] != expected[path]
        )
        raise AssertionError(
            f"Release manifest mismatch: missing={missing}, extra={extra}, changed={changed}"
        )


def test_release_file_inventory_is_the_data_only_v13_candidate() -> None:
    """The release tree has HF-main data, rebuilt cloze, and README only."""
    actual = {path.relative_to(RELEASE).as_posix() for path in release_files()}
    expected = _expected_release_paths() | {"README.md"}
    if actual != expected:
        raise AssertionError(
            f"Release file inventory mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def test_readme_front_matter_and_config_splits_match_release_files() -> None:
    """The card declares required metadata and exactly the five supported configs."""
    front = _read_front_matter()
    required = {"license", "language", "pretty_name", "size_categories", "tags", "configs"}
    if not required.issubset(front):
        raise AssertionError(f"Missing README front-matter keys: {sorted(required - set(front))}")
    configs = front["configs"]
    names = [config["config_name"] for config in configs]
    if names != ["concepts", "cloze", "repetition", "translation", "directions"]:
        raise AssertionError(f"Unexpected README configs: {names}")
    split_names = {
        config["config_name"]: {item["split"] for item in config["data_files"]}
        for config in configs
    }
    expected_splits = {
        "concepts": {"train"},
        "cloze": {"diac_set1", "nodiac_set1"},
        "repetition": {
            *(f"diac_set{index}" for index in range(1, 7)),
            *(f"nodiac_set{index}" for index in range(1, 7)),
        },
        "translation": {
            *(f"diac_set{index}" for index in range(1, 7)),
            *(f"nodiac_set{index}" for index in range(1, 7)),
        },
        "directions": {"matched"},
    }
    if split_names != expected_splits:
        raise AssertionError(f"Unexpected README split inventory: {split_names}")
    for config in configs:
        for split in config["data_files"]:
            if not (RELEASE / split["path"]).is_file():
                raise AssertionError(f"README config points to missing file: {split['path']}")
    cloze = next(config for config in configs if config["config_name"] == "cloze")
    cloze_splits = {item["split"] for item in cloze["data_files"]}
    if cloze_splits != {"diac_set1", "nodiac_set1"}:
        raise AssertionError(f"Unexpected cloze split names: {sorted(cloze_splits)}")
    cloze_paths = {item["path"] for item in cloze["data_files"]}
    if cloze_paths != {
        "prompts/cloze_diac_set1.jsonl",
        "prompts/cloze_nodiac_set1.jsonl",
    }:
        raise AssertionError(f"Unexpected cloze file paths: {sorted(cloze_paths)}")


def test_cloze_records_include_demo_metadata_and_native_json_types() -> None:
    """Cloze prompt records carry three demos and a null few-shot-set marker."""
    for path in sorted((RELEASE / "prompts").glob("cloze_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != (41 if path.name == "cloze_diac_set1.jsonl" else 29):
            raise AssertionError(f"Unexpected cloze row count in {path.name}: {len(rows)}")
        for row in rows:
            if row["fewshot_set"] is not None:
                raise AssertionError(f"Cloze fewshot_set must be JSON null in {path.name}")
            if not isinstance(row["demo_concept_ids"], list) or len(row["demo_concept_ids"]) != 3:
                raise AssertionError(f"Invalid cloze demo_concept_ids in {path.name}")
            if not isinstance(row["m1_extension"], bool):
                raise AssertionError(f"Cloze m1_extension must be a JSON boolean in {path.name}")


def test_readme_checksums_cover_concepts_and_every_prompt_file() -> None:
    """The Files and checksums table is complete and matches the released data."""
    readme = (RELEASE / "README.md").read_text(encoding="utf-8")
    if "## Files and checksums" not in readme:
        raise AssertionError("README is missing the Files and checksums section")
    section = readme.split("## Files and checksums", 1)[1].split("\n## ", 1)[0]
    tick = chr(96)
    found = {
        match.group(1): match.group(2)
        for match in re.finditer(
            rf"\| {tick}([^`]+){tick} \| {tick}([0-9a-f]{{64}}){tick} \|", section
        )
    }
    expected_paths = {
        path.relative_to(RELEASE).as_posix()
        for path in [RELEASE / "concepts.jsonl", *sorted((RELEASE / "prompts").glob("*.jsonl"))]
    }
    if set(found) != expected_paths:
        raise AssertionError(
            f"README checksum coverage mismatch: missing={sorted(expected_paths - set(found))}, "
            f"extra={sorted(set(found) - expected_paths)}"
        )
    for relative, digest in found.items():
        if sha256(RELEASE / relative) != digest:
            raise AssertionError(f"README checksum mismatch for {relative}")


def test_readme_links_to_no_absent_local_files() -> None:
    """All relative README links resolve inside the data-only release tree."""
    readme = (RELEASE / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
        if "://" in target or target.startswith("#"):
            continue
        local_path = target.split("#", 1)[0]
        if local_path and not (RELEASE / local_path).exists():
            raise AssertionError(f"README links to an absent release file: {target}")


def test_readme_known_issues_and_changelog_are_present() -> None:
    """The v1.3 cloze contract and frozen-data caveats are documented."""
    readme = (RELEASE / "README.md").read_text(encoding="utf-8")
    required = (
        "The v1.2 cloze set is superseded",
        "cloze_available",
        "v1.2 `prompts` config cannot be loaded with `load_dataset`",
        "schema mismatch",
        "target sense is verified; uniqueness is not",
        "1626f004dcd7",
        "f918ffa42beb",
        "prompts preserve case",
        "thứ tư, tiệc, and sắt",
        "151a3fe224ea",
        "17781bd98b3c",
        "922cd1167021",
        "c9584b841817",
        "c505c130104c",
        "**v1.3 (27 Sep 2026, tag `v1.3`):**",
        "**v1.3.1:**",
        "C1–C8 plus beam agreement",
        "semantic_min_beam_matches=3",
        "41 diac records (34 main + 7 extension)",
        "29 nodiac records (29 main + 0 extension)",
        "few-shot set 1",
        "few-shot set 5",
        "directions split",
        "f5411853b246",
        "ne... verb ...pas",
        "Tokens of the form after a colon, including its leading space",
    )
    missing = [phrase for phrase in required if phrase not in readme]
    if missing:
        raise AssertionError(f"README is missing required v1.3 documentation: {missing}")


def test_no_flores_sentence_occurs_in_any_release_file() -> None:
    """No complete FLORES dev/devtest sentence is present in released files."""
    sentences = _flores_sentences()
    combined = "\0".join(path.read_text(encoding="utf-8") for path in release_files())
    if contains_any(combined, sentences):
        raise AssertionError("FLORES sentence text found in a release file")


def test_no_token_or_huggingface_credential_pattern_occurs_in_release() -> None:
    """The configured upload token and token-shaped strings are absent."""
    token_path = ROOT / "keys" / "hf-w.txt"
    token = token_path.read_bytes().strip()
    if not token:
        raise AssertionError("The configured Hugging Face token file is empty")
    credential_pattern = re.compile(rb"hf_[A-Za-z0-9]{10,}")
    for path in release_files():
        content = path.read_bytes()
        if token in content or credential_pattern.search(content):
            raise AssertionError(f"Credential-like content found in release file {path.relative_to(RELEASE)}")


def test_no_user_identity_except_requested_preparation_link() -> None:
    """Only the required explicit repository and HF tag links may contain the owner name."""
    allowed = (PREPARATION_TAG_URL.encode("utf-8").lower(), HF_V12_TAG_URL.encode("utf-8").lower())
    forbidden = (b"tarudesu", b"luannt@uit.edu.vn")
    for path in release_files():
        content = path.read_bytes().lower()
        for url in allowed:
            content = content.replace(url, b"")
        if any(value in content for value in forbidden):
            raise AssertionError(f"User identity found in release file {path.relative_to(RELEASE)}")


def test_concepts_are_jsonl_and_omit_the_muse_flag() -> None:
    """Concept objects are valid JSON Lines and do not distribute the MUSE flag."""
    path = RELEASE / "concepts.jsonl"
    count = 0
    expected_fields: set[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            fields = set(row)
            if expected_fields is None:
                expected_fields = fields
            elif fields != expected_fields:
                raise AssertionError("Concept JSONL rows have inconsistent fields")
            if "in_muse" in fields:
                raise AssertionError("The MUSE attestation flag must not be released")
    if count != 1837:
        raise AssertionError(f"Expected 1,837 concept records, found {count}")
