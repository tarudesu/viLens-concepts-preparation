"""Measure Vietnamese diacritic collapse and tokenize stripped VI forms."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from wordfreq import top_n_list

try:
    from common import contextual_tokenization, iter_jsonl, load_config, normalize_nfc, normalize_strings, setup_logging, strip_diacritics, vi_orth_key
except ModuleNotFoundError:
    from data.build.common import contextual_tokenization, iter_jsonl, load_config, normalize_nfc, normalize_strings, setup_logging, strip_diacritics, vi_orth_key


STEP = "11_diacritics"
MODEL_KEYS = ("gemma", "qwen", "llama")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
CANONICAL_COLUMNS = {"vi": "vi_canonical", "en": "en_lemma", "zh": "zh_canonical", "fr": "fr_canonical", "id": "id_canonical"}
SPLITS = ("fewshot_reservoir", "directions", "test")


def add_reference_form(index: dict[str, set[str]], form: str) -> None:
    """Index a Vietnamese surface under its stripped spelling and orthographic key."""
    if not isinstance(form, str):
        raise TypeError(f"Vietnamese reference form must be a string, got {type(form).__name__}")
    value = normalize_nfc(form).strip()
    if not value:
        raise ValueError("Vietnamese reference form is empty")
    stripped = strip_diacritics(value)
    key = vi_orth_key(value)
    if not key:
        raise ValueError(f"Vietnamese reference form normalized to an empty key: {form!r}")
    index.setdefault(stripped, set()).add(key)


def find_collapse_partners(canonical: str, index: dict[str, set[str]], limit: int) -> list[str]:
    """Return stable other orthographic keys sharing the canonical stripped form."""
    if type(limit) is not int or limit < 0:
        raise ValueError(f"collapse partner limit must be a non-negative integer, got {limit!r}")
    value = normalize_nfc(canonical).strip()
    stripped = strip_diacritics(value)
    own_key = vi_orth_key(value)
    return sorted(index.get(stripped, set()) - {own_key})[:limit]


def _read_reference_index(dump_path: Path, wordfreq_top: list[str], requested_n: int, logger: logging.Logger) -> tuple[dict[str, set[str]], int, int]:
    """Stream Wiktextract headwords and add the complete configured wordfreq list."""
    index: dict[str, set[str]] = {}
    record_count = 0
    distinct_headwords: set[str] = set()
    empty_stripped_headwords = 0
    for record_count, record in enumerate(iter_jsonl(dump_path), start=1):
        if not isinstance(record, dict) or not isinstance(record.get("word"), str) or not record["word"].strip():
            raise ValueError(f"Vietnamese Wiktextract record {record_count} has no non-empty word headword")
        headword = normalize_nfc(record["word"].strip())
        distinct_headwords.add(headword)
        empty_stripped_headwords += not bool(strip_diacritics(headword))
        add_reference_form(index, headword)
    logger.info("Wiktextract Vietnamese headwords: records=%d distinct=%d empty_stripped=%d", record_count, len(distinct_headwords), empty_stripped_headwords)
    empty_stripped_wordfreq = 0
    for form in wordfreq_top:
        empty_stripped_wordfreq += not bool(strip_diacritics(form))
        add_reference_form(index, form)
    logger.info("wordfreq Vietnamese list request=%d returned=%d distinct=%d empty_stripped=%d", requested_n, len(wordfreq_top), len(set(wordfreq_top)), empty_stripped_wordfreq)
    return index, record_count, len(distinct_headwords)


def _load_tokenizer(model_key: str, model: dict[str, Any], cache_dir: Path, use_fast: bool) -> Any:
    """Load tokenizer artifacts only from the local cache at a pinned commit."""
    model_id, revision = model.get("hf_id"), model.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"models.{model_key}.hf_id must be non-empty")
    if not isinstance(revision, str) or not COMMIT_RE.fullmatch(revision):
        raise ValueError(f"models.{model_key}.revision must be a pinned commit hash, got {revision!r}")
    return AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=str(cache_dir), use_fast=use_fast, local_files_only=True,
    )


def _write_parquet(table: pa.Table, path: Path, row_group_size: int) -> None:
    """Atomically write deterministic Zstandard Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as handle:
            temporary = handle.name
        pq.write_table(table, temporary, compression="zstd", version="2.6", row_group_size=row_group_size, write_statistics=True)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _validate_rows(rows: list[dict[str, Any]], languages: list[str]) -> None:
    """Fail on unexpected upstream fields or malformed split metadata."""
    if not rows:
        raise ValueError("Step-10 input contains no concepts")
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        required = {"concept_id", "split", "stratum", *[CANONICAL_COLUMNS[language] for language in languages]}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Step-10 row {index} is missing required fields: {missing!r}")
        concept_id = row["concept_id"]
        if not isinstance(concept_id, str) or not concept_id or concept_id in ids:
            raise ValueError(f"Malformed or duplicate concept_id at step-10 row {index}: {concept_id!r}")
        ids.add(concept_id)
        if row["split"] not in SPLITS:
            raise ValueError(f"Unexpected split for {concept_id}: {row['split']!r}")
        if not isinstance(row["stratum"], str) or not row["stratum"]:
            raise ValueError(f"Missing stratum for {concept_id}: {row['stratum']!r}")
        for language in languages:
            field = CANONICAL_COLUMNS[language]
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Missing canonical {language} form for {concept_id}: {row[field]!r}")


def _report(rows: list[dict[str, Any]], logger: logging.Logger, example_n: int) -> dict[str, Any]:
    """Log test-set collapse rates, examples, and E9 non-collapsed availability."""
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise ValueError("Step-10 input has no test concepts")
    counts: dict[str, tuple[int, int]] = {}
    groups = [("overall", test_rows)]
    groups.extend((f"stratum:{label}", [row for row in test_rows if row["stratum"] == label]) for label in sorted({row["stratum"] for row in test_rows}))
    groups.extend([
        ("syllables:1", [row for row in test_rows if row["n_syllables"] == 1]),
        ("syllables:2", [row for row in test_rows if row["n_syllables"] == 2]),
        ("syllables:3+", [row for row in test_rows if row["n_syllables"] >= 3]),
    ])
    for label, selected in groups:
        numerator = sum(bool(row["collapsed"]) for row in selected)
        denominator = len(selected)
        counts[label] = (numerator, denominator)
        logger.info("TEST collapse | %s | %d/%d (%.6f)", label, numerator, denominator, numerator / denominator if denominator else 0.0)
    examples = [row for row in test_rows if row["collapsed"]]
    examples.sort(key=lambda row: row["concept_id"])
    logger.info("Collapsed test examples (up to %d): vi | collapse_partners", example_n)
    for row in examples[:example_n]:
        logger.info("%s | %s", row["vi_canonical"], " / ".join(row["collapse_partners"]))
    e9_n = sum(not row["collapsed"] for row in test_rows)
    logger.info("E9 test concepts available (non-collapsed): %d/%d", e9_n, len(test_rows))
    return {"collapse_counts": counts, "e9_noncollapsed_test_n": e9_n, "examples": examples[:example_n]}


def run(config_path: str | Path) -> dict[str, Any]:
    """Stream the reference vocabulary, add collapse flags, and count stripped-form tokens."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["diacritics"]
    languages = config["langs"]
    if len(languages) != len(set(languages)) or set(languages) != set(CANONICAL_COLUMNS):
        raise ValueError(f"langs must contain exactly {sorted(CANONICAL_COLUMNS)!r}, got {languages!r}")
    paths = settings["paths"]
    input_path, output_path = Path(paths["input"]), Path(paths["output"])
    dump_path = Path(paths["vietnamese_dump"])
    for required_path in (input_path, dump_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Step-11 required input not found: {required_path}")
    table = pq.read_table(input_path)
    rows = [normalize_strings(row) for row in table.to_pylist()]
    rows.sort(key=lambda row: normalize_nfc(row.get("concept_id", "")))
    _validate_rows(rows, languages)
    if any(name in table.column_names for name in ("vi_nodiac", "collapsed", "collapse_partners", *[f"ntok_{model}_vi_nodiac" for model in MODEL_KEYS])):
        raise ValueError("Step-10 input unexpectedly contains step-11 output columns")

    requested_n = settings["wordfreq_top_n"]
    expected_n = settings["expected_wordfreq_n"]
    partner_limit = settings["max_collapse_partners"]
    example_n = settings["report_examples_n"]
    if any(type(value) is not int or value < 0 for value in (requested_n, expected_n, partner_limit, example_n)) or requested_n < 1:
        raise ValueError("Invalid integer in diacritics configuration")
    vocabulary = [normalize_nfc(value) for value in top_n_list("vi", requested_n)]
    if len(vocabulary) != expected_n:
        raise ValueError(f"wordfreq Vietnamese vocabulary size mismatch: expected {expected_n}, received {len(vocabulary)}")
    index, headword_records, unique_headwords = _read_reference_index(dump_path, vocabulary, requested_n, logger)
    logger.info("Reference index sizes | Wiktextract headword records=%d unique_headwords=%d | wordfreq words=%d", headword_records, unique_headwords, len(vocabulary))

    token_settings = config["tokens"]
    prefix = token_settings["context_prefix"]
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("tokens.context_prefix must be non-empty")
    cache_dir = Path(token_settings["paths"]["tokenizer_cache"])
    use_fast = token_settings["use_fast"]
    row_group_size = config["split"]["parquet_row_group_size"]
    progress_every = config["logging"]["progress_every"]
    if type(row_group_size) is not int or row_group_size < 1 or type(progress_every) is not int or progress_every < 1:
        raise ValueError("parquet_row_group_size and logging.progress_every must be positive integers")

    for row in rows:
        canonical = row["vi_canonical"]
        nodiac = strip_diacritics(canonical)
        partners = find_collapse_partners(canonical, index, partner_limit)
        row["vi_nodiac"] = nodiac
        row["collapsed"] = bool(partners)
        row["collapse_partners"] = partners

    for model_key in MODEL_KEYS:
        logger.info("Loading tokenizer only from local cache: %s@%s", config["models"][model_key]["hf_id"], config["models"][model_key]["revision"])
        tokenizer = _load_tokenizer(model_key, config["models"][model_key], cache_dir, use_fast)
        field = f"ntok_{model_key}_vi_nodiac"
        for index_row, row in enumerate(rows, start=1):
            count, _ = contextual_tokenization(tokenizer, prefix, row["vi_nodiac"])
            row[field] = count
            if index_row % progress_every == 0 or index_row == len(rows):
                logger.info("Tokenized diacritic-stripped VI | %s: %d/%d concepts", model_key, index_row, len(rows))
        del tokenizer

    additions = [
        pa.field("vi_nodiac", pa.string(), nullable=False),
        pa.field("collapsed", pa.bool_(), nullable=False),
        pa.field("collapse_partners", pa.list_(pa.string()), nullable=False),
        *[pa.field(f"ntok_{model_key}_vi_nodiac", pa.int32(), nullable=False) for model_key in MODEL_KEYS],
    ]
    out_table = pa.Table.from_pylist(rows, schema=pa.schema([*table.schema, *additions]))
    _write_parquet(out_table, output_path, row_group_size)
    report = _report(rows, logger, example_n)
    runtime = time.monotonic() - started
    logger.info("Output: %s (%d concepts)", output_path, len(rows))
    logger.info("Runtime seconds: %.3f", runtime)
    return {"rows": len(rows), "output": str(output_path), "report": report, "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        logger = logging.getLogger(STEP)
        if logger.handlers:
            logger.error("Step 11 stopped: %s: %s", type(exc).__name__, exc)
        else:
            print(f"ERROR Step 11 stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
