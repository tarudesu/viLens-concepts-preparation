"""Count contextual tokens for the three study tokenizers (never load weights)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

try:
    from common import load_config, normalize_nfc, normalize_strings, setup_logging
except ModuleNotFoundError:
    from data.build.common import load_config, normalize_nfc, normalize_strings, setup_logging


STEP = "10_tokens"
MODEL_KEYS = ("gemma", "qwen", "llama")
LANGUAGE_COLUMNS = {
    "vi": "vi_canonical",
    "en": "en_lemma",
    "zh": "zh_canonical",
    "fr": "fr_canonical",
    "id": "id_canonical",
}
EXPECTED_SPLITS = {"fewshot_reservoir", "directions", "test"}
EXPECTED_STRATA = {"sino", "nonsino", "ambiguous", "other_loan"}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def contextual_tokenization(tokenizer: Any, prefix: str, word: str) -> tuple[int, list[str]]:
    """Return the contextual token-count difference and its token-string suffix."""
    prefix = normalize_nfc(prefix)
    word = normalize_nfc(word)
    if not prefix.strip():
        raise ValueError("Token context prefix must be non-empty")
    if not word.strip():
        raise ValueError("Cannot tokenize an empty canonical form")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    context_ids = tokenizer.encode(f"{prefix} {word}", add_special_tokens=False)
    if len(context_ids) < len(prefix_ids):
        raise ValueError(
            "Context encoding is shorter than prefix encoding for "
            f"prefix={prefix!r}, word={word!r}: {len(context_ids)} < {len(prefix_ids)}"
        )
    token_strings = tokenizer.convert_ids_to_tokens(context_ids)
    if isinstance(token_strings, str):
        token_strings = [token_strings]
    if not isinstance(token_strings, list) or len(token_strings) != len(context_ids):
        raise ValueError(
            "Tokenizer returned an unexpected token-string sequence for "
            f"prefix={prefix!r}, word={word!r}"
        )
    if any(not isinstance(token, str) for token in token_strings):
        raise ValueError(f"Tokenizer returned a non-string token for {word!r}: {token_strings!r}")
    count = len(context_ids) - len(prefix_ids)
    incremental_tokens = [normalize_nfc(token) for token in token_strings[len(prefix_ids):]]
    if len(incremental_tokens) != count:
        raise ValueError(f"Tokenizer count/token-string mismatch for {word!r}")
    return count, incremental_tokens


def syllable_bucket(word: str, edges: list[int]) -> str:
    """Bucket a Vietnamese form by its configured whitespace-token count."""
    if len(edges) != 2 or any(type(edge) is not int for edge in edges) or edges != sorted(set(edges)) or edges[0] < 1:
        raise ValueError(f"syllable bins must be two ascending positive integers, got {edges!r}")
    count = len(normalize_nfc(word).strip().split())
    if count <= edges[0]:
        return str(edges[0])
    if count <= edges[1]:
        return str(edges[1])
    return f"{edges[1] + 1}+"


def _validate_rows(rows: list[dict[str, Any]], languages: list[str]) -> None:
    """Fail with diagnostics if step-08 rows do not match the expected schema."""
    required = {"concept_id", "split", "stratum", *LANGUAGE_COLUMNS.values()}
    if not rows:
        raise ValueError("Tokenization input contains no concepts")
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Tokenization input row {index} is missing required fields: {missing!r}")
        concept_id = row["concept_id"]
        if not isinstance(concept_id, str) or not concept_id:
            raise ValueError(f"Tokenization input row {index} has malformed concept_id: {concept_id!r}")
        if concept_id in ids:
            raise ValueError(f"Duplicate concept_id in step-08 input: {concept_id!r}")
        ids.add(concept_id)
        if not isinstance(row["split"], str) or row["split"] not in EXPECTED_SPLITS:
            raise ValueError(f"Unexpected split in tokenization row {concept_id}: {row['split']!r}")
        if not isinstance(row["stratum"], str) or row["stratum"] not in EXPECTED_STRATA:
            raise ValueError(f"Unexpected etymology stratum in tokenization row {concept_id}: {row['stratum']!r}")
        for language in languages:
            field = LANGUAGE_COLUMNS[language]
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing/non-string canonical {language} value for {concept_id}: {value!r}")


def _load_tokenizer(
    model_key: str, settings: dict[str, Any], cache_dir: Path,
) -> Any:
    """Load only tokenizer artifacts from the model's pinned Hub commit."""
    model_id = settings.get("hf_id")
    revision = settings.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"models.{model_key}.hf_id must be a non-empty string")
    if not isinstance(revision, str) or not COMMIT_RE.fullmatch(revision):
        raise ValueError(f"models.{model_key}.revision must be a pinned 40-character commit hash, got {revision!r}")
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache_dir),
        use_fast=bool(settings.get("use_fast", True)),
    )


def _schema_with_tokens(input_schema: pa.Schema, model_keys: tuple[str, ...], languages: list[str]) -> pa.Schema:
    """Append stable count, token-string, and VI single-token fields."""
    additions: list[pa.Field] = []
    for model_key in model_keys:
        for language in languages:
            additions.extend([
                pa.field(f"ntok_{model_key}_{language}", pa.int32(), nullable=False),
                pa.field(f"tokstr_{model_key}_{language}", pa.list_(pa.string()), nullable=False),
            ])
        additions.append(pa.field(f"single_token_vi_{model_key}", pa.bool_(), nullable=False))
    collisions = sorted({field.name for field in additions}.intersection(input_schema.names))
    if collisions:
        raise ValueError(f"Tokenization input unexpectedly already contains step-10 columns: {collisions!r}")
    return pa.schema([*input_schema, *additions])


def _write_parquet(table: pa.Table, path: Path, row_group_size: int) -> None:
    """Atomically write deterministic Zstandard Parquet output."""
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


def _report(
    rows: list[dict[str, Any]], config: dict[str, Any], logger: logging.Logger,
    model_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Log test-set token summaries overall, by etymology stratum, and VI syllables."""
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise ValueError("Step-08 input has no test-split concepts")
    languages = config["langs"]
    gate_threshold = float(config["tokens"]["e1_gate_share_threshold"])
    m1_minimum = config["tokens"]["m1_min_single_token_n"]
    if not 0.0 <= gate_threshold <= 1.0 or type(m1_minimum) is not int or m1_minimum < 0:
        raise ValueError("Invalid E1 share threshold or M1 minimum in tokens config")
    group_rows: list[tuple[str, list[dict[str, Any]]]] = [("all", test_rows)]
    for stratum in sorted({row["stratum"] for row in test_rows}):
        group_rows.append((f"stratum:{stratum}", [row for row in test_rows if row["stratum"] == stratum]))

    logger.info("TEST split contextual token counts (mean / median / max):")
    token_stats: dict[str, dict[str, float | int]] = {}
    for group_name, selected in group_rows:
        for model_key in model_keys:
            for language in languages:
                values = np.asarray([row[f"ntok_{model_key}_{language}"] for row in selected], dtype=np.int64)
                stats: dict[str, float | int] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "max": int(np.max(values)),
                }
                token_stats[f"{group_name}|{model_key}|{language}"] = stats
                logger.info(
                    "%s | %s x %s | n=%d mean=%.6f median=%.6f max=%d",
                    group_name, model_key, language, len(selected), stats["mean"], stats["median"], stats["max"],
                )

    single_counts: dict[str, int] = {}
    overall_shares: dict[str, float] = {}
    logger.info("TEST single-token Vietnamese coverage by model, stratum, and syllable bucket:")
    for model_key in model_keys:
        field = f"single_token_vi_{model_key}"
        count = sum(bool(row[field]) for row in test_rows)
        share = count / len(test_rows)
        single_counts[model_key] = count
        overall_shares[model_key] = share
        logger.info("all | %s | %d/%d (%.6f)", model_key, count, len(test_rows), share)
        for stratum in sorted({row["stratum"] for row in test_rows}):
            selected = [row for row in test_rows if row["stratum"] == stratum]
            n_single = sum(bool(row[field]) for row in selected)
            logger.info("stratum=%s | %s | %d/%d (%.6f)", stratum, model_key, n_single, len(selected), n_single / len(selected))
        syllable_edges = config["split"]["syllable_bins"]
        for bucket in (str(syllable_edges[0]), str(syllable_edges[1]), f"{syllable_edges[1] + 1}+"):
            selected = [row for row in test_rows if syllable_bucket(row["vi_canonical"], syllable_edges) == bucket]
            n_single = sum(bool(row[field]) for row in selected)
            logger.info("syllables=%s | %s | %d/%d (%.6f)", bucket, model_key, n_single, len(selected), n_single / len(selected) if selected else 0.0)

    gate = all(overall_shares[model_key] > gate_threshold for model_key in model_keys)
    adequate = {model_key: single_counts[model_key] >= m1_minimum for model_key in model_keys}
    for model_key in model_keys:
        logger.info(
            "E1 preview | %s | single_token_vi=%d/%d share=%.6f | M1 >=%d: %s",
            model_key, single_counts[model_key], len(test_rows), overall_shares[model_key], m1_minimum,
            adequate[model_key],
        )
    logger.info("E1 gate (>%.1f%% for all three models): %s", gate_threshold * 100.0, gate)
    return {"test_n": len(test_rows), "single_counts": single_counts, "single_shares": overall_shares, "e1_gate": gate, "m1_adequate": adequate, "token_stats": token_stats}


def run(config_path: str | Path) -> dict[str, Any]:
    """Run contextual token counts on the complete step-09 concept table."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["tokens"]
    languages = config["langs"]
    if len(languages) != len(set(languages)) or set(languages) != set(LANGUAGE_COLUMNS):
        raise ValueError(f"langs must contain exactly {sorted(LANGUAGE_COLUMNS)!r}, got {languages!r}")
    model_keys = tuple(settings.get("models", MODEL_KEYS))
    if set(model_keys) != set(MODEL_KEYS) or len(model_keys) != len(MODEL_KEYS):
        raise ValueError(f"tokens.models must name exactly {MODEL_KEYS!r}, got {model_keys!r}")
    prefix = settings.get("context_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("tokens.context_prefix must be a non-empty string")
    if type(settings.get("use_fast")) is not bool:
        raise ValueError("tokens.use_fast must be a boolean")
    row_group_size = config["split"]["parquet_row_group_size"]
    progress_every = config["logging"]["progress_every"]
    if type(row_group_size) is not int or row_group_size < 1 or type(progress_every) is not int or progress_every < 1:
        raise ValueError("parquet_row_group_size and logging.progress_every must be positive integers")

    input_path = Path(settings["paths"]["input"])
    output_path = Path(settings["paths"]["output"])
    cache_dir = Path(settings["paths"]["tokenizer_cache"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Step-09 input not found: {input_path}")
    table = pq.read_table(input_path)
    input_schema = table.schema
    rows = [normalize_strings(row) for row in table.to_pylist()]
    rows.sort(key=lambda row: normalize_nfc(row["concept_id"]) if isinstance(row.get("concept_id"), str) else "")
    _validate_rows(rows, languages)
    logger.info("Loaded %d concepts from %s; tokenizers only", len(rows), input_path)

    all_models = config["models"]
    output_schema = _schema_with_tokens(input_schema, model_keys, languages)
    for model_key in model_keys:
        model_settings = dict(all_models[model_key])
        model_settings["use_fast"] = settings.get("use_fast", True)
        revision = model_settings.get("revision")
        logger.info("Loading tokenizer only: %s@%s", model_settings.get("hf_id"), revision)
        tokenizer = _load_tokenizer(model_key, model_settings, cache_dir)
        for language in languages:
            field = LANGUAGE_COLUMNS[language]
            counts: list[int] = []
            token_strings: list[list[str]] = []
            for index, row in enumerate(rows, start=1):
                count, tokens = contextual_tokenization(tokenizer, prefix, row[field])
                counts.append(count)
                token_strings.append(tokens)
                if index % progress_every == 0 or index == len(rows):
                    logger.info("Tokenized %s | %s: %d/%d concepts", model_key, language, index, len(rows))
            count_name = f"ntok_{model_key}_{language}"
            token_name = f"tokstr_{model_key}_{language}"
            for index, row in enumerate(rows):
                row[count_name] = counts[index]
                row[token_name] = token_strings[index]
        vi_counts = [row["ntok_" + model_key + "_vi"] for row in rows]
        for row, count in zip(rows, vi_counts, strict=True):
            row[f"single_token_vi_{model_key}"] = count == 1
        del tokenizer
        logger.info("Finished tokenizer %s@%s", model_settings["hf_id"], revision)

    out_table = pa.Table.from_pylist(rows, schema=output_schema)
    _write_parquet(out_table, output_path, row_group_size)
    logger.info("tokstr fields store the no-special-token context encoding after subtracting the prefix token count.")
    report = _report(rows, config, logger, model_keys)
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
            logger.error("Step 10 stopped: %s: %s", type(exc).__name__, exc)
        else:
            print(f"ERROR Step 10 stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
