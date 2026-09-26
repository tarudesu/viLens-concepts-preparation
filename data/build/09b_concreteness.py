"""Resolve pending English concreteness values for the step-11 concept tables."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import spacy

try:
    from common import load_config, normalize_nfc, normalize_strings, setup_logging
except ModuleNotFoundError:
    from data.build.common import load_config, normalize_nfc, normalize_strings, setup_logging


STEP = "09b_concreteness"


def missing_inputs_message(paths: list[Path]) -> str:
    """Return the stable, actionable diagnostic for absent step-11 tables."""
    return "Step 09b requires step-11 main and M1 extension inputs: missing " + ", ".join(map(str, paths))


def missing_norms_message(directory: Path) -> str:
    """Return the stable manual-download diagnostic for absent source files."""
    return (
        f"Step 09b requires Brysbaert file(s) in {directory}/; none found. "
        "Place the original norms file there, then run make concreteness."
    )


def norm_key(value: str) -> str:
    """Normalize a norms-table word for exact lexical lookup."""
    return normalize_nfc(value).strip().lower()


def _parse_norm_value(raw_value: Any, *, value_column: str, row_label: str) -> float | None:
    """Parse a possibly missing concreteness score without imputing it."""
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and not raw_value.strip():
        return None
    try:
        if math.isnan(float(raw_value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Brysbaert {row_label} has invalid {value_column}: {raw_value!r}") from exc


def _parse_binary_bigram(value: Any, *, row_label: str, column: str) -> int:
    """Require Brysbaert's Bigram marker to be exactly 0 or 1."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Brysbaert {row_label} has invalid {column}: {value!r}; expected 0 or 1") from exc
    if numeric not in (0.0, 1.0):
        raise ValueError(f"Brysbaert {row_label} has invalid {column}: {value!r}; expected 0 or 1")
    return int(numeric)


def load_norms(
    path: str | Path, *, word_column: str, value_column: str,
    bigram_column: str | None = None, raters_column: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[tuple[str, int], float | None]:
    """Read Brysbaert CSV/XLSX values, using the highest-rater duplicate row."""
    norms_path = Path(path)
    if logger is None:
        logger = logging.getLogger(STEP)
    rows: list[tuple[int, str, int, float | None, float]] = []
    if norms_path.suffix.casefold() in {".xlsx", ".xlsm"}:
        import pandas as pd

        frame = pd.read_excel(
            norms_path, engine="openpyxl", keep_default_na=False, na_values={},
        )
        required_columns = {word_column, value_column}
        if bigram_column:
            required_columns.add(bigram_column)
        if raters_column:
            required_columns.add(raters_column)
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(
                f"Unexpected Brysbaert workbook headers in {norms_path}; "
                f"missing required columns {missing!r}, found {list(frame.columns)!r}"
            )
        for row_number, record in enumerate(frame.to_dict(orient="records"), start=2):
            raw_word = record.get(word_column)
            if not isinstance(raw_word, str) or not raw_word.strip():
                raise ValueError(f"Brysbaert workbook row {row_number} has no word in {word_column!r}")
            key = norm_key(raw_word)
            if not key:
                raise ValueError(f"Brysbaert workbook row {row_number} normalizes to an empty word")
            bigram = _parse_binary_bigram(record.get(bigram_column), row_label=f"workbook row {row_number}", column=bigram_column) if bigram_column else int(len(key.split()) > 1)
            score = _parse_norm_value(record.get(value_column), value_column=value_column, row_label=f"workbook row {row_number}")
            raw_total = record.get(raters_column) if raters_column else 0
            try:
                total = float(raw_total)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Brysbaert workbook row {row_number} has invalid {raters_column}: {raw_total!r}") from exc
            if not math.isfinite(total) or total < 0:
                raise ValueError(f"Brysbaert workbook row {row_number} has invalid {raters_column}: {raw_total!r}")
            rows.append((row_number, key, bigram, score, total))
    else:
        with norms_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Brysbaert CSV has no header: {norms_path}")
            missing = sorted({word_column, value_column} - set(reader.fieldnames))
            if missing:
                raise ValueError(
                    f"Unexpected Brysbaert CSV headers in {norms_path}; "
                    f"missing required columns {missing!r}, found {reader.fieldnames!r}"
                )
            for row_number, record in enumerate(reader, start=2):
                raw_word = record.get(word_column)
                if not isinstance(raw_word, str) or not raw_word.strip():
                    raise ValueError(f"Brysbaert CSV row {row_number} has no word in {word_column!r}")
                key = norm_key(raw_word)
                if not key:
                    raise ValueError(f"Brysbaert CSV row {row_number} normalizes to an empty word")
                bigram = int(len(key.split()) > 1)
                score = _parse_norm_value(record.get(value_column), value_column=value_column, row_label=f"CSV row {row_number}")
                rows.append((row_number, key, bigram, score, 0.0))

    word_counts: dict[str, int] = {}
    grouped: dict[tuple[str, int], list[tuple[int, float | None, float]]] = {}
    for row_number, key, bigram, score, total in rows:
        word_counts[key] = word_counts.get(key, 0) + 1
        grouped.setdefault((key, bigram), []).append((row_number, score, total))
    for key, count in sorted(word_counts.items()):
        if count > 1:
            categories = sorted(bigram for word, bigram in grouped if word == key)
            logger.warning("Duplicate normalized Brysbaert Word=%r rows=%d Bigram categories=%s", key, count, categories)

    result: dict[tuple[str, int], float | None] = {}
    for key, candidates in sorted(grouped.items()):
        max_total = max(candidate[2] for candidate in candidates)
        winners = [candidate for candidate in candidates if candidate[2] == max_total]
        winner_scores = {candidate[1] for candidate in winners}
        if len(winner_scores) > 1:
            raise ValueError(
                f"Brysbaert duplicate {key[0]!r} Bigram={key[1]} has tied highest Total="
                f"{max_total} with conflicting Conc.M values {sorted(winner_scores, key=str)!r}"
            )
        if len(candidates) > 1:
            logger.info(
                "Resolved duplicate Brysbaert Word=%r Bigram=%d using row %d (highest Total=%s)",
                key[0], key[1], winners[0][0], max_total,
            )
        result[key] = winners[0][1]
    if not result:
        raise ValueError(f"Brysbaert norms file contains no rows: {norms_path}")
    return result


def _head_lemma(text: str, pos: str, nlp: Callable[[str], Any]) -> str | None:
    """Extract the noun phrase head for nouns, or the syntactic root otherwise."""
    doc = nlp(text)
    tokens = list(doc)
    if not tokens:
        return None
    if pos.casefold() in {"noun", "n"}:
        nouns = [token for token in tokens if getattr(token, "pos_", "") == "NOUN"]
        if nouns:
            roots = [token for token in nouns if getattr(token, "dep_", "") == "ROOT"]
            selected = roots[0] if roots else nouns[-1]
        else:
            selected = next((token for token in tokens if getattr(token, "dep_", "") == "ROOT"), tokens[0])
    elif pos.casefold() in {"verb", "adj", "adjective", "v", "a"}:
        selected = next((token for token in tokens if getattr(token, "dep_", "") == "ROOT"), tokens[0])
    else:
        raise ValueError(f"Unexpected POS for concreteness head lookup: {pos!r}")
    lemma = getattr(selected, "lemma_", "")
    if not isinstance(lemma, str) or not lemma.strip():
        lemma = getattr(selected, "text", "")
    return normalize_nfc(lemma).strip() or None


def match_concreteness(
    en_form: str, pos: str, norms: dict[tuple[str, int], float | None], nlp: Callable[[str], Any],
) -> tuple[float | None, str]:
    """Match exact expression, then spaCy head lemma, otherwise leave NA."""
    exact_key = norm_key(en_form)
    word_count = len(exact_key.split())
    if word_count in (1, 2):
        key = (exact_key, word_count - 1)
        if key in norms:
            return norms[key], "exact"
    head = _head_lemma(normalize_nfc(en_form).strip(), normalize_nfc(pos).strip(), nlp)
    if head is not None and (norm_key(head), 0) in norms:
        return norms[(norm_key(head), 0)], "head"
    return None, "none"


def _hash_manual_source(config_path: Path) -> None:
    """Delegate provenance recording to step 01's hash-only local-file path."""
    script = Path(__file__).with_name("01_download.py")
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), "--only", "brysbaert"],
        check=False, text=True, capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode:
        raise RuntimeError(f"01_download.py --only brysbaert failed with exit code {result.returncode}")


def _atomic_update_table(table: pa.Table, rows: list[dict[str, Any]], path: Path) -> None:
    """Write a stable parquet replacement while preserving the original schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as handle:
            temporary = handle.name
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary,
                       compression="zstd", version="2.6", write_statistics=True)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _enrich_table(path: Path, norms: dict[str, float | None], nlp: Callable[[str], Any]) -> tuple[pa.Table, list[dict[str, Any]]]:
    """Attach matched scores to a main or extension parquet table."""
    table = pq.read_table(path)
    required = {"concept_id", "en_lemma", "pos", "concreteness", "concreteness_match"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"Step-11 table {path} is missing required columns: {missing!r}")
    rows = [normalize_strings(row) for row in table.to_pylist()]
    for row in rows:
        if not isinstance(row["en_lemma"], str) or not isinstance(row["pos"], str):
            raise ValueError(f"Malformed English lemma/POS in {path}: {row.get('concept_id')!r}")
        row["concreteness"], row["concreteness_match"] = match_concreteness(
            row["en_lemma"], row["pos"], norms, nlp,
        )
    return table, rows


def run(config_path: str | Path) -> dict[str, Any]:
    """Hash the supplied norms file and resolve both step-11 tables in place."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["concreteness"]
    paths = settings["paths"]
    main_input, extension_input = Path(paths["main_input"]), Path(paths["extension_input"])
    missing = [path for path in (main_input, extension_input) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing_inputs_message(missing))
    raw_dir = Path(paths["raw_dir"])
    raw_files = sorted(path for path in raw_dir.iterdir() if path.is_file() and not path.name.startswith(".")) if raw_dir.is_dir() else []
    if not raw_files:
        raise FileNotFoundError(missing_norms_message(raw_dir))

    _hash_manual_source(Path(config_path).resolve())
    norm_candidates = [path for path in raw_files if path.suffix.casefold() in {".csv", ".xlsx", ".xlsm"}]
    if len(norm_candidates) != 1:
        raise ValueError(f"Expected exactly one Brysbaert CSV/XLSX in {raw_dir}; found {[str(path) for path in norm_candidates]!r}")
    norms_path = norm_candidates[0]
    norm_settings = settings["xlsx"] if norms_path.suffix.casefold() in {".xlsx", ".xlsm"} else settings["csv"]
    norms = load_norms(
        norms_path,
        word_column=norm_settings["word_column"],
        value_column=norm_settings["value_column"],
        bigram_column=norm_settings.get("bigram_column"),
        raters_column=norm_settings.get("raters_column"),
        logger=logger,
    )
    try:
        nlp = spacy.load(settings["spacy_model"])
    except OSError as exc:
        raise RuntimeError(f"Required spaCy model {settings['spacy_model']!r} is unavailable; install it before step 09b") from exc

    main_table, main_rows = _enrich_table(main_input, norms, nlp)
    extension_table, extension_rows = _enrich_table(extension_input, norms, nlp)
    output_paths = (Path(paths["main_output"]), Path(paths["extension_output"]))
    if output_paths != (main_input, extension_input):
        raise ValueError("Step 09b is configured to update the step-11 tables in place; input and output paths must match")
    _atomic_update_table(main_table, main_rows, output_paths[0])
    _atomic_update_table(extension_table, extension_rows, output_paths[1])
    for label, rows in (("main", main_rows), ("M1 extension", extension_rows)):
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["concreteness_match"]] = counts.get(row["concreteness_match"], 0) + 1
        logger.info("%s concreteness_match counts: %s", label, dict(sorted(counts.items())))
    runtime = time.monotonic() - started
    logger.info("Updated %d main + %d M1 extension rows using %d norms", len(main_rows), len(extension_rows), len(norms))
    logger.info("Runtime seconds: %.3f", runtime)
    return {"main_n": len(main_rows), "extension_n": len(extension_rows), "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        logger = logging.getLogger(STEP)
        message = f"Step 09b stopped: {type(exc).__name__}: {exc}"
        if logger.handlers:
            logger.error(message)
        else:
            print(f"ERROR {message}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
