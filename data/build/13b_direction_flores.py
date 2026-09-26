"""Build fit/validate language-direction records from FLORES+ dev sentences."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import pyarrow.parquet as pq

try:
    from common import load_config, normalize_nfc, setup_logging
except ModuleNotFoundError:
    from data.build.common import load_config, normalize_nfc, setup_logging


STEP = "13b_direction_flores"


def missing_flores_message(paths: list[Path]) -> str:
    """Return the stable diagnostic for absent FLORES dev language files."""
    return "Step 13b requires FLORES dev input files: missing " + ", ".join(map(str, paths))


def sentence_id_key(value: Any) -> str:
    """Normalize a sentence ID to stable NFC text without conflating nulls."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"Malformed FLORES sentence id: {value!r}")
    key = normalize_nfc(str(value)).strip()
    if not key:
        raise ValueError("FLORES sentence id is empty")
    return key


def build_direction_records(
    datasets: dict[str, list[dict[str, Any]]], *, fit_fraction: float, seed: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Join language rows by sentence ID and assign one shared seeded split."""
    if not datasets:
        raise ValueError("No FLORES language datasets were supplied")
    ids_by_language: dict[str, set[str]] = {}
    normalized: dict[str, dict[str, str]] = {}
    for language in sorted(datasets):
        normalized[language] = {}
        for index, row in enumerate(datasets[language], start=1):
            sentence_id = sentence_id_key(row.get("sentence_id"))
            sentence = row.get("sentence")
            if not isinstance(sentence, str) or not normalize_nfc(sentence).strip():
                raise ValueError(f"FLORES {language} row {index} has no non-empty sentence")
            if sentence_id in normalized[language]:
                raise ValueError(f"Duplicate FLORES sentence id {sentence_id!r} for {language}")
            normalized[language][sentence_id] = normalize_nfc(sentence)
        ids_by_language[language] = set(normalized[language])
    reference_language = sorted(normalized)[0]
    reference_ids = ids_by_language[reference_language]
    for language in sorted(ids_by_language):
        if ids_by_language[language] != reference_ids:
            missing = sorted(reference_ids - ids_by_language[language])[:10]
            extra = sorted(ids_by_language[language] - reference_ids)[:10]
            raise ValueError(
                f"FLORES sentence IDs differ for {language} vs {reference_language}: "
                f"missing={missing!r}, extra={extra!r}"
            )
    step13 = _load_step13()
    splits = step13.assign_direction_splits(sorted(reference_ids), fit_fraction=fit_fraction, seed=seed)
    records: list[dict[str, str]] = []
    for language in sorted(normalized):
        for sentence_id in sorted(reference_ids):
            records.append({
                "lang": language, "sentence_id": sentence_id, "split": splits[sentence_id],
                "sentence": normalized[language][sentence_id],
            })
    return records, {split: sum(value == split for value in splits.values()) for split in ("fit", "validate")}


def _load_step13() -> Any:
    """Load the exact sentence/concept seeded split helper used by step 13a."""
    module_path = Path(__file__).with_name("13_direction_data.py")
    spec = importlib.util.spec_from_file_location("vilens_13_direction_for_13b", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load direction split helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    """Write sorted UTF-8 JSONL atomically with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def run(config_path: str | Path) -> dict[str, Any]:
    """Read five FLORES dev tables and write sentence-level fit/validate rows."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["direction_flores"]
    languages = list(settings["languages"])
    codes = settings["codes"]
    if len(languages) != len(set(languages)) or set(languages) != set(config["langs"]):
        raise ValueError(f"direction_flores.languages must match langs exactly: {languages!r}")
    if set(codes) != set(languages):
        raise ValueError(f"direction_flores.codes must cover configured languages: {sorted(languages)!r}")
    flores_dir = Path(settings["paths"]["flores_dir"])
    files = {language: flores_dir / f"dev_{codes[language]}.parquet" for language in languages}
    missing = [files[language] for language in languages if not files[language].is_file()]
    if missing:
        raise FileNotFoundError(missing_flores_message(missing))
    datasets: dict[str, list[dict[str, Any]]] = {}
    id_column, sentence_column = settings["id_column"], settings["sentence_column"]
    for language in languages:
        table = pq.read_table(files[language])
        missing_fields = sorted({id_column, sentence_column} - set(table.column_names))
        if missing_fields:
            raise ValueError(f"Unexpected FLORES fields in {files[language]}: missing {missing_fields!r}; found {table.column_names!r}")
        ids = table.column(id_column).to_pylist()
        sentences = table.column(sentence_column).to_pylist()
        datasets[language] = [
            {"sentence_id": sentence_id, "sentence": sentence}
            for sentence_id, sentence in zip(ids, sentences, strict=True)
        ]
    records, split_counts = build_direction_records(
        datasets, fit_fraction=float(settings["fit_fraction"]), seed=int(config["seed"]),
    )
    output_path = Path(settings["paths"]["output"])
    _atomic_write_jsonl(records, output_path)
    logger.info("FLORES dev sentence IDs=%d; seed=%d; fit=%d; validate=%d",
                len(records) // len(languages), int(config["seed"]), split_counts["fit"], split_counts["validate"])
    for language in languages:
        for split in ("fit", "validate"):
            count = sum(row["lang"] == language and row["split"] == split for row in records)
            logger.info("%s | %s | %d", language, split, count)
    logger.info("Output: %s (%d records; FLORES text is excluded from version control)", output_path, len(records))
    runtime = time.monotonic() - started
    logger.info("Runtime seconds: %.3f", runtime)
    return {"records": len(records), "sentence_ids": len(records) // len(languages), "split_counts": split_counts,
            "output": str(output_path), "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        logger = logging.getLogger(STEP)
        message = f"Step 13b stopped: {type(exc).__name__}: {exc}"
        if logger.handlers:
            logger.error(message)
        else:
            print(f"ERROR {message}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
