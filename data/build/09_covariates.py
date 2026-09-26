"""Add frequency, length, syllable, and pending concreteness covariates."""

from __future__ import annotations

import argparse
from collections import Counter
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wordfreq import zipf_frequency

try:
    from common import load_config, normalize_nfc, normalize_strings, setup_logging
except ModuleNotFoundError:
    from data.build.common import load_config, normalize_nfc, normalize_strings, setup_logging


STEP = "09_covariates"
CANONICAL_COLUMNS = {"vi": "vi_canonical", "en": "en_lemma", "zh": "zh_canonical", "fr": "fr_canonical", "id": "id_canonical"}
STRATA = ("sino", "nonsino", "ambiguous")


def n_syllables(value: str) -> int:
    """Count whitespace-separated Vietnamese syllables after NFC normalization."""
    if not isinstance(value, str):
        raise TypeError("n_syllables expects a string")
    return len(normalize_nfc(value).strip().split())


def n_characters(value: str) -> int:
    """Count Unicode code points in the NFC canonical string, including spaces."""
    if not isinstance(value, str):
        raise TypeError("n_characters expects a string")
    return len(normalize_nfc(value))


def standardized_mean_difference(left: list[float], right: list[float]) -> float | None:
    """Return pooled-SD standardized mean difference, or None if undefined."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return None
    pooled_variance = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2) if a.size + b.size > 2 else 0.0
    if pooled_variance <= 0 or not math.isfinite(float(pooled_variance)):
        return None
    return float((np.mean(a) - np.mean(b)) / math.sqrt(float(pooled_variance)))


def _validate_input(rows: list[dict[str, Any]], languages: list[str]) -> None:
    """Check required source columns and all split/stratum labels."""
    if not rows:
        raise ValueError("Step-08 input contains no concepts")
    required = {"concept_id", "split", "stratum", *[CANONICAL_COLUMNS[language] for language in languages]}
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        absent = sorted(required - row.keys())
        if absent:
            raise ValueError(f"Input row {index} is missing required columns: {absent!r}")
        if row["concept_id"] in ids:
            raise ValueError(f"Duplicate concept_id in step-08 input: {row['concept_id']!r}")
        ids.add(row["concept_id"])
        if row["split"] not in {"fewshot_reservoir", "directions", "test"}:
            raise ValueError(f"Unexpected split for {row['concept_id']!r}: {row['split']!r}")
        if row["stratum"] not in {"sino", "nonsino", "ambiguous", "other_loan"}:
            raise ValueError(f"Unexpected stratum for {row['concept_id']!r}: {row['stratum']!r}")
        for language in languages:
            field = CANONICAL_COLUMNS[language]
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Missing canonical {language} form for {row['concept_id']!r}: {row[field]!r}")


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


def _report(rows: list[dict[str, Any]], languages: list[str], logger: logging.Logger) -> dict[str, Any]:
    """Log means, sample SDs, and missing rates by split and stratum."""
    metrics = [f"zipf_{lang}" for lang in languages] + [f"{lang}_freq_missing" for lang in languages]
    metrics += ["n_syllables", *[f"n_chars_{lang}" for lang in languages], "concreteness"]
    report: dict[str, Any] = {}
    split_names = ("fewshot_reservoir", "directions", "test")
    for split in split_names:
        split_rows = [row for row in rows if row["split"] == split]
        for stratum in (*STRATA, "other_loan"):
            selected = [row for row in split_rows if row["stratum"] == stratum]
            if not selected:
                continue
            for name in metrics:
                values = [row[name] for row in selected if row[name] is not None]
                missing_rate = (len(selected) - len(values)) / len(selected)
                mean = float(np.mean(values)) if values else None
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
                report[f"{split}|{stratum}|{name}"] = {"n": len(selected), "mean": mean, "sd": sd, "missing_rate": missing_rate}
                logger.info("%s | %s | %s | n=%d mean=%s sd=%s missing_rate=%.6f", split, stratum, name, len(selected), f"{mean:.6f}" if mean is not None else "NA", f"{sd:.6f}" if sd is not None else "NA", missing_rate)
    match_counts = Counter(row["concreteness_match"] for row in rows)
    logger.info("concreteness_match counts: %s", dict(sorted(match_counts.items())))
    test_sino = [row for row in rows if row["split"] == "test" and row["stratum"] == "sino"]
    test_nonsino = [row for row in rows if row["split"] == "test" and row["stratum"] == "nonsino"]
    smd: dict[str, float | None] = {}
    for name in ("zipf_vi", "n_syllables", "concreteness"):
        left = [float(row[name]) for row in test_sino if row[name] is not None]
        right = [float(row[name]) for row in test_nonsino if row[name] is not None]
        smd[name] = standardized_mean_difference(left, right)
        logger.info("TEST standardized mean difference sino - nonsino | %s | %s", name, f"{smd[name]:.6f}" if smd[name] is not None else "NA")
    logger.info("concreteness intentionally pending: no norms file loaded or values imputed")
    return {"metrics": report, "concreteness_match_counts": dict(match_counts), "test_smd_sino_minus_nonsino": smd}


def run(config_path: str | Path) -> dict[str, Any]:
    """Run step 09, leaving concreteness explicitly pending."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["covariates"]
    languages = config["langs"]
    if len(languages) != len(set(languages)) or set(languages) != set(CANONICAL_COLUMNS):
        raise ValueError(f"langs must contain exactly {sorted(CANONICAL_COLUMNS)!r}, got {languages!r}")
    input_path, output_path = Path(settings["paths"]["input"]), Path(settings["paths"]["output"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Step-08 input not found: {input_path}")
    table = pq.read_table(input_path)
    rows = [normalize_strings(row) for row in table.to_pylist()]
    rows.sort(key=lambda row: normalize_nfc(row["concept_id"]))
    _validate_input(rows, languages)
    if "concreteness" in table.column_names or "concreteness_match" in table.column_names:
        raise ValueError("Step-08 input unexpectedly contains concreteness columns")
    pending = settings["pending_concreteness_label"]
    if pending != "pending":
        raise ValueError(f"Expected pending concreteness label 'pending', got {pending!r}")

    additions: list[pa.Field] = []
    for language in languages:
        additions.extend([pa.field(f"zipf_{language}", pa.float64(), nullable=False), pa.field(f"{language}_freq_missing", pa.bool_(), nullable=False)])
    additions.append(pa.field("n_syllables", pa.int32(), nullable=False))
    additions.extend(pa.field(f"n_chars_{language}", pa.int32(), nullable=False) for language in languages)
    additions.extend([pa.field("concreteness", pa.float64(), nullable=True), pa.field("concreteness_match", pa.string(), nullable=False)])
    collisions = sorted({field.name for field in additions}.intersection(table.column_names))
    if collisions:
        raise ValueError(f"Step-08 input unexpectedly contains covariate columns: {collisions!r}")

    for index, row in enumerate(rows, start=1):
        for language in languages:
            canonical = row[CANONICAL_COLUMNS[language]]
            value = float(zipf_frequency(canonical, language))
            row[f"zipf_{language}"] = value
            row[f"{language}_freq_missing"] = value == 0.0
            row[f"n_chars_{language}"] = n_characters(canonical)
        row["n_syllables"] = n_syllables(row["vi_canonical"])
        row["concreteness"] = None
        row["concreteness_match"] = pending
        progress_every = config["logging"]["progress_every"]
        if index % progress_every == 0 or index == len(rows):
            logger.info("Computed covariates: %d/%d concepts", index, len(rows))

    schema = pa.schema([*table.schema, *additions])
    output_table = pa.Table.from_pylist(rows, schema=schema)
    row_group_size = config["split"]["parquet_row_group_size"]
    _write_parquet(output_table, output_path, row_group_size)
    report = _report(rows, languages, logger)
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
            logger.error("Step 09 stopped: %s: %s", type(exc).__name__, exc)
        else:
            print(f"ERROR Step 09 stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
