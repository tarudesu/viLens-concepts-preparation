"""Measure tokenizer fertility on the FLORES+ devtest sentences."""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


STEP = "10b_fertility"
MODEL_KEYS = ("gemma", "qwen", "llama")


def missing_flores_message(paths: list[Path]) -> str:
    """Return the stable diagnostic for missing devtest language files."""
    return "Step 10b requires FLORES devtest input files: missing " + ", ".join(map(str, paths))


def sentence_counts(tokenizer: Any, sentence: str, *, language: str) -> tuple[int, int, int | None]:
    """Return no-special-token count, Unicode code points, and VI syllables."""
    if not isinstance(sentence, str) or not normalize_nfc(sentence).strip():
        raise ValueError(f"FLORES {language} sentence must be a non-empty string")
    normalized = normalize_nfc(sentence)
    token_ids = tokenizer.encode(normalized, add_special_tokens=False)
    if not isinstance(token_ids, (list, tuple)):
        raise ValueError("Tokenizer encode() must return a token ID sequence")
    syllables = len(normalized.split()) if language == "vi" else None
    return len(token_ids), len(normalized), syllables


def aggregate_fertility(sentences: list[str], tokenizer: Any, *, language: str) -> dict[str, int | float | None]:
    """Aggregate token/character and Vietnamese token/syllable ratios."""
    if not sentences:
        raise ValueError(f"FLORES devtest has no sentences for {language}")
    totals = [0, 0, 0]
    for sentence in sentences:
        tokens, characters, syllables = sentence_counts(tokenizer, sentence, language=language)
        totals[0] += tokens
        totals[1] += characters
        if syllables is not None:
            totals[2] += syllables
    if totals[1] == 0 or (language == "vi" and totals[2] == 0):
        raise ValueError(f"FLORES fertility denominator is zero for {language}")
    return {
        "n_sentences": len(sentences), "total_tokens": totals[0], "total_characters": totals[1],
        "tokens_per_character": totals[0] / totals[1],
        "total_syllables": totals[2] if language == "vi" else None,
        "tokens_per_syllable": totals[0] / totals[2] if language == "vi" else None,
    }


def _load_step10() -> Any:
    """Load the audited step-10 tokenizer-only loader."""
    module_path = Path(__file__).with_name("10_tokens.py")
    spec = importlib.util.spec_from_file_location("vilens_10_tokens_for_10b", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tokenizer helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_csv(records: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    """Write deterministic UTF-8 CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def run(config_path: str | Path) -> dict[str, Any]:
    """Tokenize each FLORES devtest language with each pinned study tokenizer."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["fertility"]
    flores_dir = Path(settings["paths"]["flores_dir"])
    direction_codes = config["direction_flores"]["codes"]
    languages = list(config["langs"])
    if set(languages) != set(direction_codes) or len(languages) != len(direction_codes):
        raise ValueError(f"Configured FLORES codes must match langs: {languages!r} vs {sorted(direction_codes)!r}")
    paths = {language: flores_dir / f"devtest_{direction_codes[language]}.parquet" for language in languages}
    missing = [paths[language] for language in languages if not paths[language].is_file()]
    if missing:
        raise FileNotFoundError(missing_flores_message(missing))
    if settings.get("character_unit") != "unicode_codepoints_including_spaces":
        raise ValueError("fertility.character_unit must be unicode_codepoints_including_spaces")
    sentences_by_language: dict[str, list[str]] = {}
    for language in languages:
        table = pq.read_table(paths[language])
        sentence_column = config["direction_flores"]["sentence_column"]
        if sentence_column not in table.column_names:
            raise ValueError(f"Unexpected FLORES fields in {paths[language]}: missing {sentence_column!r}; found {table.column_names!r}")
        sentences: list[str] = []
        for row_number, value in enumerate(table.column(sentence_column).to_pylist(), start=1):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Malformed FLORES sentence {row_number} in {paths[language]}: {value!r}")
            sentences.append(normalize_nfc(value))
        sentences_by_language[language] = sentences

    step10 = _load_step10()
    token_settings = config["tokens"]
    cache_dir = Path(token_settings["paths"]["tokenizer_cache"])
    output: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        model_settings = dict(config["models"][model_key])
        model_settings["use_fast"] = bool(token_settings["use_fast"])
        logger.info("Loading tokenizer only: %s@%s", model_settings["hf_id"], model_settings["revision"])
        tokenizer = step10._load_tokenizer(model_key, model_settings, cache_dir)
        for language in languages:
            metrics = aggregate_fertility(sentences_by_language[language], tokenizer, language=language)
            record = {
                "model": model_key, "hf_id": model_settings["hf_id"], "revision": model_settings["revision"],
                "lang": language, **metrics,
            }
            output.append(record)
            logger.info("%s | %s | sentences=%d tokens/char=%.8f tokens/syllable=%s",
                        model_key, language, metrics["n_sentences"], metrics["tokens_per_character"],
                        f"{metrics['tokens_per_syllable']:.8f}" if metrics["tokens_per_syllable"] is not None else "NA")
        del tokenizer
    fields = ["model", "hf_id", "revision", "lang", "n_sentences", "total_tokens", "total_characters",
              "tokens_per_character", "total_syllables", "tokens_per_syllable"]
    output_path = Path(settings["paths"]["output"])
    _atomic_csv(output, output_path, fields)
    runtime = time.monotonic() - started
    logger.info("Output: %s (%d model × language rows)", output_path, len(output))
    logger.info("Runtime seconds: %.3f", runtime)
    return {"rows": len(output), "output": str(output_path), "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        logger = logging.getLogger(STEP)
        message = f"Step 10b stopped: {type(exc).__name__}: {exc}"
        if logger.handlers:
            logger.error(message)
        else:
            print(f"ERROR {message}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
