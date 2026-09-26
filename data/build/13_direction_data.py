"""Create prompt-matched language-direction fit/validate data (without FLORES)."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from common import load_config, normalize_nfc, normalize_strings, setup_logging
except ModuleNotFoundError:
    from data.build.common import load_config, normalize_nfc, normalize_strings, setup_logging


STEP = "13_direction_data"
CANONICAL_COLUMNS = {"vi": "vi_canonical", "en": "en_lemma", "zh": "zh_canonical", "fr": "fr_canonical", "id": "id_canonical"}
SPLIT_ORDER = ("fit", "validate")


def assign_direction_splits(concept_ids: list[str], *, fit_fraction: float, seed: int) -> dict[str, str]:
    """Assign a stable seeded 80/20-style fit/validate split over concept IDs."""
    ordered_ids = sorted(normalize_nfc(value) for value in concept_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Direction split received duplicate concept IDs")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must be between zero and one, got {fit_fraction!r}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered_ids)).tolist()
    fit_n = int(round(len(ordered_ids) * fit_fraction))
    fit_ids = {ordered_ids[index] for index in order[:fit_n]}
    return {concept_id: ("fit" if concept_id in fit_ids else "validate") for concept_id in ordered_ids}


def render_direction_prompt(
    format_name: str, *, examples: list[dict[str, str]], test_form: str,
    label: str, ru_label: str, ru_test: str, k: int,
) -> str:
    """Render the step-12 repetition or Russian-translation template for language L."""
    if len(examples) != k:
        raise ValueError(f"Direction prompt requires exactly {k} examples; received {len(examples)}")
    if format_name == "repetition":
        lines = [f"{label}: {item['w']} - {label}: {item['w']}" for item in examples]
        lines.append(f"{label}: {test_form} - {label}:")
    elif format_name == "translation":
        lines = [f"{ru_label}: {item['ru']} - {label}: {item['w']}" for item in examples]
        lines.append(f"{ru_label}: {ru_test} - {label}:")
    else:
        raise ValueError(f"Unsupported direction prompt format: {format_name!r}")
    return "\n".join(lines)


def _load_step12() -> Any:
    """Load step-12's tested seeded few-shot selection helpers."""
    module_path = Path(__file__).resolve().with_name("12_prompts.py")
    spec = importlib.util.spec_from_file_location("vilens_12_prompts_for_13", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load step-12 selection helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Atomically write deterministic UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def run(config_path: str | Path) -> dict[str, Any]:
    """Build matched repetition/translation prompts for the directions split."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["direction_prompts"]
    input_path, output_path = Path(settings["paths"]["input"]), Path(settings["paths"]["output"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Step 13a requires step-12 input: {input_path}")
    table = pq.read_table(input_path)
    rows = [normalize_strings(row) for row in table.to_pylist()]
    if not rows:
        raise ValueError("Step-12 RU output contains no concepts")
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        concept_id = row.get("concept_id")
        if not isinstance(concept_id, str) or not concept_id or concept_id in seen_ids:
            raise ValueError(f"Step-12 row {row_number} has a missing or duplicate concept_id: {concept_id!r}")
        seen_ids.add(concept_id)

    language_order = list(settings["languages"])
    if set(language_order) != set(CANONICAL_COLUMNS) or len(language_order) != len(CANONICAL_COLUMNS):
        raise ValueError(f"direction_prompts.languages must be exactly {sorted(CANONICAL_COLUMNS)!r}")
    direction_rows = sorted((row for row in rows if row["split"] == "directions"), key=lambda row: row["concept_id"])
    expected_n = int(settings["expected_concepts"])
    if len(direction_rows) != expected_n:
        raise ValueError(f"Expected {expected_n} directions concepts, found {len(direction_rows)}")

    step12 = _load_step12()
    fewshot_settings = config["prompts"]["fewshot"]
    reservoir = [row for row in rows if row["split"] == "fewshot_reservoir"]
    eligible = [row for row in reservoir if row.get("ru_canonical") and row.get("ru_nllb_agree") is True and row.get("bt_pass") is True]
    n_selected, set_count = step12.fewshot_selection_size(
        len(eligible), requested_n=int(fewshot_settings["n_final"]),
        set_size=int(fewshot_settings["set_size"]), max_sets=int(fewshot_settings["n_sets"]),
        minimum_n=int(fewshot_settings["minimum_eligible_n"]),
    )
    selected = step12.select_balanced_fewshot(
        eligible, n_final=n_selected, set_count=set_count, set_size=int(fewshot_settings["set_size"]),
        seed=int(config["seed"]), allowed_strata=sorted({row["stratum"] for row in eligible}),
        allowed_pos=["adj", "noun", "verb"],
    )
    set_size = int(fewshot_settings["set_size"])
    primary_set = int(settings["fewshot_set"])
    if not 1 <= primary_set <= set_count:
        raise ValueError(f"Requested few-shot set {primary_set} is unavailable; selected sets={set_count}")
    fewshot = selected[(primary_set - 1) * set_size:primary_set * set_size]
    for row in fewshot:
        for language in language_order:
            form = row.get(CANONICAL_COLUMNS[language])
            if not isinstance(form, str) or not form.strip():
                raise ValueError(f"Few-shot set {primary_set} concept {row['concept_id']} lacks canonical {language}")
        if not isinstance(row.get("ru_canonical"), str) or not row["ru_canonical"].strip():
            raise ValueError(f"Few-shot set {primary_set} concept {row['concept_id']} lacks Russian canonical")

    split_by_id = assign_direction_splits(
        [row["concept_id"] for row in direction_rows], fit_fraction=float(settings["fit_fraction"]),
        seed=int(config["seed"]),
    )
    records: list[dict[str, Any]] = []
    skipped_by_language: dict[str, set[str]] = {language: set() for language in language_order}
    skip_reasons: dict[str, Counter[str]] = {language: Counter() for language in language_order}
    counts: Counter[tuple[str, str, str]] = Counter()
    prompt_examples: dict[str, list[dict[str, str]]] = {}
    formats = list(settings["formats"])
    if formats != ["repetition", "translation"]:
        raise ValueError(f"Unexpected configured direction formats: {formats!r}")
    ru_label = config["prompts"]["label"]["ru"]

    for language in language_order:
        canonical_field = CANONICAL_COLUMNS[language]
        examples = [
            {"w": row[canonical_field], "ru": row["ru_canonical"]}
            for row in fewshot
        ]
        prompt_examples[language] = examples
        for row in direction_rows:
            concept_id = row["concept_id"]
            form = row.get(canonical_field)
            split_name = split_by_id[concept_id]
            has_form = isinstance(form, str) and bool(form.strip())
            has_ru = isinstance(row.get("ru_canonical"), str) and bool(row["ru_canonical"].strip())
            for format_name in formats:
                if not has_form or (format_name == "translation" and not has_ru):
                    skipped_by_language[language].add(concept_id)
                    if not has_form:
                        skip_reasons[language]["missing_language_form"] += 1
                    if format_name == "translation" and not has_ru:
                        skip_reasons[language]["missing_ru_form"] += 1
                    continue
                prompt = render_direction_prompt(
                    format_name, examples=examples, test_form=form.strip(), label=settings["labels"][language],
                    ru_label=ru_label, ru_test=row["ru_canonical"] if format_name == "translation" else "",
                    k=int(config["prompts"][format_name]["k"]),
                )
                records.append({
                    "lang": language, "format": format_name, "split": split_name,
                    "concept_id": concept_id, "prompt": prompt, "target": " " + normalize_nfc(form).strip(),
                })
                counts[(language, format_name, split_name)] += 1
    records.sort(key=lambda row: (
        language_order.index(row["lang"]), formats.index(row["format"]), SPLIT_ORDER.index(row["split"]), row["concept_id"],
    ))
    _atomic_write_jsonl(records, output_path)

    logger.info("Direction concepts=%d; fit=%d; validate=%d; seed=%d", len(direction_rows),
                sum(value == "fit" for value in split_by_id.values()),
                sum(value == "validate" for value in split_by_id.values()), int(config["seed"]))
    logger.info("Few-shot source=set %d; eligible=%d; selected=%d; IDs=%s", primary_set, len(eligible), len(selected),
                json.dumps([row["concept_id"] for row in fewshot], ensure_ascii=False))
    logger.info("Matched direction prompt counts by lang × format × split:")
    for language in language_order:
        for format_name in formats:
            for split_name in SPLIT_ORDER:
                logger.info("%s | %s | %s | %d", language, format_name, split_name,
                            counts[(language, format_name, split_name)])
    logger.info("Skipped unique direction concepts by language and missing-form counts:")
    for language in language_order:
        logger.info("%s | skipped=%d | reasons=%s", language, len(skipped_by_language[language]),
                    json.dumps(dict(sorted(skip_reasons[language].items())), sort_keys=True))
    logger.info("One repetition example per language (first fit concept):")
    for language in language_order:
        example_record = next((row for row in records if row["lang"] == language and row["format"] == "repetition" and row["split"] == "fit"), None)
        if example_record is None:
            logger.info("%s | no available fit concept", language)
        else:
            logger.info("%s | %s | target=%s\n%s", language, example_record["concept_id"],
                        example_record["target"], example_record["prompt"])
    logger.info("Output: %s (%d records)", output_path, len(records))
    runtime = time.monotonic() - started
    logger.info("Runtime seconds: %.3f", runtime)
    return {"counts": counts, "skipped": {lang: len(ids) for lang, ids in skipped_by_language.items()},
            "records": len(records), "runtime_seconds": runtime, "examples": prompt_examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
