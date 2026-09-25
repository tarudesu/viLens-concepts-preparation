"""Build sense-aligned four-language candidate groups from Wiktextract."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from opencc import OpenCC

try:  # Direct script execution (pipeline convention) and package-based tests.
    from common import DropflowLogger, iter_jsonl, load_config, normalize_nfc, normalize_strings, setup_logging
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, iter_jsonl, load_config, normalize_nfc, normalize_strings, setup_logging


STEP = "03_pool"
LANGUAGES = ("vi", "zh", "fr", "id")
_XREF_RE = re.compile(r"\b(?:see|translations)\b", re.IGNORECASE)


def has_cjk(text: str, ranges: Iterable[Iterable[int]]) -> bool:
    """Whether text contains a code point in one of the configured CJK ranges."""
    limits = tuple((int(start), int(end)) for start, end in ranges)
    return any(start <= ord(char) <= end for char in text for start, end in limits)


def has_latin_letter(text: str) -> bool:
    """Whether text contains a Unicode letter whose script name is Latin."""
    return any(
        unicodedata.category(char).startswith("L")
        and "LATIN" in unicodedata.name(char, "")
        for char in text
    )


def mandarin_candidates(word: str, split_on: str, converter: Any) -> list[str]:
    """Split Mandarin alternatives and prefer parts unchanged by t2s conversion."""
    if not split_on:
        raise ValueError("chinese.split_on must be a non-empty string")
    parts = list(dict.fromkeys(part.strip() for part in word.split(split_on) if part.strip()))
    simplified = [part for part in parts if converter.convert(part) == part]
    selected = simplified or [converter.convert(part) for part in parts]
    return list(dict.fromkeys(part for part in selected if part))


def _item_structure_rule(item: Any) -> str | None:
    """Return the first structural defect that makes a translation unusable."""
    if not isinstance(item, dict):
        return "non_object_item"
    code = item.get("lang_code")
    if not isinstance(code, str) or not code.strip():
        return "missing_lang_code" if code is None or code == "" else "invalid_lang_code"
    word = item.get("word")
    if not isinstance(word, str) or not word.strip():
        return "missing_word"
    sense = item.get("sense")
    if not isinstance(sense, str):
        return "missing_sense"
    for field in ("tags", "raw_tags"):
        values = item.get(field, [])
        if values is not None and (not isinstance(values, list) or any(not isinstance(value, str) for value in values)):
            return f"invalid_{field}"
    note = item.get("note")
    if note is not None and not isinstance(note, str):
        return "invalid_note"
    return None


def deduplicate_translation_items(items: Iterable[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    """Deduplicate by the agreed identity and merge top/senses provenance."""
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item, location in items:
        rule = _item_structure_rule(item)
        if rule:
            raise ValueError(f"Cannot deduplicate malformed translation item ({rule}): {item!r}")
        lang_code = normalize_nfc(item["lang_code"])
        word = normalize_nfc(item["word"]).strip()
        sense = normalize_nfc(item["sense"])
        key = (lang_code, word, sense)
        if key not in result:
            result[key] = {**item, "word": word, "_locations": {location}}
        else:
            result[key]["_locations"].add(location)
    return [
        {**item, "source_location": _location_label(item.pop("_locations"))}
        for _, item in sorted(result.items(), key=lambda pair: pair[0])
    ]


def _location_label(locations: set[str]) -> str:
    if locations == {"top", "senses"}:
        return "both"
    if len(locations) != 1 or next(iter(locations)) not in {"top", "senses"}:
        raise ValueError(f"Unexpected translation location set: {locations!r}")
    return next(iter(locations))


def clean_translation_item(item: dict[str, Any], drop_tags: set[str]) -> tuple[dict[str, Any] | None, str | None]:
    """Apply ordered item-level cleaning, returning its first matching rule."""
    structural_rule = _item_structure_rule(item)
    if structural_rule:
        return None, structural_rule
    word = item["word"].strip()
    cleaned = {**item, "word": word}
    note = item.get("note", "")
    if _XREF_RE.search(word) or (note and _XREF_RE.search(note)):
        return None, "cross_reference"
    tags = item.get("tags", [])
    raw_tags = item.get("raw_tags", [])
    if drop_tags.intersection(tags or []) or drop_tags.intersection(raw_tags or []):
        return None, "drop_tags"
    if any(char in word for char in "()…") or any(char.isdigit() for char in word):
        return None, "forbidden_characters"
    return cleaned, None


def count_item_drop(
    counts: dict[str, Counter[str]],
    examples: dict[tuple[str, str], list[dict[str, Any]]],
    rule: str,
    language_code: str,
    item: Any,
    lemma: str,
    location: str,
) -> None:
    """Increment a rule/language count and retain up to ten compact examples."""
    counts[rule][language_code] += 1
    saved_examples = examples[(rule, language_code)]
    if len(saved_examples) < 10:
        if isinstance(item, dict):
            sample = {key: item.get(key) for key in ("lang_code", "word", "sense", "note", "tags", "raw_tags") if key in item}
        else:
            sample = item
        saved_examples.append({"en_lemma": lemma, "location": location, "item": sample})


def join_vietnamese(word: str, pos: str, by_pair: dict[tuple[str, str], list[dict[str, Any]]], by_word: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], str | None]:
    """Prefer exact (word, POS), falling back to all entries for the word."""
    exact = by_pair.get((word, pos), [])
    if exact:
        return exact, "word_pos"
    fallback = by_word.get(word, [])
    return (fallback, "word_only") if fallback else ([], None)


def _unique_json_values(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    result = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _category_name(category: Any) -> str:
    if not isinstance(category, dict) or not isinstance(category.get("name"), str):
        raise ValueError(f"Unexpected category shape (expected object with name): {category!r}")
    return category["name"]


def _prepare_vi_entry(entry: dict[str, Any], ranges: list[list[int]]) -> dict[str, Any]:
    word, pos = entry.get("word"), entry.get("pos")
    if not isinstance(word, str) or not isinstance(pos, str):
        raise ValueError(f"Vietnamese entry missing string word/pos: {entry!r}")
    senses = entry.get("senses", [])
    if not isinstance(senses, list) or any(not isinstance(sense, dict) for sense in senses):
        raise ValueError(f"Unexpected Vietnamese senses shape for {word!r}")
    categories: list[Any] = []
    top_categories = entry.get("categories", [])
    if top_categories is not None:
        if not isinstance(top_categories, list):
            raise ValueError(f"Unexpected top-level categories for {word!r}")
        categories.extend(top_categories)
    for sense in senses:
        value = sense.get("categories", [])
        if value is not None:
            if not isinstance(value, list):
                raise ValueError(f"Unexpected sense categories for {word!r}")
            categories.extend(value)
    categories = _unique_json_values(categories)
    category_names = [_category_name(category) for category in categories]

    templates = entry.get("etymology_templates", [])
    if templates is None:
        templates = []
    if not isinstance(templates, list) or any(not isinstance(template, dict) for template in templates):
        raise ValueError(f"Unexpected etymology_templates for {word!r}")
    prepared_templates = []
    for template in templates:
        if not isinstance(template.get("name"), str):
            raise ValueError(f"Etymology template missing string name for {word!r}: {template!r}")
        args = template.get("args", {})
        prepared_templates.append({
            "name": template["name"],
            "args_json": json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        })
    etymology_text = entry.get("etymology_text")
    if etymology_text is not None and not isinstance(etymology_text, str):
        raise ValueError(f"Unexpected etymology_text for {word!r}")

    forms = entry.get("forms", [])
    if forms is None:
        forms = []
    if not isinstance(forms, list) or any(not isinstance(form, dict) for form in forms):
        raise ValueError(f"Unexpected forms for {word!r}")
    cjk_forms = []
    for form in forms:
        form_word = form.get("form")
        if not isinstance(form_word, str):
            raise ValueError(f"Vietnamese form missing string form for {word!r}: {form!r}")
        if has_cjk(form_word, ranges):
            cjk_forms.append({
                "form": form_word,
                "tags": form.get("tags") or [],
                "raw_tags": form.get("raw_tags") or [],
            })
    return {
        "word": word,
        "pos": pos,
        "n_senses": len(senses),
        "categories": category_names,
        "etymology_templates": prepared_templates,
        "etymology_text": etymology_text,
        "cjk_forms": cjk_forms,
    }


def _find_dump(raw_dir: Path, source: str) -> Path:
    source_dir = raw_dir / source
    candidates = sorted([*source_dir.glob("*.jsonl"), *source_dir.glob("*.jsonl.gz")])
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one JSONL dump in {source_dir}; found {len(candidates)}: {candidates}")
    return candidates[0]


def _iter_records(path: Path):
    if path.suffix != ".gz":
        yield from iter_jsonl(path)
        return
    with gzip.open(path, "rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield normalize_strings(json.loads(line.decode("utf-8")))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc


def _translation_lists(entry: dict[str, Any], locations: list[str]):
    items: list[tuple[dict[str, Any], str]] = []
    if "top" in locations and "translations" in entry:
        translations = entry["translations"]
        if not isinstance(translations, list):
            raise ValueError(f"Top-level translations must be a list for {entry.get('word')!r}")
        items.extend((item, "top") for item in translations)
    if "senses" in locations:
        senses = entry.get("senses", [])
        if not isinstance(senses, list) or any(not isinstance(sense, dict) for sense in senses):
            raise ValueError(f"Unexpected senses shape for {entry.get('word')!r}")
        for sense in senses:
            if "translations" in sense:
                translations = sense["translations"]
                if not isinstance(translations, list):
                    raise ValueError(f"Sense translations must be a list for {entry.get('word')!r}")
                items.extend((item, "senses") for item in translations)
    return items


def _clean_source_location(items: list[dict[str, Any]]) -> str:
    locations = {item["source_location"] for item in items}
    if "both" in locations or locations == {"top", "senses"}:
        return "both"
    if len(locations) == 1:
        return next(iter(locations))
    raise ValueError(f"Unexpected candidate locations {locations!r}")


def _without_step_dropflow(path: Path, step: str) -> None:
    """Atomically remove this step's previous records, preserving other steps."""
    if not path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with path.open("r", encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as target:
            temp_name = target.name
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed dropflow line {line_number} in {path}: {exc}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("step"), str):
                    raise ValueError(f"Unexpected dropflow record at line {line_number} in {path}")
                if record["step"] != step:
                    target.write(line if line.endswith("\n") else line + "\n")
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _schema() -> pa.Schema:
    string_list = pa.list_(pa.string())
    category_lists = pa.list_(string_list)
    template = pa.struct([("name", pa.string()), ("args_json", pa.string())])
    template_lists = pa.list_(pa.list_(template))
    cjk_form = pa.struct([("form", pa.string()), ("tags", string_list), ("raw_tags", string_list)])
    cjk_form_lists = pa.list_(pa.list_(cjk_form))
    vi_candidate = pa.struct([
        ("word", pa.string()), ("join_type", pa.string()), ("n_vi_entries", pa.int32()),
        ("vi_pos", string_list), ("n_senses_vi", pa.list_(pa.int32())),
        ("categories", category_lists), ("etymology_templates", template_lists),
        ("etymology_text", string_list), ("cjk_forms", cjk_form_lists),
        ("source_location", pa.string()),
    ])
    candidate = pa.struct([("word", pa.string()), ("source_location", pa.string())])
    return pa.schema([
        ("concept_id", pa.string()), ("en_lemma", pa.string()), ("pos", pa.string()),
        ("sense_gloss", pa.string()), ("vi_cands", pa.list_(vi_candidate)),
        ("zh_cands", pa.list_(candidate)), ("fr_cands", pa.list_(candidate)),
        ("id_cands", pa.list_(candidate)),
    ])


def _candidate_words(candidates: list[dict[str, Any]]) -> list[str]:
    return [candidate["word"] for candidate in candidates]


def _render_words(candidates: list[dict[str, Any]]) -> str:
    return ", ".join(_candidate_words(candidates))


def run(config_path: str) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(config_path)
    paths = config["paths"]
    logger = setup_logging(STEP, paths["logs"], level=config.get("logging", {}).get("level", "INFO"))
    raw_dir, interim_dir = Path(paths["raw"]), Path(paths["interim"])
    en_path, vi_path = _find_dump(raw_dir, "wiktextract_en"), _find_dump(raw_dir, "wiktextract_vi")
    pool = config["pool"]
    allowed_pos = set(pool["pos_keep"])
    max_words = pool["max_en_words"]
    drop_tags = set(pool["drop_tags"])
    locations = pool["translation_locations"]
    if not allowed_pos or any(pos not in {"noun", "verb", "adj"} for pos in allowed_pos):
        raise ValueError(f"Unexpected pool.pos_keep values: {sorted(allowed_pos)}")
    if set(locations) - {"top", "senses"} or not locations:
        raise ValueError(f"Unexpected pool.translation_locations: {locations!r}")
    zh_code = config["chinese"]["lang_code"]
    split_on = config["chinese"]["split_on"]
    prefer_simplified = config["chinese"]["prefer_simplified"]
    ranges = config["inspection"]["cjk_ranges"]
    converter = OpenCC("t2s")

    logger.info("Input English dump: %s", en_path)
    logger.info("Input Vietnamese dump: %s", vi_path)
    logger.info("Item cleaning counts are item-level; malformed records are skipped, never repaired.")
    clean_counts: dict[str, Counter[str]] = defaultdict(Counter)
    clean_examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    target_item_counts: Counter[str] = Counter()

    n_en_entries = n_eligible_entries = n_distinct_items = 0
    eligible_by_pos: Counter[str] = Counter()
    groups: dict[tuple[str, str, str], dict[str, dict[str, dict[str, Any]]]] = {}
    for entry in _iter_records(en_path):
        n_en_entries += 1
        if not isinstance(entry, dict):
            raise ValueError(f"English entry must be an object at record {n_en_entries}")
        word, pos = entry.get("word"), entry.get("pos")
        if not isinstance(word, str) or not isinstance(pos, str):
            raise ValueError(f"English entry missing string word/pos at record {n_en_entries}")
        if pos not in allowed_pos or len(word.split()) > max_words:
            continue
        n_eligible_entries += 1
        eligible_by_pos[pos] += 1
        relevant_items = []
        target_codes = {"vi", "fr", "id", zh_code}
        for translation_item, location in _translation_lists(entry, locations):
            if not isinstance(translation_item, dict):
                count_item_drop(clean_counts, clean_examples, "non_object_item", "<missing>", translation_item, word, location)
                continue
            item_code = translation_item.get("lang_code")
            if not isinstance(item_code, str) or not item_code.strip():
                rule = _item_structure_rule(translation_item) or "missing_lang_code"
                count_item_drop(clean_counts, clean_examples, rule, "<missing>", translation_item, word, location)
                continue
            language = {"vi": "vi", "fr": "fr", "id": "id", zh_code: "zh"}.get(item_code)
            structural_rule = _item_structure_rule(translation_item)
            if language is not None:
                target_item_counts[language] += 1
            if structural_rule:
                count_item_drop(clean_counts, clean_examples, structural_rule, item_code, translation_item, word, location)
                continue
            if item_code in target_codes:
                relevant_items.append((translation_item, location))
        items = deduplicate_translation_items(relevant_items)
        n_distinct_items += len(items)
        lemma = word.strip()
        for item in items:
            cleaned, rule = clean_translation_item(item, drop_tags)
            if rule:
                language_code = {"vi": "vi", "fr": "fr", "id": "id", zh_code: "zh"}.get(item["lang_code"], item["lang_code"])
                count_item_drop(clean_counts, clean_examples, rule, language_code, item, lemma, item["source_location"])
                continue
            lang_code = cleaned["lang_code"]
            language = {"vi": "vi", "fr": "fr", "id": "id", zh_code: "zh"}.get(lang_code)
            if language is None:
                continue
            candidate_words = [cleaned["word"]]
            if language == "zh":
                candidate_words = mandarin_candidates(cleaned["word"], split_on, converter) if prefer_simplified else list(dict.fromkeys(p.strip() for p in cleaned["word"].split(split_on) if p.strip()))
            elif language == "vi":
                if has_cjk(cleaned["word"], ranges) or not has_latin_letter(cleaned["word"]):
                    count_item_drop(clean_counts, clean_examples, "vietnamese_non_latin_or_cjk", language, item, lemma, item["source_location"])
                    continue
            key = (lemma, pos, cleaned["sense"])
            language_values = groups.setdefault(key, {lang: {} for lang in LANGUAGES})[language]
            for candidate_word in candidate_words:
                candidate_key = (candidate_word, cleaned["sense"])
                if candidate_key not in language_values:
                    language_values[candidate_key] = {"word": candidate_word, "locations": set()}
                language_values[candidate_key]["locations"].add(cleaned["source_location"])

    logger.info("English entries: %d; eligible POS/token entries: %d", n_en_entries, n_eligible_entries)
    logger.info("Distinct translation items after location deduplication: %d", n_distinct_items)
    for rule in sorted(clean_counts):
        logger.info("Item cleaning %s by lang_code: %s", rule, dict(sorted(clean_counts[rule].items())))
    logger.info("Target translation-item denominators: %s", dict(sorted(target_item_counts.items())))
    drop_rate_limit = pool["max_item_drop_rate"]
    if not isinstance(drop_rate_limit, (int, float)) or not 0 <= drop_rate_limit <= 1:
        raise ValueError(f"pool.max_item_drop_rate must be between 0 and 1: {drop_rate_limit!r}")
    exceeded = []
    for rule, counts in clean_counts.items():
        for language in LANGUAGES:
            denominator = target_item_counts[language]
            count = counts[language]
            if denominator and count / denominator > drop_rate_limit:
                exceeded.append((rule, language, count, denominator, clean_examples[(rule, language)]))
    if exceeded:
        for rule, language, count, denominator, examples in exceeded:
            logger.error("SAFETY STOP: item rule %s removed %d/%d (%0.3f) for %s; examples=%s",
                         rule, count, denominator, count / denominator, language,
                         json.dumps(examples, ensure_ascii=False, sort_keys=True))
        raise ValueError("Item cleaning exceeded pool.max_item_drop_rate; no parquet or dropflow was written")

    vi_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    vi_by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n_vi_entries = 0
    for entry in _iter_records(vi_path):
        if not isinstance(entry, dict):
            raise ValueError(f"Vietnamese entry must be an object at record {n_vi_entries + 1}")
        prepared = _prepare_vi_entry(entry, ranges)
        vi_by_pair[(prepared["word"], prepared["pos"])].append(prepared)
        vi_by_word[prepared["word"]].append(prepared)
        n_vi_entries += 1
    logger.info("Vietnamese entries indexed: %d", n_vi_entries)

    eligible_groups = len(groups)
    eligible_groups_by_pos = Counter(key[1] for key in groups)
    five_way = {
        key: language_values for key, language_values in groups.items()
        if all(language_values[language] for language in LANGUAGES)
    }
    five_way_by_pos = Counter(key[1] for key in five_way)
    rows: list[dict[str, Any]] = []
    join_types: Counter[str] = Counter()
    vi_entry_counts: Counter[int] = Counter()
    final_by_pos: Counter[str] = Counter()
    dropped_missing_vi = 0
    for (lemma, pos, sense), language_values in sorted(five_way.items()):
        vi_candidates = []
        for candidate_key in sorted(language_values["vi"]):
            candidate = language_values["vi"][candidate_key]
            matches, join_type = join_vietnamese(candidate["word"], pos, vi_by_pair, vi_by_word)
            if not matches:
                continue
            assert join_type is not None
            join_types[join_type] += 1
            vi_entry_counts[len(matches)] += 1
            vi_candidates.append({
                "word": candidate["word"],
                "join_type": join_type,
                "n_vi_entries": len(matches),
                "vi_pos": [match["pos"] for match in matches],
                "n_senses_vi": [match["n_senses"] for match in matches],
                "categories": [match["categories"] for match in matches],
                "etymology_templates": [match["etymology_templates"] for match in matches],
                "etymology_text": [match["etymology_text"] for match in matches],
                "cjk_forms": [match["cjk_forms"] for match in matches],
                "source_location": _location_label(candidate["locations"]),
            })
        if not vi_candidates:
            dropped_missing_vi += 1
            continue
        lang_columns: dict[str, list[dict[str, str]]] = {}
        for language in ("zh", "fr", "id"):
            lang_columns[f"{language}_cands"] = [
                {"word": value["word"], "source_location": _location_label(value["locations"])}
                for _, value in sorted(language_values[language].items())
            ]
        concept_id = hashlib.sha1(f"{lemma}|{pos}|{sense}".encode("utf-8")).hexdigest()[:12]
        rows.append({
            "concept_id": concept_id,
            "en_lemma": lemma,
            "pos": pos,
            "sense_gloss": sense,
            "vi_cands": vi_candidates,
            **lang_columns,
        })
        final_by_pos[pos] += 1

    dropflow_path = Path(paths["dropflow"])
    _without_step_dropflow(dropflow_path, STEP)
    dropflow = DropflowLogger(dropflow_path)
    dropflow.record(step=STEP, stage="raw_groups_pos_filtered", n_in=n_en_entries,
                    n_out=n_eligible_entries, n_out_by_pos=dict(sorted(eligible_by_pos.items())))
    dropflow.record(step=STEP, stage="five_way", n_in=eligible_groups,
                    n_out=len(five_way), n_out_by_pos=dict(sorted(five_way_by_pos.items())))
    dropflow.record(step=STEP, stage="filter6_missing_vi_entry", n_in=len(five_way),
                    n_out=len(rows), n_out_by_pos=dict(sorted(final_by_pos.items())))

    schema = _schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / "03_pool.parquet"
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=interim_dir, prefix=".03_pool.", suffix=".parquet", delete=False) as temp:
            temp_name = temp.name
        pq.write_table(table, temp_name, compression="zstd", version="2.6", row_group_size=65536, write_statistics=True)
        os.replace(temp_name, output_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    logger.info("Dropflow raw_groups_pos_filtered: n_in=%d n_out=%d by_pos=%s", n_en_entries, n_eligible_entries, dict(sorted(eligible_by_pos.items())))
    logger.info("Raw sense groups: %d; five-way groups: %d; groups missing VI dictionary entry: %d; output rows: %d", eligible_groups, len(five_way), dropped_missing_vi, len(rows))
    logger.info("Dropflow five_way: n_in=%d n_out=%d by_pos=%s", eligible_groups, len(five_way), dict(sorted(five_way_by_pos.items())))
    logger.info("Dropflow filter6_missing_vi_entry: n_in=%d n_out=%d by_pos=%s", len(five_way), len(rows), dict(sorted(final_by_pos.items())))
    logger.info("Vietnamese join_type distribution: %s", dict(sorted(join_types.items())))
    logger.info("Vietnamese n_vi_entries distribution: %s", dict(sorted(vi_entry_counts.items())))
    for language in LANGUAGES:
        column = f"{language}_cands"
        counts = Counter(min(len(row[column]), 5) for row in rows)
        logger.info("Candidate lengths %s (1,2,3,4,5+): %s", language, [counts[n] for n in range(1, 6)])
    form_concepts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for candidate in row["vi_cands"]:
            form_concepts[candidate["word"]].add(row["concept_id"])
    repeated_forms = [(word, len(concepts)) for word, concepts in form_concepts.items() if len(concepts) > 1]
    repeated_forms.sort(key=lambda item: (-item[1], item[0]))
    logger.info("Vietnamese forms in >1 concept: %d", len(repeated_forms))
    logger.info("Top repeated Vietnamese forms: %s", repeated_forms[:20])
    rng = np.random.default_rng(config["seed"])
    sample_n = min(20, len(rows))
    sample_indices = sorted(rng.choice(len(rows), size=sample_n, replace=False).tolist()) if sample_n else []
    logger.info("Seeded sample rows (%d):", sample_n)
    for index in sample_indices:
        row = rows[index]
        logger.info("%s | %s | %s | %s | %s | %s | %s", row["en_lemma"], row["pos"], row["sense_gloss"],
                    _render_words(row["vi_cands"]), _render_words(row["zh_cands"]),
                    _render_words(row["fr_cands"]), _render_words(row["id_cands"]))
    elapsed = time.monotonic() - started
    logger.info("Output: %s (%d rows)", output_path, len(rows))
    logger.info("Runtime seconds: %.3f", elapsed)
    return {"output": output_path, "rows": len(rows), "runtime_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
