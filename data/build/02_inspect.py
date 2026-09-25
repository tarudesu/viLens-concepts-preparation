"""Read-only schema diagnostics for the local source dumps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from common import DataConfigError, iter_jsonl, load_config, normalize_nfc, setup_logging


class DiagnosticError(RuntimeError):
    """Raised when a local source does not match the task's expected schema."""


def contains_cjk(value: Any, ranges: list[list[int]]) -> bool:
    """Return whether any string recursively contains a configured CJK codepoint."""

    if isinstance(value, str):
        return any(start <= ord(char) <= end for char in value for start, end in ranges)
    if isinstance(value, dict):
        return any(contains_cjk(item, ranges) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_cjk(item, ranges) for item in value)
    return False


def translation_matches(item: dict[str, Any], target: str, aliases: dict[str, list[str]]) -> bool:
    """Match a translation using observed language name/code fields and aliases."""

    accepted = {normalize_nfc(alias).casefold() for alias in aliases[target]}
    lang = item.get("lang")
    if isinstance(lang, str):
        folded = normalize_nfc(lang).casefold()
        if target == "zh" and any(term in folded for term in ("chinese", "mandarin")):
            return True
        if folded in accepted:
            return True
    return any(
        isinstance(item.get(field), str)
        and normalize_nfc(item[field]).casefold() in accepted
        for field in ("code", "lang_code")
    )


def translation_buckets(translations: list[dict[str, Any]], aliases: dict[str, list[str]]) -> dict[str, bool]:
    """Return whether one entry's translations attest each requested language."""

    return {
        language: any(translation_matches(item, language, aliases) for item in translations)
        for language in ("vi", "zh", "fr", "id")
    }


def sense_related_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Keep the translation fields explicitly related to sense annotation."""

    return {key: item[key] for key in sorted(item) if "sense" in key.casefold()}


def _strings(value: Any) -> list[str]:
    """Collect NFC strings recursively for field-location diagnostics."""

    if isinstance(value, str):
        return [normalize_nfc(value)]
    if isinstance(value, dict):
        return [text for key in sorted(value) for text in _strings(value[key])]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return []


def _require_list(record: dict[str, Any], key: str, file_label: str, line_number: int) -> list[Any]:
    value = record.get(key, [])
    if not isinstance(value, list):
        raise DiagnosticError(f"{file_label}:{line_number}: expected {key!r} to be a list; found {type(value).__name__}")
    return value


def _english_dump(path: Path, settings: dict[str, Any], logger) -> None:
    aliases = settings["language_aliases"]
    required_n = settings["english_vietnamese_n"]
    examples: list[tuple[int, dict[str, Any]]] = []
    keys: set[str] = set()
    chinese_combinations: Counter[tuple[str, str, str, str]] = Counter()
    entries_total = 0
    entries_with_translations = 0
    entries_with_vi = 0
    complete_groups: set[tuple[str, str, str]] = set()
    found_languages: set[str] = set()

    for line_number, entry in enumerate(iter_jsonl(path), start=1):
        entries_total += 1
        if not isinstance(entry, dict):
            raise DiagnosticError(f"English dump line {line_number}: expected an object, found {type(entry).__name__}")
        translations = _require_list(entry, "translations", "English dump", line_number)
        if translations:
            entries_with_translations += 1
        if not translations:
            continue
        for item in translations:
            if not isinstance(item, dict):
                raise DiagnosticError(f"English dump line {line_number}: translation item must be an object, found {type(item).__name__}")
            keys.update(str(key) for key in item)
            for language in ("vi", "zh", "fr", "id"):
                if translation_matches(item, language, aliases):
                    found_languages.add(language)
            lang = item.get("lang")
            if isinstance(lang, str) and any(term in lang.casefold() for term in ("chinese", "mandarin")):
                code = item.get("code", "<missing>")
                lang_code = item.get("lang_code", "<missing>")
                tags = item.get("tags", "<missing>")
                chinese_combinations[(lang, str(code), str(lang_code), json.dumps(tags, ensure_ascii=False, sort_keys=True))] += 1

        has_vi = any(translation_matches(item, "vi", aliases) for item in translations)
        if has_vi:
            entries_with_vi += 1
            if len(examples) < required_n:
                examples.append((line_number, entry))

        word, pos = entry.get("word"), entry.get("pos")
        if not isinstance(word, str) or not isinstance(pos, str):
            raise DiagnosticError(f"English dump line {line_number}: expected string word and pos fields")
        languages_by_sense: dict[str, set[str]] = defaultdict(set)
        for item in translations:
            sense = item.get("sense", "")
            if sense is None:
                sense = ""
            if not isinstance(sense, str):
                raise DiagnosticError(f"English dump line {line_number}: translation sense must be a string or null")
            for language in ("vi", "zh", "fr", "id"):
                if translation_matches(item, language, aliases):
                    languages_by_sense[normalize_nfc(sense)].add(language)
        for sense, languages in languages_by_sense.items():
            if {"vi", "zh", "fr", "id"}.issubset(languages):
                complete_groups.add((normalize_nfc(word), normalize_nfc(pos), sense))

    if entries_total == 0:
        raise DiagnosticError(f"English dump is empty: {path}")
    if len(examples) != required_n:
        raise DiagnosticError(f"English dump contains only {len(examples)} entries with Vietnamese translations; expected {required_n}")
    missing = sorted({"vi", "zh", "fr", "id"} - found_languages)
    if missing:
        raise DiagnosticError(f"English dump is missing translations for configured languages: {missing}")

    logger.info("A. English dump: %s entries; selected first %s entries containing Vietnamese (line numbers: %s)", entries_total, required_n, [line for line, _ in examples])
    logger.info("A. Union of translation dictionary keys across selected entries: %s", json.dumps(sorted(keys), ensure_ascii=False))
    logger.info("A. Chinese/Mandarin lang/code/lang_code/tags combinations and occurrence counts:")
    for combo, count in sorted(chinese_combinations.items()):
        logger.info("  %s", json.dumps({"lang": combo[0], "code": combo[1], "lang_code": combo[2], "tags": json.loads(combo[3]), "count": count}, ensure_ascii=False, sort_keys=True))

    relevant: list[tuple[dict[str, Any], dict[str, Any]]] = []
    sense_strings: list[tuple[dict[str, Any], str]] = []
    sense_field_present = False
    index_or_id_present = False
    for _, entry in examples:
        for item in entry["translations"]:
            if any(translation_matches(item, language, aliases) for language in ("vi", "zh", "fr", "id")):
                relevant.append((entry, item))
                fields = sense_related_fields(item)
                sense_field_present |= "sense" in fields and isinstance(fields["sense"], str)
                index_or_id_present |= any("index" in key.casefold() or "id" in key.casefold() for key in fields if key.casefold() != "sense")
                value = item.get("sense")
                if isinstance(value, str) and value:
                    sense_strings.append((entry, normalize_nfc(value)))
    empty_count = sum(not isinstance(item.get("sense"), str) or not item.get("sense", "").strip() for _, item in relevant)
    glosses_by_entry: dict[int, set[str]] = {}
    for _, entry in examples:
        senses = _require_list(entry, "senses", "English dump sample", 0)
        glosses_by_entry[id(entry)] = {
            normalize_nfc(gloss)
            for sense in senses if isinstance(sense, dict)
            for gloss in sense.get("glosses", []) if isinstance(gloss, str)
        }
    exact_matches = sum(sense in glosses_by_entry[id(entry)] for entry, sense in sense_strings)
    entry_exact_matches = sum(any(entry is candidate and sense in glosses_by_entry[id(entry)] for candidate, sense in sense_strings) for _, entry in examples)
    sense_status = "sense string" if sense_field_present else "no sense string"
    sense_status += "; sense index/id present" if index_or_id_present else "; no sense index/id observed"
    logger.info("A. Translation sense annotation: %s", sense_status)
    logger.info("A. Empty/missing sense share for selected vi/zh/fr/id translations: %s/%s = %.4f", empty_count, len(relevant), empty_count / len(relevant) if relevant else 0.0)
    logger.info("A. Exact translation-sense-to-entry-gloss match: %s/%s non-empty sense strings = %.4f; entries with >=1 exact match: %s/%s", exact_matches, len(sense_strings), exact_matches / len(sense_strings) if sense_strings else 0.0, entry_exact_matches, len(examples))

    logger.info("A. Five full target-language translation lists with sense-related fields:")
    printed = 0
    for line_number, entry in examples:
        selected = [
            {**{key: value for key, value in item.items() if not ("sense" in key.casefold())}, **sense_related_fields(item)}
            for item in entry["translations"]
            if any(translation_matches(item, language, aliases) for language in ("vi", "zh", "fr", "id"))
        ]
        if not selected:
            continue
        logger.info("A. entry line=%s word=%r pos=%r translations=%s", line_number, entry.get("word"), entry.get("pos"), json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True))
        printed += 1
        if printed >= settings["english_pretty_n"]:
            break

    logger.info("A. Whole English dump: entries with any translations=%s; entries with Vietnamese translations=%s; distinct (word,pos,sense) groups with vi+Chinese+fr+id=%s", entries_with_translations, entries_with_vi, len(complete_groups))


def _vietnamese_dump(path: Path, settings: dict[str, Any], seed: int, logger) -> None:
    sample_n = settings["vietnamese_sample_n"]
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[int, dict[str, Any]]] = []
    total = 0
    top_keys: set[str] = set()
    category_sino = 0
    template_sino = 0
    cjk_form_entries = 0
    proto_text_entries = 0
    other_field_cjk: Counter[str] = Counter()
    cjk_ranges = settings["cjk_ranges"]

    for line_number, entry in enumerate(iter_jsonl(path), start=1):
        total += 1
        if not isinstance(entry, dict):
            raise DiagnosticError(f"Vietnamese dump line {line_number}: expected an object, found {type(entry).__name__}")
        top_keys.update(str(key) for key in entry)
        forms = _require_list(entry, "forms", "Vietnamese dump", line_number)
        categories = _require_list(entry, "categories", "Vietnamese dump", line_number)
        templates = _require_list(entry, "etymology_templates", "Vietnamese dump", line_number)
        senses = _require_list(entry, "senses", "Vietnamese dump", line_number)
        category_names = [item.get("name", "") for item in categories if isinstance(item, dict)]
        if any("sino-vietnamese" in name.casefold() for name in category_names if isinstance(name, str)):
            category_sino += 1
        if any(isinstance(item, dict) and item.get("name") == "vi-etym-sino" for item in templates):
            template_sino += 1
        if any(contains_cjk(form, cjk_ranges) for form in forms):
            cjk_form_entries += 1
        etymology = entry.get("etymology_text", "")
        if etymology is not None and not isinstance(etymology, str):
            raise DiagnosticError(f"Vietnamese dump line {line_number}: etymology_text must be a string or null")
        if isinstance(etymology, str) and any(phrase in etymology.casefold() for phrase in ("proto-vietic", "proto-mon-khmer")):
            proto_text_entries += 1
        head_templates = _require_list(entry, "head_templates", "Vietnamese dump", line_number)
        for field, value in (("head_templates.args", [item.get("args", {}) for item in head_templates if isinstance(item, dict)]), ("etymology_templates.args", [item.get("args", {}) for item in templates]), ("senses", senses)):
            if contains_cjk(value, cjk_ranges):
                other_field_cjk[field] += 1

        item = (line_number, entry)
        if len(reservoir) < sample_n:
            reservoir.append(item)
        else:
            replace_at = int(rng.integers(0, total))
            if replace_at < sample_n:
                reservoir[replace_at] = item

    if total == 0:
        raise DiagnosticError(f"Vietnamese dump is empty: {path}")
    if len(reservoir) != sample_n:
        raise DiagnosticError(f"Vietnamese dump contains only {len(reservoir)} entries; expected seeded sample of {sample_n}")

    logger.info("B. Vietnamese dump: entries=%s; seeded reservoir sample=%s using seed=%s; top-level key union=%s", total, sample_n, seed, json.dumps(sorted(top_keys), ensure_ascii=False))
    logger.info("B. Whole Vietnamese dump counts: top-level categories containing 'Sino-Vietnamese'=%s; etymology_templates with vi-etym-sino=%s; entries with any CJK form=%s; etymology_text mentioning Proto-Vietic or Proto-Mon-Khmer=%s", category_sino, template_sino, cjk_form_entries, proto_text_entries)
    if cjk_form_entries == 0:
        logger.info("B. No CJK in forms; CJK-bearing entry counts in alternate fields: %s", json.dumps(dict(sorted(other_field_cjk.items())), ensure_ascii=False, sort_keys=True))
    else:
        logger.info("B. CJK is present in forms; alternate-field CJK entry counts (also scanned): %s", json.dumps(dict(sorted(other_field_cjk.items())), ensure_ascii=False, sort_keys=True))

    for line_number, entry in sorted(reservoir)[:settings["vietnamese_print_n"]]:
        categories = _require_list(entry, "categories", "Vietnamese sample", line_number)
        templates = _require_list(entry, "etymology_templates", "Vietnamese sample", line_number)
        senses = _require_list(entry, "senses", "Vietnamese sample", line_number)
        forms = _require_list(entry, "forms", "Vietnamese sample", line_number)
        cjk_forms = [
            {"form": form.get("form"), "tags": form.get("tags", [])}
            for form in forms
            if isinstance(form, dict) and contains_cjk(form.get("form", ""), cjk_ranges)
        ]
        summary = {
            "line": line_number,
            "word": entry.get("word"),
            "pos": entry.get("pos"),
            "number_of_senses": len(senses),
            "category_names": [item.get("name") for item in categories if isinstance(item, dict)],
            "etymology_template_names": [item.get("name") for item in templates if isinstance(item, dict)],
            "cjk_forms_and_tags": cjk_forms,
        }
        logger.info("B. sampled entry: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _unihan(path: Path, chunk_lines: int, logger) -> None:
    files = sorted(path.glob("*.txt"))
    if not files:
        raise DiagnosticError(f"No extracted Unihan text files found in {path}")
    k_vietnamese_file: str | None = None
    lines: list[str] = []
    characters: set[str] = set()
    syllables: set[str] = set()
    other_fields: set[str] = set()
    for file_path in files:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = normalize_nfc(raw_line.rstrip("\r\n"))
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    raise DiagnosticError(f"Malformed Unihan row at {file_path}:{line_number}: expected at least 3 tab-separated fields")
                codepoint, field, value = parts[0], parts[1], "\t".join(parts[2:])
                if "viet" in field.casefold() or "hanviet" in field.casefold():
                    other_fields.add(field)
                if field == "kVietnamese":
                    if k_vietnamese_file is not None and k_vietnamese_file != file_path.name:
                        raise DiagnosticError(f"kVietnamese occurs in multiple files: {k_vietnamese_file}, {file_path.name}")
                    k_vietnamese_file = file_path.name
                    characters.add(codepoint)
                    syllables.update(normalize_nfc(token) for token in value.split() if token)
                    if len(lines) < chunk_lines:
                        lines.append(line)
    if k_vietnamese_file is None:
        logger.error("C. kVietnamese is MISSING from Unihan Unicode 18.0 extracted files.")
        raise DiagnosticError("kVietnamese is MISSING from Unihan Unicode 18.0 extracted files")
    logger.info("C. Unihan kVietnamese exists in %s", k_vietnamese_file)
    for line in lines:
        logger.info("C. kVietnamese row: %s", line)
    logger.info("C. kVietnamese total: characters=%s; distinct whitespace-delimited syllables=%s", len(characters), len(syllables))
    logger.info("C. Unihan field names containing Viet or HanViet: %s", json.dumps(sorted(other_fields), ensure_ascii=False))


def _cedict(path: Path, logger) -> None:
    header: list[str] = []
    entries = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        in_header = True
        for line_number, raw_line in enumerate(handle, start=1):
            line = normalize_nfc(raw_line.rstrip("\r\n"))
            if in_header and line.startswith("#"):
                header.append(line)
            else:
                in_header = False
                if line.strip():
                    entries += 1
    declared = next((line.partition("=")[2] for line in header if line.startswith("#! entries=")), None)
    if declared is not None and declared.isdigit() and int(declared) != entries:
        raise DiagnosticError(f"CC-CEDICT header declares {declared} entries but scan found {entries}")
    logger.info("D. CC-CEDICT header lines:")
    for line in header:
        logger.info("  %s", line)
    logger.info("D. CC-CEDICT total entries=%s", entries)


def run(config_path: Path) -> int:
    started = time.perf_counter()
    config = load_config(config_path)
    repo_root = config_path.resolve().parents[1]
    paths = config["paths"]

    def repo_path(key: str) -> Path:
        value = Path(paths[key])
        return value if value.is_absolute() else repo_root / value

    logger = setup_logging(Path(__file__).stem, repo_path("logs"), level=config["logging"]["level"])
    raw = repo_path("raw")
    settings = config["inspection"]
    try:
        _english_dump(raw / "wiktextract_en" / "kaikki.org-dictionary-English.jsonl", settings, logger)
        _vietnamese_dump(raw / "wiktextract_vi" / "kaikki.org-dictionary-Vietnamese.jsonl", settings, config["seed"], logger)
        _unihan(raw / "unihan" / "extracted", settings["unihan_print_n"], logger)
        _cedict(raw / "cedict" / "cedict_1_0_ts_utf-8_mdbg.txt.gz", logger)
    except (OSError, ValueError, KeyError, DiagnosticError) as exc:
        logger.error("STOP: %s", exc)
        return 2
    logger.info("Step 02 completed in %.2f seconds (streaming diagnostics; no source data written).", time.perf_counter() - started)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.config)
    except (DataConfigError, KeyError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
