"""Read-only schema diagnostics for the local source dumps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterator

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


def _walk_named_lists(
    value: Any,
    name: str,
    path: str = "",
    sense_context: dict[str, Any] | None = None,
) -> Iterator[tuple[str, list[Any], dict[str, Any] | None]]:
    """Yield recursively nested lists named *name* with stable array paths."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            context = sense_context
            if key == "senses" and isinstance(child, list):
                for sense in child:
                    if isinstance(sense, dict):
                        yield from _walk_named_lists(sense, name, f"{child_path}[]", sense)
                    else:
                        yield from _walk_named_lists(sense, name, f"{child_path}[]", sense_context)
                continue
            if key == name:
                if not isinstance(child, list):
                    raise DiagnosticError(f"Field {child_path!r} must be a list, found {type(child).__name__}")
                yield child_path, child, context
            if isinstance(child, list):
                for item in child:
                    yield from _walk_named_lists(item, name, f"{child_path}[]", context)
            elif isinstance(child, dict):
                yield from _walk_named_lists(child, name, child_path, context)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_named_lists(item, name, f"{path}[]", sense_context)


def _is_mandarin(item: dict[str, Any], settings: dict[str, Any]) -> tuple[bool, bool]:
    """Return (rule match, matched the lang-name-plus-tag branch)."""

    lang_code = item.get("lang_code")
    branch_code = lang_code == settings["mandarin_code"]
    lang = item.get("lang")
    tags: list[str] = []
    for field in ("tags", "raw_tags"):
        value = item.get(field, [])
        if value is None:
            continue
        if isinstance(value, str):
            tags.append(normalize_nfc(value))
        elif isinstance(value, list):
            if any(not isinstance(tag, str) for tag in value):
                raise DiagnosticError(f"Translation {field} values must be strings: {value!r}")
            tags.extend(normalize_nfc(tag) for tag in value)
        else:
            raise DiagnosticError(f"Translation {field} must be a string, list, or null; found {type(value).__name__}")
    exact_names = {normalize_nfc(name).casefold() for name in settings["mandarin_exact_lang_names"]}
    branch_name = isinstance(lang, str) and normalize_nfc(lang).casefold() in exact_names and settings["mandarin_required_tag"].casefold() in {tag.casefold() for tag in tags}
    return branch_code or branch_name, branch_name


def _language_item_matches(item: dict[str, Any], target: str, aliases: dict[str, list[str]], mandarin_settings: dict[str, Any]) -> bool:
    """Apply configured aliases for vi/fr/id and the explicit Mandarin rule."""

    if target == "zh":
        return _is_mandarin(item, mandarin_settings)[0]
    return translation_matches(item, target, aliases)


def _translation_sense(item: dict[str, Any], context: dict[str, Any] | None) -> str:
    """Choose the available translation sense, falling back to its enclosing sense."""

    sense = item.get("sense")
    if isinstance(sense, str) and sense:
        return normalize_nfc(sense)
    if context is None:
        return ""
    glosses = context.get("glosses")
    if isinstance(glosses, list) and all(isinstance(gloss, str) for gloss in glosses):
        return "\u241f".join(normalize_nfc(gloss) for gloss in glosses)
    sense_id = context.get("id")
    return normalize_nfc(sense_id) if isinstance(sense_id, str) else ""


def _tag_list(item: dict[str, Any], field: str) -> list[str]:
    """Validate and normalize optional tag arrays for form/item diagnostics."""

    value = item.get(field, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [normalize_nfc(value)]
    if not isinstance(value, list) or any(not isinstance(tag, str) for tag in value):
        raise DiagnosticError(f"{field} must contain strings, got {value!r}")
    return [normalize_nfc(tag) for tag in value]


def mentions_translation_reference(value: Any) -> bool:
    """Detect whole-word 'translations' or 'see' references in text fields."""

    return isinstance(value, str) and re.search(r"\b(?:translations|see)\b", normalize_nfc(value), re.IGNORECASE) is not None


def _english_yield(path: Path, settings: dict[str, Any], aliases: dict[str, list[str]], logger) -> set[str]:
    """Run the yield-focused English diagnostics in a single streaming pass."""

    discovery_limit = settings["recursive_discovery_entries"]
    marker_limit = settings["trans_marker_entries"]
    locations: dict[str, dict[str, int]] = defaultdict(lambda: {"entries_with_any": 0, "entries_with_vi": 0, "translation_items": 0, "vi_translation_items": 0})
    discovered_paths: set[str] = set()
    marker_counts: Counter[str] = Counter()
    marker_scanned = 0
    entries = 0
    entries_any_translation = 0
    vi_items_total = 0
    entries_with_subpage_word = 0
    item_word_mentions = 0
    item_note_mentions = 0
    subpage_examples: list[dict[str, Any]] = []
    spot_matches: dict[str, list[dict[str, Any]]] = {word: [] for word in settings["spot_lemmas"]}
    mandarin_code_branch = 0
    mandarin_name_branch = 0
    mandarin_both_branches = 0
    mandarin_matched_items = 0
    script_counts: Counter[str] = Counter()
    excluded_chinese: list[dict[str, Any]] = []
    complete_groups: set[tuple[str, str, str]] = set()
    vietnamese_words_in_groups: set[str] = set()

    for line_number, entry in enumerate(iter_jsonl(path), start=1):
        entries += 1
        if not isinstance(entry, dict):
            raise DiagnosticError(f"English dump line {line_number}: expected an object, found {type(entry).__name__}")
        per_entry: dict[str, dict[str, int]] = defaultdict(lambda: {"translation_items": 0, "vi_translation_items": 0})
        per_entry_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
        per_entry_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
        for location, items, context in _walk_named_lists(entry, "translations"):
            if line_number <= discovery_limit:
                discovered_paths.add(location)
            local_vi = 0
            for item in items:
                if not isinstance(item, dict):
                    raise DiagnosticError(f"English dump line {line_number}, {location}: translation item must be an object, found {type(item).__name__}")
                if not isinstance(item.get("lang"), str):
                    raise DiagnosticError(f"English dump line {line_number}, {location}: translation item is missing string field 'lang'")
                per_entry[location]["translation_items"] += 1
                per_entry_items[location].append(item)
                local_vi += int(translation_matches(item, "vi", aliases))
                vi_items_total += int(translation_matches(item, "vi", aliases))

                mandarin, name_branch = _is_mandarin(item, settings)
                mandarin_matched_items += int(mandarin)
                code_branch = item.get("lang_code") == settings["mandarin_code"]
                mandarin_code_branch += int(code_branch)
                mandarin_name_branch += int(name_branch)
                mandarin_both_branches += int(code_branch and name_branch)
                tags = _tag_list(item, "tags") + _tag_list(item, "raw_tags")
                has_simplified = any(settings["simplified_tag"].casefold() in tag.casefold() for tag in tags)
                has_traditional = any(settings["traditional_tag"].casefold() in tag.casefold() for tag in tags)
                script_category = "both" if has_simplified and has_traditional else "simplified" if has_simplified else "traditional" if has_traditional else "neither"
                script_counts[script_category] += 1
                lang = item["lang"]
                if "chinese" in lang.casefold() and not mandarin and len(excluded_chinese) < settings["mandarin_nonmatch_examples"]:
                    excluded_chinese.append({"entry_word": entry.get("word"), **item})

                word_value = item.get("word")
                note_value = item.get("note")
                word_mention = mentions_translation_reference(word_value)
                note_mention = mentions_translation_reference(note_value)
                item_word_mentions += int(word_mention)
                item_note_mentions += int(note_mention)
                if (word_mention or note_mention) and len(subpage_examples) < settings["subpage_examples"]:
                    subpage_examples.append({"entry_word": entry.get("word"), "translation": item})

                word, pos = entry.get("word"), entry.get("pos")
                if not isinstance(word, str) or not isinstance(pos, str):
                    raise DiagnosticError(f"English dump line {line_number}: expected string word and pos fields")
                sense = _translation_sense(item, context)
                group_key = (normalize_nfc(word), normalize_nfc(pos), sense)
                group = per_entry_groups.setdefault(group_key, {"languages": set(), "vi_words": set()})
                if translation_matches(item, "vi", aliases):
                    group["languages"].add("vi")
                    if isinstance(word_value, str) and word_value:
                        group["vi_words"].add(normalize_nfc(word_value))
                if translation_matches(item, "fr", aliases):
                    group["languages"].add("fr")
                if translation_matches(item, "id", aliases):
                    group["languages"].add("id")
                if mandarin:
                    group["languages"].add("zh")

            per_entry[location]["vi_translation_items"] += local_vi
            per_entry[location]["entries_with_any"] = int(bool(items))
            per_entry[location]["entries_with_vi"] = int(local_vi > 0)

        if any(per_entry[location]["translation_items"] for location in per_entry):
            entries_any_translation += 1
        for location, counts in per_entry.items():
            locations[location]["entries_with_any"] += int(counts["translation_items"] > 0)
            locations[location]["entries_with_vi"] += int(counts["vi_translation_items"] > 0)
            locations[location]["translation_items"] += counts["translation_items"]
            locations[location]["vi_translation_items"] += counts["vi_translation_items"]

        word = entry.get("word")
        if isinstance(word, str) and normalize_nfc(word).casefold().endswith("/translations"):
            entries_with_subpage_word += 1
        if line_number <= marker_limit:
            marker_scanned += 1
            record_text = json.dumps(entry, ensure_ascii=False, sort_keys=True).casefold()
            for marker in settings["subpage_markers"]:
                if marker.casefold() in record_text:
                    marker_counts[marker] += 1

        pos = entry.get("pos")
        if isinstance(word, str) and isinstance(pos, str):
            for group_key, group in per_entry_groups.items():
                if {"vi", "zh", "fr", "id"}.issubset(group["languages"]):
                    complete_groups.add(group_key)
                    vietnamese_words_in_groups.update(group["vi_words"])
            folded_word = normalize_nfc(word).casefold()
            for lemma in spot_matches:
                if folded_word == normalize_nfc(lemma).casefold():
                    spot_matches[lemma].append({
                        "pos": pos,
                        "locations": {
                            location: {
                                "translation_items": per_entry[location]["translation_items"],
                                "present": {
                                    language: any(_language_item_matches(item, language, aliases, settings) for item in per_entry_items[location])
                                    for language in ("vi", "fr", "id", "zh")
                                },
                            }
                            for location in per_entry_items
                        },
                    })

    all_locations = sorted(locations)
    for matches in spot_matches.values():
        for match in matches:
            for location in all_locations:
                match["locations"].setdefault(location, {
                    "translation_items": 0,
                    "present": {language: False for language in ("vi", "fr", "id", "zh")},
                })

    missing_paths = sorted(set(locations) - discovered_paths)
    logger.info("A. English pass: entries=%s; entries with translations at any location=%s; recursive path discovery window=%s entries; marker scan window=%s entries", entries, entries_any_translation, min(entries, discovery_limit), marker_scanned)
    logger.info("A.1 Translation-list paths found recursively in discovery window: %s", json.dumps(sorted(discovered_paths), ensure_ascii=False))
    if missing_paths:
        logger.info("A.1 Additional translation-list paths first observed after discovery window (also counted): %s", json.dumps(missing_paths, ensure_ascii=False))
    logger.info("A.2 Per-location translation totals: path | entries with any | entries with vi | translation items | vi translation items")
    for location, counts in sorted(locations.items()):
        logger.info("A.2 %s | %s | %s | %s | %s", location, counts["entries_with_any"], counts["entries_with_vi"], counts["translation_items"], counts["vi_translation_items"])
    logger.info("A.2 Distinct English entries with any translation item across locations=%s; vi translation items across locations=%s", entries_any_translation, vi_items_total)

    logger.info("A.3 Requested lemma spot checks (all exact-word entries):")
    for lemma, matches in spot_matches.items():
        if not matches:
            logger.info("A.3 %s | NO ENTRY FOUND", lemma)
        for match in matches:
            pieces = [f"{location}:n={data['translation_items']},present={json.dumps(data['present'], sort_keys=True)}" for location, data in sorted(match["locations"].items())]
            logger.info("A.3 %s | pos=%s | %s", lemma, match["pos"], "; ".join(pieces))

    logger.info("A.4 Entries whose word ends in /translations=%s", entries_with_subpage_word)
    logger.info("A.4 Translation items with word mentioning translations/see=%s; items with note mentioning translations/see=%s; first examples=%s", item_word_mentions, item_note_mentions, json.dumps(subpage_examples, ensure_ascii=False, sort_keys=True))
    logger.info("A.4 Whole-record subpage marker counts in first %s entries: %s", marker_scanned, json.dumps({marker: marker_counts[marker] for marker in settings["subpage_markers"]}, ensure_ascii=False, sort_keys=True))
    logger.info("A.5 Distinct (word,pos,sense) groups across all locations with vi + Mandarin + fr + id=%s; distinct Vietnamese translation words in those groups=%s", len(complete_groups), len(vietnamese_words_in_groups))
    logger.info("B. Mandarin rule over all translation items=%s; branch lang_code==%r=%s; exact lang-name plus Mandarin tag=%s; overlap=%s", mandarin_matched_items, settings["mandarin_code"], mandarin_code_branch, mandarin_name_branch, mandarin_both_branches)
    logger.info("B. Script tags over all translation items (exclusive categories)=%s", json.dumps({key: script_counts[key] for key in ("simplified", "traditional", "both", "neither")}, sort_keys=True))
    logger.info("B. First %s Chinese-labeled items excluded by the Mandarin rule=%s", settings["mandarin_nonmatch_examples"], json.dumps(excluded_chinese, ensure_ascii=False, sort_keys=True))
    return vietnamese_words_in_groups


def _vietnamese_yield(path: Path, settings: dict[str, Any], english_words: set[str], logger) -> None:
    category_counts: Counter[str] = Counter()
    category_paths: set[str] = set()
    template_counts: Counter[str] = Counter()
    form_tag_counts: Counter[str] = Counter()
    form_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_tab: Counter[tuple[bool, bool, bool]] = Counter()
    five_way_cross_tab: Counter[tuple[bool, bool, bool]] = Counter()
    proto_term_counts: Counter[str] = Counter()
    total = 0
    matched_word_entries = 0
    matched_words: set[str] = set()
    cjk_ranges = settings["cjk_ranges"]
    proto_terms = [term.casefold() for term in settings["yield"]["proto_terms"]]
    etym_names = {"vi-etym-sino"}
    yield_settings = settings["yield"]

    for line_number, entry in enumerate(iter_jsonl(path), start=1):
        total += 1
        if not isinstance(entry, dict):
            raise DiagnosticError(f"Vietnamese dump line {line_number}: expected an object, found {type(entry).__name__}")
        for location, categories, _ in _walk_named_lists(entry, "categories"):
            category_paths.add(location)
            for category in categories:
                if not isinstance(category, dict):
                    raise DiagnosticError(f"Vietnamese dump line {line_number}, {location}: category must be an object")
                name = category.get("name")
                if not isinstance(name, str):
                    raise DiagnosticError(f"Vietnamese dump line {line_number}, {location}: category is missing string name")
                if "vietnamese" in normalize_nfc(name).casefold():
                    category_counts[normalize_nfc(name)] += 1

        templates = _require_list(entry, "etymology_templates", "Vietnamese dump", line_number)
        has_sino = False
        for template in templates:
            if not isinstance(template, dict):
                raise DiagnosticError(f"Vietnamese dump line {line_number}: etymology template must be an object")
            name = template.get("name")
            if isinstance(name, str):
                template_counts[normalize_nfc(name)] += 1
                has_sino |= name in etym_names

        forms = _require_list(entry, "forms", "Vietnamese dump", line_number)
        cjk_forms: list[dict[str, Any]] = []
        for form in forms:
            if not isinstance(form, dict):
                raise DiagnosticError(f"Vietnamese dump line {line_number}: form item must be an object")
            if contains_cjk(form.get("form", ""), cjk_ranges):
                tags = _tag_list(form, "tags")
                raw_tags = _tag_list(form, "raw_tags")
                signature = json.dumps({"tags": tags, "raw_tags": raw_tags}, ensure_ascii=False, sort_keys=True)
                form_tag_counts[signature] += 1
                cjk_forms.append(form)
                if len(form_examples[signature]) < yield_settings["cjk_examples_per_tag_set"]:
                    form_examples[signature].append({
                        "word": entry.get("word"),
                        "form": form.get("form"),
                        "tags": tags,
                        "raw_tags": raw_tags,
                        "has_vi_etym_sino": has_sino,
                    })

        etymology_text = entry.get("etymology_text", "")
        if etymology_text is not None and not isinstance(etymology_text, str):
            raise DiagnosticError(f"Vietnamese dump line {line_number}: etymology_text must be a string or null")
        etymology_folded = (etymology_text or "").casefold()
        for term in proto_terms:
            proto_term_counts[term] += int(term in etymology_folded)
        has_proto = any(term in etymology_folded for term in proto_terms)
        has_cjk_form = bool(cjk_forms)
        cross_tab[(has_sino, has_cjk_form, has_proto)] += 1
        word = entry.get("word")
        if isinstance(word, str) and normalize_nfc(word) in english_words:
            matched_word_entries += 1
            matched_words.add(normalize_nfc(word))
            five_way_cross_tab[(has_sino, has_cjk_form, has_proto)] += 1

    if not total:
        raise DiagnosticError(f"Vietnamese dump is empty: {path}")
    logger.info("C. Vietnamese entries scanned=%s; categories list paths=%s", total, json.dumps(sorted(category_paths), ensure_ascii=False))
    logger.info("C. Etymology-text mention counts by Proto term=%s", json.dumps(dict(sorted(proto_term_counts.items())), ensure_ascii=False, sort_keys=True))
    logger.info("C.1 Top %s category names containing 'Vietnamese' (category-object occurrences):", yield_settings["category_names_top_n"])
    for name, count in sorted(category_counts.items(), key=lambda row: (-row[1], row[0]))[:yield_settings["category_names_top_n"]]:
        logger.info("C.1 %s | %s", count, name)
    logger.info("C.2 Top %s etymology_templates[].name values (template occurrences):", yield_settings["etymology_templates_top_n"])
    for name, count in sorted(template_counts.items(), key=lambda row: (-row[1], row[0]))[:yield_settings["etymology_templates_top_n"]]:
        logger.info("C.2 %s | %s", count, name)

    top_signatures = sorted(form_tag_counts.items(), key=lambda row: (-row[1], row[0]))[:yield_settings["cjk_tag_sets_top_n"]]
    logger.info("C.3 CJK forms by (tags, raw_tags): distinct tag sets=%s; total CJK forms=%s", len(form_tag_counts), sum(form_tag_counts.values()))
    for signature, count in sorted(form_tag_counts.items(), key=lambda row: (-row[1], row[0])):
        logger.info("C.3 tag-set count=%s | %s", count, signature)
    for rank, (signature, count) in enumerate(top_signatures, start=1):
        logger.info("C.3 examples for top tag set %s (count=%s, signature=%s):", rank, count, signature)
        for example in form_examples[signature]:
            logger.info("C.3 %s | %s | tags=%s | raw_tags=%s | has vi-etym-sino=%s", example["word"], example["form"], json.dumps(example["tags"], ensure_ascii=False), json.dumps(example["raw_tags"], ensure_ascii=False), example["has_vi_etym_sino"])

    logger.info("C.4 Cross-tab: has vi-etym-sino | has CJK form | etymology mentions any configured Proto term | entries")
    for has_sino in (False, True):
        for has_form in (False, True):
            for has_proto in (False, True):
                logger.info("C.4 %s | %s | %s | %s", has_sino, has_form, has_proto, cross_tab[(has_sino, has_form, has_proto)])
    logger.info("C.5 Restricted to Vietnamese entries whose word appears among vi words from A.5: matched entries=%s; distinct matched words=%s of %s candidate forms", matched_word_entries, len(matched_words), len(english_words))
    logger.info("C.5 Cross-tab: has vi-etym-sino | has CJK form | etymology mentions any configured Proto term | entries")
    for has_sino in (False, True):
        for has_form in (False, True):
            for has_proto in (False, True):
                logger.info("C.5 %s | %s | %s | %s", has_sino, has_form, has_proto, five_way_cross_tab[(has_sino, has_form, has_proto)])


def run_yield(config_path: Path) -> int:
    """Run the extended yield diagnostics without mutating source/interim data."""

    config = load_config(config_path)
    repo_root = config_path.resolve().parents[1]
    paths = config["paths"]

    def repo_path(key: str) -> Path:
        value = Path(paths[key])
        return value if value.is_absolute() else repo_root / value

    logger = setup_logging("02_inspect", repo_path("logs"), level=config["logging"]["level"])
    raw = repo_path("raw")
    try:
        english_words = _english_yield(
            raw / "wiktextract_en" / "kaikki.org-dictionary-English.jsonl",
            config["inspection"]["yield"],
            config["inspection"]["language_aliases"],
            logger,
        )
        _vietnamese_yield(
            raw / "wiktextract_vi" / "kaikki.org-dictionary-Vietnamese.jsonl",
            config["inspection"],
            english_words,
            logger,
        )
    except (OSError, ValueError, KeyError, DiagnosticError) as exc:
        logger.error("STOP: %s", exc)
        return 2
    logger.info("Yield diagnostics completed (streaming; no source data written).")
    return 0


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
    parser.add_argument("--part", choices=("schema", "yield"), default="schema")
    args = parser.parse_args()
    try:
        return run_yield(args.config) if args.part == "yield" else run(args.config)
    except (DataConfigError, KeyError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
