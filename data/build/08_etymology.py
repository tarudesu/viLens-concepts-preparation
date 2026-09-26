"""Classify Wiktionary etymology signals and verify Han spellings externally."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from itertools import islice, product
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import nltk
import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from opencc import OpenCC

try:
    from common import DropflowLogger, iter_jsonl, load_config, normalize_nfc, normalize_strings, setup_logging, vi_orth_key
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, iter_jsonl, load_config, normalize_nfc, normalize_strings, setup_logging, vi_orth_key


STEP = "08_etymology"
SPLITS = ("fewshot_reservoir", "directions", "test")
A_CLASSES = ("sino", "nonsino", "other_loan", "conflict")
STRATA = ("sino", "nonsino", "other_loan", "ambiguous")
FAILURE_REASONS = ("no_han_string", "reading", "not_in_cedict", "meaning")
CEDICT_PATTERN = re.compile(r"^(\S+)\s+(\S+)\s+\[[^]]+\]\s+/(.*)/\s*$")
CODEPOINT_PATTERN = re.compile(r"U\+([0-9A-Fa-f]{4,6})")


def is_cjk(character: str, ranges: list[list[int]]) -> bool:
    """Return whether a character belongs to a configured CJK code-point range."""
    codepoint = ord(character)
    return any(int(start) <= codepoint <= int(end) for start, end in ranges)


def _category_names(entry: dict[str, Any]) -> list[str]:
    """Collect category names from the entry and all senses, validating shapes."""
    names: list[str] = []
    for field, owner in (("categories", "entry"),):
        categories = entry.get(field, [])
        if categories is None:
            categories = []
        if not isinstance(categories, list):
            raise ValueError(f"Unexpected {owner}.{field} in Vietnamese entry {entry.get('word')!r}")
        for category in categories:
            if not isinstance(category, dict) or not isinstance(category.get("name"), str):
                raise ValueError(f"Malformed category in Vietnamese entry {entry.get('word')!r}: {category!r}")
            names.append(normalize_nfc(category["name"]))
    senses = entry.get("senses", [])
    if senses is None:
        senses = []
    if not isinstance(senses, list):
        raise ValueError(f"Unexpected senses in Vietnamese entry {entry.get('word')!r}")
    for sense in senses:
        if not isinstance(sense, dict):
            raise ValueError(f"Malformed sense in Vietnamese entry {entry.get('word')!r}: {sense!r}")
        categories = sense.get("categories", [])
        if categories is None:
            categories = []
        if not isinstance(categories, list):
            raise ValueError(f"Unexpected sense.categories in Vietnamese entry {entry.get('word')!r}")
        for category in categories:
            if not isinstance(category, dict) or not isinstance(category.get("name"), str):
                raise ValueError(f"Malformed sense category in Vietnamese entry {entry.get('word')!r}: {category!r}")
            names.append(normalize_nfc(category["name"]))
    return names


def _templates(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated etymology templates for a Wiktextract entry."""
    templates = entry.get("etymology_templates", [])
    if templates is None:
        templates = []
    if not isinstance(templates, list):
        raise ValueError(f"Unexpected etymology_templates in Vietnamese entry {entry.get('word')!r}")
    for template in templates:
        if not isinstance(template, dict) or not isinstance(template.get("name"), str):
            raise ValueError(f"Malformed etymology template in Vietnamese entry {entry.get('word')!r}: {template!r}")
        if not isinstance(template.get("args", {}), dict):
            raise ValueError(f"Malformed template args in Vietnamese entry {entry.get('word')!r}: {template!r}")
    return templates


def _sinitic(source: str, settings: dict[str, Any]) -> bool:
    """Check exact Sinitic language codes and configured code prefixes."""
    folded = source.casefold().strip()
    return folded in {str(code).casefold() for code in settings["sinitic_codes"]} or any(
        folded.startswith(str(prefix).casefold()) for prefix in settings["sinitic_prefixes"]
    )


def _category_sino(entry: dict[str, Any]) -> bool:
    """Test the Wiktionary Sino-Vietnamese category across category locations."""
    return any(name.casefold() == "sino-vietnamese words" for name in _category_names(entry))


def classify_entry_a(
    entry: dict[str, Any], settings: dict[str, Any], cjk_ranges: list[list[int]],
) -> dict[str, Any]:
    """Classify one homograph entry for Signal A and collect its audit signals."""
    templates = _templates(entry)
    template_names = {str(item["name"]).casefold() for item in templates}
    has_sino_template = "vi-etym-sino" in template_names
    has_sino_category = _category_sino(entry)
    sino_via: set[str] = set()
    if has_sino_template or has_sino_category:
        sino_via.add("template")

    non_sinitic_sources: set[str] = set()
    native_evidence = False
    etymology_text = entry.get("etymology_text", "")
    if etymology_text is None:
        etymology_text = ""
    if not isinstance(etymology_text, str):
        raise ValueError(f"Unexpected etymology_text in Vietnamese entry {entry.get('word')!r}")
    folded_text = etymology_text.casefold()
    native_evidence = any(str(marker).casefold() in folded_text for marker in settings["native_markers_text"])
    native_evidence = native_evidence or bool(template_names.intersection({str(item).casefold() for item in settings["native_templates"]}))

    loan_names = {str(item).casefold() for item in settings["loan_templates"]}
    calque_names = {str(item).casefold() for item in settings["calque_templates"]}
    for template in templates:
        name = str(template["name"]).casefold()
        args = template.get("args", {})
        source_value = args.get("2")
        argument_value = args.get("3")
        if name in loan_names | calque_names:
            if not isinstance(source_value, str) or not source_value.strip():
                raise ValueError(f"Template {name!r} in {entry.get('word')!r} lacks args['2'] source language")
            source = normalize_nfc(source_value).casefold().strip()
            cjk_argument = isinstance(argument_value, str) and any(is_cjk(char, cjk_ranges) for char in argument_value)
            if _sinitic(source, settings):
                sino_via.add("calque" if name in calque_names else "chinese")
            elif name in loan_names:
                if source in {str(code).casefold() for code in settings["sino_japanese_codes"]} and cjk_argument:
                    sino_via.add("japanese_kango")
                else:
                    is_proto = source.endswith("-pro")
                    if not is_proto:
                        non_sinitic_sources.add(source)
            if name in {"der", "inh", "inh+"} and source.endswith("-pro"):
                native_evidence = True

    label = "sino" if sino_via else "other_loan" if non_sinitic_sources else "nonsino"
    return {
        "label": label,
        "sino_via": sorted(sino_via),
        "other_loan_sources": sorted(non_sinitic_sources),
        "native_evidence": native_evidence,
    }


def aggregate_signal_a(entry_results: list[dict[str, Any]]) -> tuple[str, list[str], list[str], bool]:
    """Aggregate homograph-level Signal A labels, retaining the specified conflict."""
    if not entry_results:
        raise ValueError("Cannot aggregate Signal A with no in-scope homograph entries")
    labels = [result["label"] for result in entry_results]
    if any(label not in {"sino", "nonsino", "other_loan"} for label in labels):
        raise ValueError(f"Unexpected homograph Signal A labels: {labels!r}")
    has_sino = "sino" in labels
    has_non_sino = any(label != "sino" for label in labels)
    if has_sino and has_non_sino:
        signal_a = "conflict"
    elif has_sino:
        signal_a = "sino"
    elif "other_loan" in labels:
        signal_a = "other_loan"
    else:
        signal_a = "nonsino"
    via = sorted({item for result in entry_results for item in result["sino_via"]})
    other_sources = sorted({item for result in entry_results for item in result["other_loan_sources"]})
    native_strict = signal_a == "nonsino" and any(bool(result["native_evidence"]) for result in entry_results)
    return signal_a, via, other_sources, native_strict


def classify_stratum(signal_a: str, signal_b: str) -> str:
    """Apply the predeclared A/B stratum mapping."""
    if signal_a not in {"sino", "nonsino", "other_loan", "conflict"}:
        raise ValueError(f"Unknown Signal A class: {signal_a!r}")
    if signal_b not in {"sino", "nonsino"}:
        raise ValueError(f"Unknown Signal B class: {signal_b!r}")
    if signal_a == "sino" and signal_b == "sino":
        return "sino"
    if signal_a == "nonsino" and signal_b == "nonsino":
        return "nonsino"
    if signal_a == "other_loan":
        return "other_loan"
    return "ambiguous"


def extract_han_strings(values: Iterable[str], cjk_ranges: list[list[int]], syllable_count: int) -> list[str]:
    """Extract distinct CJK runs from values, retaining only syllable-length strings."""
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        for piece in re.split(r"[/\s]+", normalize_nfc(value).strip()):
            run: list[str] = []
            for character in piece:
                if is_cjk(character, cjk_ranges):
                    run.append(character)
                elif run:
                    candidate = "".join(run)
                    if len(candidate) == syllable_count:
                        result.add(candidate)
                    run = []
            if run:
                candidate = "".join(run)
                if len(candidate) == syllable_count:
                    result.add(candidate)
    return sorted(result)


def _template_variant_groups(args: dict[str, Any], cjk_ranges: list[list[int]], split_on: str) -> list[list[str]]:
    """Return Han-only positional template arguments as ordered variant groups."""
    groups: list[list[str]] = []
    positional = sorted(
        ((int(key), value) for key, value in args.items() if str(key).isdigit()),
        key=lambda item: item[0],
    )
    for _, value in positional:
        if not isinstance(value, str):
            continue
        variants = sorted({normalize_nfc(piece.strip()) for piece in value.split(split_on) if piece.strip()})
        if variants and all(all(is_cjk(character, cjk_ranges) for character in variant) for variant in variants):
            groups.append(variants)
    return groups


def extract_sino_template_strings(
    entries: list[dict[str, Any]], cjk_ranges: list[list[int]], syllable_count: int,
    *, split_on: str, max_combinations: int,
) -> list[str]:
    """Build length-matched Han strings from positional Han/gloss template args."""
    if not split_on:
        raise ValueError("Template variant separator must be non-empty")
    if type(max_combinations) is not int or max_combinations < 1:
        raise ValueError("Maximum template combinations must be a positive integer")
    result: set[str] = set()
    for entry in entries:
        for template in _templates(entry):
            if str(template["name"]).casefold() != "vi-etym-sino":
                continue
            groups = _template_variant_groups(template.get("args", {}), cjk_ranges, split_on)
            if not groups:
                continue
            for combination in islice(product(*groups), max_combinations):
                candidate = "".join(combination)
                if len(candidate) == syllable_count:
                    result.add(candidate)
    return sorted(result)


def collect_han_strings(
    entries: list[dict[str, Any]], cjk_ranges: list[list[int]], syllable_count: int,
    *, split_on: str, max_combinations: int,
) -> list[str]:
    """Gather template-product strings plus length-matched CJK forms."""
    form_values: list[str] = []
    for entry in entries:
        forms = entry.get("forms", [])
        if forms is None:
            forms = []
        if not isinstance(forms, list):
            raise ValueError(f"Unexpected forms in Vietnamese entry {entry.get('word')!r}")
        for form in forms:
            if not isinstance(form, dict) or not isinstance(form.get("form"), str):
                raise ValueError(f"Malformed form in Vietnamese entry {entry.get('word')!r}: {form!r}")
            form_values.append(form["form"])
    template_strings = extract_sino_template_strings(
        entries, cjk_ranges, syllable_count, split_on=split_on, max_combinations=max_combinations,
    )
    form_strings = extract_han_strings(form_values, cjk_ranges, syllable_count)
    return sorted(set(template_strings) | set(form_strings))


def load_unihan(readings_path: str | Path, variants_path: str | Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Read kVietnamese readings and undirected traditional/simplified variant links."""
    readings: dict[str, set[str]] = defaultdict(set)
    found_reading_field = False
    with Path(readings_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed Unihan Readings line {line_number}: expected 3 tab-separated fields")
            code, field, value = fields
            if field != "kVietnamese":
                continue
            match = CODEPOINT_PATTERN.fullmatch(code)
            if not match:
                raise ValueError(f"Malformed Unihan code point at line {line_number}: {code!r}")
            character = chr(int(match.group(1), 16))
            alternatives = [normalize_nfc(item) for item in re.split(r"\s+", value.strip()) if item]
            if not alternatives:
                raise ValueError(f"Empty kVietnamese value at line {line_number}")
            readings[character].update(alternatives)
            found_reading_field = True
    if not found_reading_field:
        raise ValueError(f"Unihan Readings has no kVietnamese field: {readings_path}")

    variants: dict[str, set[str]] = defaultdict(set)
    fields_seen: set[str] = set()
    with Path(variants_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed Unihan Variants line {line_number}: expected 3 tab-separated fields")
            code, field, value = fields
            if field not in {"kTraditionalVariant", "kSimplifiedVariant"}:
                continue
            fields_seen.add(field)
            source_match = CODEPOINT_PATTERN.fullmatch(code)
            targets = CODEPOINT_PATTERN.findall(value)
            if not source_match or not targets:
                raise ValueError(f"Malformed Unihan {field} entry at line {line_number}: {line.rstrip()!r}")
            source_character = chr(int(source_match.group(1), 16))
            for target in targets:
                target_character = chr(int(target, 16))
                variants[source_character].add(target_character)
                variants[target_character].add(source_character)
    expected_fields = {"kTraditionalVariant", "kSimplifiedVariant"}
    if fields_seen != expected_fields:
        raise ValueError(f"Unihan Variants missing required fields: {sorted(expected_fields - fields_seen)}")
    return dict(readings), dict(variants)


def load_cedict(path: str | Path) -> dict[str, list[str]]:
    """Stream CC-CEDICT and index glosses under both traditional and simplified forms."""
    cedict: dict[str, list[str]] = defaultdict(list)
    opener = gzip.open if str(path).endswith(".gz") else open
    header_found = False
    entry_count = 0
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = normalize_nfc(line.rstrip("\r\n"))
            if line.startswith("# CC-CEDICT"):
                header_found = True
                continue
            if not line or line.startswith("#"):
                continue
            match = CEDICT_PATTERN.match(line)
            if not match:
                raise ValueError(f"Malformed CC-CEDICT entry at {path}:{line_number}: {line[:160]!r}")
            traditional, simplified, gloss_text = match.groups()
            glosses = [part.strip() for part in gloss_text.split("/") if part.strip()]
            if not glosses:
                raise ValueError(f"CC-CEDICT entry has no gloss at {path}:{line_number}")
            for headword in sorted({traditional, simplified}):
                cedict[headword].extend(glosses)
            entry_count += 1
    if not header_found or not entry_count:
        raise ValueError(f"File does not look like a non-empty CC-CEDICT dump: {path}")
    return {headword: sorted(set(glosses)) for headword, glosses in cedict.items()}


def variant_closure(character: str, variants: dict[str, set[str]]) -> set[str]:
    """Return a deterministic transitive closure over Unihan variant links."""
    found = {character}
    frontier = [character]
    while frontier:
        current = frontier.pop()
        for candidate in sorted(variants.get(current, set())):
            if candidate not in found:
                found.add(candidate)
                frontier.append(candidate)
    return found


def _single_character_readings(entry: dict[str, Any], cjk_ranges: list[list[int]]) -> set[str]:
    """Collect explicit Hán-Việt readings plus romanization/head transliterations."""
    word = entry.get("word")
    if not isinstance(word, str) or len(word) != 1 or not is_cjk(word, cjk_ranges):
        return set()
    found: set[str] = set()

    def add_reading(value: Any) -> None:
        if not isinstance(value, str):
            return
        normalized = normalize_nfc(value).strip()
        if not normalized or any(is_cjk(character, cjk_ranges) for character in normalized):
            return
        for reading in re.split(r"[,;/]", normalized):
            reading = normalize_nfc(reading.strip())
            if reading and not any(is_cjk(character, cjk_ranges) for character in reading):
                found.add(reading)

    forms = entry.get("forms") or []
    if not isinstance(forms, list):
        raise ValueError(f"Unexpected forms on single-character Vietnamese entry {word!r}")
    for form in forms:
        if not isinstance(form, dict) or not isinstance(form.get("form"), str):
            raise ValueError(f"Malformed form on single-character Vietnamese entry {word!r}: {form!r}")
        tags = form.get("tags") or []
        raw_tags = form.get("raw_tags") or []
        if not isinstance(tags, list) or not isinstance(raw_tags, list):
            raise ValueError(f"Malformed form tags on single-character Vietnamese entry {word!r}: {form!r}")
        tag_names = {str(tag).casefold() for tag in [*tags, *raw_tags]}
        value = normalize_nfc(form["form"])
        marker = re.search(r"hán[- ]việt readings?\s*:\s*(.+)$", value, flags=re.IGNORECASE)
        if marker:
            add_reading(marker.group(1))
        elif tag_names.intersection({"romanization", "canonical", "hán-nôm", "han-viet-reading"}):
            add_reading(value)

    head_templates = entry.get("head_templates") or []
    if not isinstance(head_templates, list):
        raise ValueError(f"Unexpected head_templates on single-character Vietnamese entry {word!r}")
    for template in head_templates:
        if not isinstance(template, dict) or not isinstance(template.get("name"), str):
            raise ValueError(f"Malformed head template on single-character Vietnamese entry {word!r}: {template!r}")
        args = template.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"Malformed head-template args on single-character Vietnamese entry {word!r}: {template!r}")
        add_reading(args.get("tr"))
        if template["name"].casefold().startswith("vi-"):
            add_reading(args.get("1"))

    senses = entry.get("senses") or []
    if not isinstance(senses, list):
        raise ValueError(f"Unexpected senses on single-character Vietnamese entry {word!r}")
    for sense in senses:
        if not isinstance(sense, dict):
            raise ValueError(f"Malformed sense on single-character Vietnamese entry {word!r}: {sense!r}")
        related = sense.get("related") or []
        if not isinstance(related, list):
            raise ValueError(f"Unexpected related readings on single-character Vietnamese entry {word!r}")
        for item in related:
            if not isinstance(item, dict):
                raise ValueError(f"Malformed related reading on single-character Vietnamese entry {word!r}: {item!r}")
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                raise ValueError(f"Malformed related-reading tags on {word!r}: {item!r}")
            if "han-viet-reading" in {str(tag).casefold() for tag in tags}:
                add_reading(item.get("word"))
    return found


def reading_match_source(
    han_string: str, canonical_vi: str, readings: dict[str, set[str]],
    variants: dict[str, set[str]], wiktionary_readings: dict[str, set[str]],
) -> str | None:
    """Return Unihan or Wiktionary fallback when every Han character reading matches."""
    syllables = vi_orth_key(canonical_vi).split()
    if len(han_string) != len(syllables):
        return None
    used_fallback = False
    for character, syllable in zip(han_string, syllables, strict=True):
        variant_chars = variant_closure(character, variants)
        target_key = vi_orth_key(syllable)
        unihan_keys = {
            vi_orth_key(reading)
            for variant in variant_chars
            for reading in readings.get(variant, set())
        }
        if target_key in unihan_keys:
            continue
        fallback_keys = {
            vi_orth_key(reading)
            for variant in variant_chars
            for reading in wiktionary_readings.get(variant, set())
        }
        if target_key not in fallback_keys:
            return None
        used_fallback = True
    return "wiktionary_char" if used_fallback else "unihan"


def reading_matches(
    han_string: str, canonical_vi: str, readings: dict[str, set[str]],
    variants: dict[str, set[str]], wiktionary_readings: dict[str, set[str]] | None = None,
) -> bool:
    """Compare per-character Hán-Việt readings using Unihan then local fallback."""
    return reading_match_source(han_string, canonical_vi, readings, variants, wiktionary_readings or {}) is not None


def cedict_forms(han_string: str, t2s: OpenCC, s2t: OpenCC) -> list[str]:
    """Return the input and OpenCC traditional/simplified variants in stable order."""
    return sorted({normalize_nfc(han_string), normalize_nfc(t2s.convert(han_string)), normalize_nfc(s2t.convert(han_string))})


def evaluate_b_candidate(
    reading_ok: bool, headword_found: bool, overlap_count: int, overlap_min: int, syllable_count: int,
) -> dict[str, Any]:
    """Evaluate primary, strict and relaxed Signal B rules for one Han candidate."""
    if type(syllable_count) is not int or syllable_count < 1:
        raise ValueError("Signal-B syllable count must be a positive integer")
    relaxed_pass = reading_ok and headword_found
    meaning_pass = overlap_count >= overlap_min
    strict_pass = relaxed_pass and meaning_pass
    primary_pass = relaxed_pass and (syllable_count >= 2 or meaning_pass)
    if primary_pass:
        failure = None
    elif not reading_ok:
        failure = "reading"
    elif not headword_found:
        failure = "not_in_cedict"
    else:
        failure = "meaning"
    return {
        "primary_pass": primary_pass,
        "strict_pass": strict_pass,
        "relaxed_pass": relaxed_pass,
        "failure_reason": failure,
    }


def summarize_signal_b(candidate_results: list[dict[str, Any]]) -> tuple[str, str, str, str | None]:
    """Aggregate per-Han checks into primary, strict, relaxed classes and a failure."""
    if not candidate_results:
        return "nonsino", "nonsino", "nonsino", "no_han_string"
    primary = any(item["primary_pass"] for item in candidate_results)
    strict = any(item["strict_pass"] for item in candidate_results)
    relaxed = any(item["relaxed_pass"] for item in candidate_results)
    if primary:
        failure = None
    elif not any(item["reading_ok"] for item in candidate_results):
        failure = "reading"
    elif not any(item["reading_ok"] and item["headword_found"] for item in candidate_results):
        failure = "not_in_cedict"
    else:
        failure = "meaning"
    return (
        "sino" if primary else "nonsino",
        "sino" if strict else "nonsino",
        "sino" if relaxed else "nonsino",
        failure,
    )


def load_local_wordnet(data_dir: str | Path) -> tuple[Any, str]:
    """Load WordNet only from the supplied local NLTK data directory."""
    local_dir = Path(data_dir).resolve()
    nltk.data.path = [str(local_dir)]
    from nltk.corpus import wordnet

    try:
        wordnet.ensure_loaded()
        version = wordnet.get_version()
        wordnet.synsets("entity")
    except LookupError as exc:
        raise RuntimeError(
            f"Local NLTK WordNet corpus is missing or unreadable under {local_dir}. "
            "Run step 01 with --only wordnet; step 08 does not access the network."
        ) from exc
    return wordnet, version


def wordnet_synonyms(wordnet_reader: Any, lemma: str) -> list[str]:
    """Return deterministic WordNet lemma names for an English lemma or phrase."""
    normalized = normalize_nfc(lemma).casefold().strip()
    queries = sorted({normalized, normalized.replace(" ", "_")})
    synonyms: set[str] = set()
    for query in queries:
        for synset in wordnet_reader.synsets(query):
            for name in synset.lemma_names():
                synonym = normalize_nfc(name.replace("_", " ")).strip()
                if synonym:
                    synonyms.add(synonym)
    return sorted(synonyms)


def max_content_lemma_overlap(target_lemmas: set[str], gloss_lemma_sets: Iterable[set[str]]) -> int:
    """Return the largest content-lemma intersection with any candidate gloss."""
    return max((len(target_lemmas.intersection(gloss_lemmas)) for gloss_lemmas in gloss_lemma_sets), default=0)


def cohens_kappa(left: list[str], right: list[str]) -> float | None:
    """Compute Cohen's kappa for paired categorical labels; return None if undefined."""
    if len(left) != len(right):
        raise ValueError("Cohen's kappa inputs must have equal length")
    if not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (len(left) ** 2)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def bootstrap_kappa_ci(
    left: list[str], right: list[str], *, resamples: int, confidence: float, seed: int,
) -> tuple[float | None, float | None, float | None]:
    """Return kappa and a seeded percentile bootstrap confidence interval."""
    if type(resamples) is not int or resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")
    estimate = cohens_kappa(left, right)
    if not left:
        return estimate, None, None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    length = len(left)
    for _ in range(resamples):
        indexes = rng.integers(0, length, size=length)
        value = cohens_kappa([left[int(index)] for index in indexes], [right[int(index)] for index in indexes])
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return estimate, None, None
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [alpha, 1.0 - alpha])
    return estimate, float(lower), float(upper)


def minimum_detectable_difference(n1: int, n2: int, *, alpha: float, power: float) -> float | None:
    """Compute the requested two-sided standardized MDE using normal quantiles."""
    if n1 < 0 or n2 < 0 or not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("Invalid power-analysis parameters")
    if n1 == 0 or n2 == 0:
        return None
    from statistics import NormalDist

    normal = NormalDist()
    return (normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)) * math.sqrt(1.0 / n1 + 1.0 / n2)


def _read_canonical_entries(
    path: Path, keys: set[str], cjk_ranges: list[list[int]], logger: logging.Logger, progress_every: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    """Stream matching entries and index readings on single-CJK-headword entries."""
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    character_readings: dict[str, set[str]] = defaultdict(set)
    entry_count = 0
    for entry_count, entry in enumerate(iter_jsonl(path), start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Vietnamese dump record {entry_count} is not an object")
        word = entry.get("word")
        pos = entry.get("pos")
        if not isinstance(word, str) or not word.strip() or not isinstance(pos, str) or not pos.strip():
            raise ValueError(f"Vietnamese dump record {entry_count} lacks a non-empty word/pos")
        key = vi_orth_key(word)
        if key in keys:
            found[key].append(entry)
        if len(word) == 1 and is_cjk(word, cjk_ranges):
            character_readings[word].update(_single_character_readings(entry, cjk_ranges))
        if entry_count % progress_every == 0:
            logger.info("Streamed Vietnamese entries: %d", entry_count)
    logger.info("Streamed Vietnamese entries: %d (end of dump)", entry_count)
    missing = sorted(keys - set(found))
    if missing:
        raise ValueError(f"Canonical VI forms have no matching Wiktextract homograph entries ({len(missing)}): {missing[:20]!r}")
    return dict(found), dict(character_readings)


def _scope_entries(entries: list[dict[str, Any]], pos: str) -> tuple[list[dict[str, Any]], str]:
    """Use exact-POS homographs where available, otherwise all matched entries."""
    matching = [entry for entry in entries if normalize_nfc(entry["pos"]) == normalize_nfc(pos)]
    return (matching, "pos_match") if matching else (entries, "all_fallback")


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path, row_group_size: int) -> None:
    """Write zstd-compressed parquet atomically with deterministic schema/order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as handle:
            temporary = handle.name
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), temporary,
            compression="zstd", version="2.6", row_group_size=row_group_size, write_statistics=True,
        )
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _remove_step_dropflow(path: Path) -> None:
    """Remove prior step-08 records before a deterministic rerun."""
    if not path.exists():
        return
    retained: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed dropflow record at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("step"), str):
            raise ValueError(f"Unexpected dropflow record at {path}:{line_number}")
        if record["step"] != "08":
            retained.append(line if line.endswith("\n") else line + "\n")
    path.write_text("".join(retained), encoding="utf-8", newline="\n")


def _dropflow_and_log(rows: list[dict[str, Any]], config: dict[str, Any], logger: logging.Logger) -> None:
    """Record the non-dropping classification stage once per split."""
    dropflow_path = Path(config["paths"]["dropflow"])
    _remove_step_dropflow(dropflow_path)
    flow = DropflowLogger(dropflow_path)
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        counts = dict(sorted(Counter(row["pos"] for row in split_rows).items()))
        flow.record(
            step="08", stage=f"etymology_classified_{split}", unit="concepts",
            n_in=len(split_rows), n_out=len(split_rows), n_out_by_pos=counts,
        )
        logger.info("etymology_classified_%s | n_in=%d | n_out=%d | n_out_by_pos=%s",
                    split, len(split_rows), len(split_rows), json.dumps(counts, sort_keys=True))


def run(config_path: str) -> dict[str, Any]:
    """Run Signals A/B, write the classified parquet, and report checkpoint-C diagnostics."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    try:
        settings = config["etymology"]
        paths = settings["paths"]
        required_settings = (
            "sinitic_codes", "sinitic_prefixes", "sino_japanese_codes", "loan_templates",
            "calque_templates", "native_markers_text", "native_templates", "gloss_overlap_min",
            "bootstrap_resamples", "bootstrap_confidence", "power_alpha", "power_holm_comparisons",
            "power_target", "report", "paths", "template_variant_separator", "max_template_combinations",
            "h3_confirmatory_kappa_lower_min", "h3_confirmatory_holm_mde_max",
        )
        missing_settings = [key for key in required_settings if key not in settings]
        if missing_settings:
            raise ValueError(f"Missing etymology config keys: {missing_settings}")
        if type(settings["gloss_overlap_min"]) is not int or settings["gloss_overlap_min"] < 1:
            raise ValueError("etymology.gloss_overlap_min must be a positive integer")
        if type(settings["power_holm_comparisons"]) is not int or settings["power_holm_comparisons"] < 1:
            raise ValueError("etymology.power_holm_comparisons must be a positive integer")
        if type(settings["max_template_combinations"]) is not int or settings["max_template_combinations"] < 1:
            raise ValueError("etymology.max_template_combinations must be a positive integer")
        if not 0.0 < float(settings["h3_confirmatory_kappa_lower_min"]) <= 1.0:
            raise ValueError("etymology.h3_confirmatory_kappa_lower_min must be in (0, 1]")
        if not 0.0 < float(settings["h3_confirmatory_holm_mde_max"]):
            raise ValueError("etymology.h3_confirmatory_holm_mde_max must be positive")
        wordnet_reader, wordnet_version = load_local_wordnet(paths["nltk_data"])
        logger.info("Loaded local NLTK WordNet %s from %s; nltk.data.path restricted to this directory",
                    wordnet_version, Path(paths["nltk_data"]).resolve())
        report = settings["report"]
        input_path = Path(paths["input"])
        table = pq.read_table(input_path)
        required_columns = {
            "concept_id", "en_lemma", "pos", "sense_gloss", "split", "vi_canonical",
            "post_promotion_survives", "zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree",
        }
        missing_columns = sorted(required_columns - set(table.column_names))
        if missing_columns:
            raise ValueError(f"Step-07 parquet is missing required columns: {missing_columns}")
        all_rows = [normalize_strings(row) for row in table.to_pylist()]
        if not all_rows:
            raise ValueError(f"Step-07 parquet has no rows: {input_path}")
        if any(row["post_promotion_survives"] is not True for row in all_rows):
            raise ValueError("Step-07 input contains concepts that did not survive filter 2/post-promotion checks")
        rows = sorted(all_rows, key=lambda row: normalize_nfc(row["concept_id"]))
        if len({row["concept_id"] for row in rows}) != len(rows):
            raise ValueError("Step-07 input contains duplicate concept_id values")
        if {row["split"] for row in rows} != set(SPLITS):
            raise ValueError(f"Step-07 split values differ from expected {SPLITS}: {sorted({row['split'] for row in rows})}")
        for row in rows:
            for field in ("concept_id", "en_lemma", "pos", "sense_gloss", "vi_canonical"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"Step-07 row {row.get('concept_id')!r} has missing/malformed {field}")
            for field in ("zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree"):
                if type(row[field]) is not bool:
                    raise ValueError(f"Step-07 row {row.get('concept_id')!r} has missing/malformed {field}")

        cjk_ranges = config["inspection"]["cjk_ranges"]
        if not isinstance(cjk_ranges, list) or not cjk_ranges:
            raise ValueError("inspection.cjk_ranges must define CJK code point ranges")
        canonical_keys = {vi_orth_key(row["vi_canonical"]) for row in rows}
        entries_by_key, wiktionary_readings = _read_canonical_entries(
            Path(paths["vietnamese_dump"]), canonical_keys, cjk_ranges, logger,
            int(config["logging"]["progress_every"]),
        )
        logger.info("Matched canonical Vietnamese forms to homograph entries: %d/%d unique forms",
                    len(entries_by_key), len(canonical_keys))
        logger.info("Indexed Wiktextract fallback readings for %d single-CJK headwords (%d readings)",
                    len(wiktionary_readings), sum(len(values) for values in wiktionary_readings.values()))
        readings, variants = load_unihan(paths["unihan_readings"], paths["unihan_variants"])
        logger.info("Loaded Unihan kVietnamese readings for %d characters and variant links for %d characters",
                    len(readings), len(variants))
        cedict = load_cedict(paths["cedict"])
        logger.info("Loaded CC-CEDICT headword forms: %d", len(cedict))
        t2s, s2t = OpenCC("t2s"), OpenCC("s2t")

        enriched: list[dict[str, Any]] = []
        target_texts: set[str] = set()
        candidate_stage: list[dict[str, Any]] = []
        for row in rows:
            canonical = row["vi_canonical"]
            matches = entries_by_key[vi_orth_key(canonical)]
            scoped, entry_scope = _scope_entries(matches, row["pos"])
            entry_results = [classify_entry_a(entry, settings, cjk_ranges) for entry in scoped]
            signal_a, sino_via, other_sources, native_strict = aggregate_signal_a(entry_results)
            syllable_count = len(vi_orth_key(canonical).split())
            han_strings = collect_han_strings(
                scoped, cjk_ranges, syllable_count,
                split_on=str(settings["template_variant_separator"]),
                max_combinations=int(settings["max_template_combinations"]),
            )
            candidate_forms: dict[str, list[str]] = {}
            for han_string in han_strings:
                forms = cedict_forms(han_string, t2s, s2t)
                candidate_forms[han_string] = sorted({form for form in forms if form in cedict})
            gloss_texts: set[str] = set()
            for entry in scoped:
                senses = entry.get("senses") or []
                if not isinstance(senses, list):
                    raise ValueError(f"Unexpected senses in Vietnamese entry {entry.get('word')!r}")
                for sense in senses:
                    if not isinstance(sense, dict):
                        raise ValueError(f"Malformed sense in Vietnamese entry {entry.get('word')!r}: {sense!r}")
                    glosses = sense.get("glosses") or []
                    if not isinstance(glosses, list):
                        raise ValueError(f"Unexpected glosses in Vietnamese entry {entry.get('word')!r}")
                    for gloss in glosses:
                        if not isinstance(gloss, str):
                            raise ValueError(f"Malformed gloss in Vietnamese entry {entry.get('word')!r}: {gloss!r}")
                        if gloss.strip():
                            gloss_texts.add(normalize_nfc(gloss).strip())
            synonyms = wordnet_synonyms(wordnet_reader, row["en_lemma"])
            meaning_texts = sorted({
                normalize_nfc(text).strip()
                for text in [row["en_lemma"], row["sense_gloss"], *gloss_texts, *synonyms]
                if isinstance(text, str) and text.strip()
            })
            target_texts.update(meaning_texts)
            candidate_glosses = sorted({gloss for values in candidate_forms.values() for form in values for gloss in cedict[form]})
            target_texts.update(candidate_glosses)
            candidate_stage.append({
                "row": row, "entries": scoped, "entry_scope": entry_scope,
                "entry_results": entry_results,
                "signal_a": signal_a, "sino_via": sino_via,
                "other_loan_sources": other_sources, "native_strict": native_strict,
                "han_strings": han_strings, "candidate_forms": candidate_forms,
                "meaning_texts": meaning_texts,
                "nllb_all_agree": all(row[field] is True for field in ("zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree")),
            })

        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError("Required spaCy model en_core_web_sm is unavailable; install it before step 08") from exc
        logger.info("Loaded spaCy en_core_web_sm for CEDICT gloss overlap")
        lemma_sets: dict[str, set[str]] = {}
        texts = sorted(target_texts)
        for index, (text, doc) in enumerate(zip(texts, nlp.pipe(texts, batch_size=256), strict=True), start=1):
            lemma_sets[text] = {
                token.lemma_.casefold()
                for token in doc
                if token.is_alpha and not token.is_stop and token.lemma_
            }
            progress_every = int(config["logging"]["progress_every"])
            if index % progress_every == 0 or index == len(texts):
                logger.info("spaCy lemma extraction: %d/%d unique English strings", index, len(texts))

        for item in candidate_stage:
            row = item["row"]
            canonical = row["vi_canonical"]
            target_lemmas: set[str] = set()
            for meaning_text in item["meaning_texts"]:
                target_lemmas.update(lemma_sets[meaning_text])
            checks: list[dict[str, Any]] = []
            for han_string in item["han_strings"]:
                reading_source = reading_match_source(
                    han_string, canonical, readings, variants, wiktionary_readings,
                )
                reading_ok = reading_source is not None
                headwords = item["candidate_forms"][han_string]
                glosses = sorted({gloss for headword in headwords for gloss in cedict[headword]})
                overlap_count = max_content_lemma_overlap(target_lemmas, (lemma_sets[gloss] for gloss in glosses))
                evaluation = evaluate_b_candidate(
                    reading_ok, bool(headwords), overlap_count,
                    int(settings["gloss_overlap_min"]), len(vi_orth_key(canonical).split()),
                )
                checks.append({
                    "han_string": han_string,
                    "reading_ok": reading_ok,
                    "reading_source": reading_source,
                    "headword_found": bool(headwords),
                    "meaning_overlap": overlap_count,
                    **evaluation,
                    "cedict_glosses": glosses,
                })
            signal_b, signal_b_strict, signal_b_relaxed, failure_reason = summarize_signal_b(checks)
            stratum = classify_stratum(item["signal_a"], signal_b)
            passing = [check for check in checks if check["primary_pass"]]
            item["output"] = {
                **row,
                "entry_scope": item["entry_scope"],
                "signal_a": item["signal_a"],
                "signal_a_entry_labels": [result["label"] for result in item["entry_results"]],
                "sino_via": item["sino_via"],
                "other_loan_sources": item["other_loan_sources"],
                "native_strict": item["native_strict"],
                "han_strings": item["han_strings"],
                "signal_b": signal_b,
                "signal_b_strict": signal_b_strict,
                "signal_b_relaxed": signal_b_relaxed,
                "signal_b_failure_reason": failure_reason,
                "verified_han_string": passing[0]["han_string"] if passing else None,
                "verified_cedict_glosses": passing[0]["cedict_glosses"] if passing else [],
                "han_verifications": checks,
                "stratum": stratum,
                "nllb_all_agree": item["nllb_all_agree"],
            }
            enriched.append(item["output"])

        logger.info("Input/output concepts: %d/%d; all splits retained", len(rows), len(enriched))
        cross = {a: Counter() for a in A_CLASSES}
        for row in enriched:
            cross[row["signal_a"]][row["signal_b"]] += 1
        logger.info("Signal A x Signal B crosstab (rows=A, columns=B):")
        logger.info("A | B=sino | B=nonsino")
        for signal_a in A_CLASSES:
            logger.info("%s | %d | %d", signal_a, cross[signal_a]["sino"], cross[signal_a]["nonsino"])

        kappa_rows = [row for row in enriched if row["signal_a"] in {"sino", "nonsino"}]
        a_labels = [row["signal_a"] for row in kappa_rows]
        kappa_results: dict[str, tuple[float | None, float | None, float | None]] = {}
        for label, field in (
            ("primary", "signal_b"), ("strict", "signal_b_strict"), ("relaxed", "signal_b_relaxed"),
        ):
            b_labels = [row[field] for row in kappa_rows]
            result = bootstrap_kappa_ci(
                a_labels, b_labels, resamples=int(settings["bootstrap_resamples"]),
                confidence=float(settings["bootstrap_confidence"]), seed=int(config["seed"]),
            )
            kappa_results[label] = result
            estimate, lower, upper = result
            logger.info("Cohen kappa (%s B): n=%d | kappa=%s | bootstrap_CI_%g=[%s, %s]",
                        label, len(kappa_rows), f"{estimate:.6f}" if estimate is not None else "NA",
                        float(settings["bootstrap_confidence"]),
                        f"{lower:.6f}" if lower is not None else "NA",
                        f"{upper:.6f}" if upper is not None else "NA")
        kappa, ci_low, ci_high = kappa_results["primary"]

        syllable_bins = report["syllable_bins"]
        if not isinstance(syllable_bins, list) or len(syllable_bins) != 2:
            raise ValueError("etymology.report.syllable_bins must contain two bucket edges")
        stratum_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
        by_pos_bucket: Counter[tuple[str, str, str]] = Counter()
        single_syllable: Counter[tuple[str, str]] = Counter()
        sino_via_counts: Counter[str] = Counter()
        nllb_agreement: Counter[str] = Counter()
        verified_string_count = 0
        verified_wiktionary_count = 0
        for row in enriched:
            split = row["split"]
            stratum_counts[split][row["stratum"]] += 1
            if row["nllb_all_agree"]:
                nllb_agreement[split] += 1
            for check in row["han_verifications"]:
                if check["primary_pass"]:
                    verified_string_count += 1
                    if check["reading_source"] == "wiktionary_char":
                        verified_wiktionary_count += 1
            if split == "test":
                bucket = _syllable_bucket(row["vi_canonical"], syllable_bins)
                by_pos_bucket[(row["pos"], bucket, row["stratum"])] += 1
                if len(vi_orth_key(row["vi_canonical"]).split()) == 1:
                    single_syllable[(row["stratum"], split)] += 1
            sino_via_counts.update(row["sino_via"])
        logger.info("Stratum sizes by split:")
        for split in SPLITS:
            logger.info("%s | %s", split, json.dumps({label: stratum_counts[split][label] for label in STRATA}, sort_keys=True))
        logger.info("TEST stratum counts by POS x syllable bucket:")
        for (pos, bucket, stratum), count in sorted(by_pos_bucket.items()):
            logger.info("%s | %s | %s | %d", pos, bucket, stratum, count)
        logger.info("sino_via counts: %s", json.dumps(dict(sorted(sino_via_counts.items())), sort_keys=True))
        logger.info("TEST native_strict concepts: %d", sum(row["split"] == "test" and row["native_strict"] for row in enriched))
        logger.info("TEST single-syllable counts by stratum:")
        for stratum in STRATA:
            logger.info("%s | %d", stratum, single_syllable[(stratum, "test")])
        logger.info("NLLB robustness subset (all zh/fr/id agree):")
        for split in SPLITS:
            total_split = sum(row["split"] == split for row in enriched)
            logger.info("%s | %d/%d", split, nllb_agreement[split], total_split)
        logger.info("Verified strings relying on Wiktionary character readings: %d/%d (%.6f)",
                    verified_wiktionary_count, verified_string_count,
                    verified_wiktionary_count / verified_string_count if verified_string_count else 0.0)

        sino_failures = [row for row in enriched if row["signal_a"] == "sino" and row["signal_b"] == "nonsino"]
        failure_groups: dict[str, list[dict[str, Any]]] = {reason: [] for reason in FAILURE_REASONS}
        for row in sino_failures:
            reason = row["signal_b_failure_reason"]
            if reason not in failure_groups:
                raise ValueError(f"Signal A=sino/B=nonsino row has unexpected failure reason: {reason!r}")
            failure_groups[reason].append(row)
        example_n = int(report["failure_examples_per_reason"])
        logger.info("A=sino & B=nonsino failure groups; examples (vi | Han strings | reason):")
        for reason in FAILURE_REASONS:
            group = sorted(failure_groups[reason], key=lambda row: row["concept_id"])
            logger.info("%s | count=%d", reason, len(group))
            for row in group[:example_n]:
                logger.info("%s | %s | %s", row["vi_canonical"], json.dumps(row["han_strings"], ensure_ascii=False), reason)

        nonsino_sino = sorted(
            (row for row in enriched if row["signal_a"] == "nonsino" and row["signal_b"] == "sino"),
            key=lambda row: row["concept_id"],
        )
        log_n = int(report["nonsino_sino_examples_max"])
        logger.info("A=nonsino & B=sino | count=%d | examples (vi | Han string | CEDICT gloss):",
                    len(nonsino_sino))
        for row in nonsino_sino[:log_n]:
            verification = next(item for item in row["han_verifications"] if item["failure_reason"] is None)
            logger.info("%s | %s | %s", row["vi_canonical"], verification["han_string"], json.dumps(verification["cedict_glosses"], ensure_ascii=False))

        test_rows = [row for row in enriched if row["split"] == "test"]
        n_sino = sum(row["stratum"] == "sino" for row in test_rows)
        n_nonsino = sum(row["stratum"] == "nonsino" for row in test_rows)
        n_native = sum(row["native_strict"] for row in test_rows)
        n_comparison = int(settings["power_holm_comparisons"])
        alpha = float(settings["power_alpha"])
        target_power = float(settings["power_target"])
        holm_mde_by_comparison: dict[str, float | None] = {}
        logger.info("Power (n1, n2, MDE d):")
        for label, first_n, second_n in (
            ("sino_vs_nonsino", n_sino, n_nonsino),
            ("sino_vs_native_strict", n_sino, n_native),
        ):
            unadjusted = minimum_detectable_difference(first_n, second_n, alpha=alpha, power=target_power)
            adjusted = minimum_detectable_difference(first_n, second_n, alpha=alpha / n_comparison, power=target_power)
            holm_mde_by_comparison[label] = adjusted
            logger.info("%s | n1=%d | n2=%d | alpha=%.6f d=%s | holm_alpha=%.6f d=%s",
                        label, first_n, second_n, alpha,
                        f"{unadjusted:.6f}" if unadjusted is not None else "NA",
                        alpha / n_comparison, f"{adjusted:.6f}" if adjusted is not None else "NA")
        h3_confirmatory = (
            ci_low is not None
            and ci_low >= float(settings["h3_confirmatory_kappa_lower_min"])
            and holm_mde_by_comparison["sino_vs_nonsino"] is not None
            and holm_mde_by_comparison["sino_vs_nonsino"] <= float(settings["h3_confirmatory_holm_mde_max"])
        )
        logger.info("H3 status rule: lower_primary_kappa_CI=%s (minimum=%.2f); Holm MDE=%s (maximum=%.2f); status=%s",
                    f"{ci_low:.6f}" if ci_low is not None else "NA",
                    float(settings["h3_confirmatory_kappa_lower_min"]),
                    f"{holm_mde_by_comparison['sino_vs_nonsino']:.6f}"
                    if holm_mde_by_comparison["sino_vs_nonsino"] is not None else "NA",
                    float(settings["h3_confirmatory_holm_mde_max"]),
                    "confirmatory" if h3_confirmatory else "exploratory")

        additions = [
            pa.field("entry_scope", pa.string()),
            pa.field("signal_a", pa.string()),
            pa.field("signal_a_entry_labels", pa.list_(pa.string())),
            pa.field("sino_via", pa.list_(pa.string())),
            pa.field("other_loan_sources", pa.list_(pa.string())),
            pa.field("native_strict", pa.bool_()),
            pa.field("han_strings", pa.list_(pa.string())),
            pa.field("signal_b", pa.string()),
            pa.field("signal_b_strict", pa.string()),
            pa.field("signal_b_relaxed", pa.string()),
            pa.field("signal_b_failure_reason", pa.string()),
            pa.field("verified_han_string", pa.string()),
            pa.field("verified_cedict_glosses", pa.list_(pa.string())),
            pa.field("han_verifications", pa.list_(pa.struct([
                pa.field("han_string", pa.string()),
                pa.field("reading_ok", pa.bool_()),
                pa.field("reading_source", pa.string()),
                pa.field("headword_found", pa.bool_()),
                pa.field("meaning_overlap", pa.int64()),
                pa.field("primary_pass", pa.bool_()),
                pa.field("strict_pass", pa.bool_()),
                pa.field("relaxed_pass", pa.bool_()),
                pa.field("failure_reason", pa.string()),
                pa.field("cedict_glosses", pa.list_(pa.string())),
            ]))),
            pa.field("stratum", pa.string()),
            pa.field("nllb_all_agree", pa.bool_()),
        ]
        existing = set(table.schema.names)
        output_schema = pa.schema([*table.schema, *(field for field in additions if field.name not in existing)])
        output_path = Path(paths["output"])
        _write_parquet(enriched, output_schema, output_path, int(config["split"]["parquet_row_group_size"]))
        _dropflow_and_log(enriched, config, logger)
        runtime = time.monotonic() - started
        logger.info("Output: %s (%d concepts)", output_path, len(enriched))
        logger.info("Runtime seconds: %.3f", runtime)
        return {"rows": enriched, "crosstab": cross, "kappa": kappa, "kappa_ci": (ci_low, ci_high), "runtime_seconds": runtime}
    except Exception as exc:
        logger.exception("Etymology step stopped: %s", exc)
        logger.info("Runtime seconds: %.3f", time.monotonic() - started)
        raise


def _syllable_bucket(word: str, edges: list[int]) -> str:
    """Map whitespace-separated Vietnamese syllables to the configured report bucket."""
    if len(edges) != 2 or any(type(edge) is not int for edge in edges) or edges != sorted(set(edges)) or edges[0] < 1:
        raise ValueError(f"Syllable bucket edges must be two ascending positive integers, got {edges!r}")
    count = len(vi_orth_key(word).split())
    return str(edges[0]) if count <= edges[0] else str(edges[1]) if count <= edges[1] else f"{edges[1] + 1}+"


def main() -> None:
    """Parse the configured pipeline path and run the etymology step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Path to pipeline YAML config")
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()
