"""Canonicalize aligned candidates and apply the step-06 concept filters."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
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
    from common import DropflowLogger, load_config, normalize_nfc, normalize_strings, normalized_levenshtein, setup_logging, strip_diacritics, vi_orth_key
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, load_config, normalize_nfc, normalize_strings, normalized_levenshtein, setup_logging, strip_diacritics, vi_orth_key


STEP = "06_filter"
SPLITS = ("fewshot_reservoir", "directions", "test")
LANGUAGES = ("zh", "fr", "id")
STAGES = (
    "canonical_missing_lang", "filter5_proper_noun", "filter3_polysemy",
    "filter4_surface", "filter4b_loanword", "filter4c_hyphen",
    "dedup_vi_form", "dedup_en_lemma",
)


def surface_norm(value: str, *, vietnamese: bool, remove_chars: list[str], vi_replace: dict[str, str]) -> str:
    """Normalize a surface form as configured for the surface-distance filter."""
    result = strip_diacritics(normalize_nfc(value)).lower()
    for char in remove_chars:
        result = result.replace(char, "")
    if vietnamese:
        for source, target in sorted(vi_replace.items(), key=lambda item: (-len(item[0]), item[0])):
            result = result.replace(source, target)
    return result


def candidate_words(candidates: list[dict[str, Any]]) -> list[str]:
    """Return unique NFC candidate words in stable input order."""
    if not isinstance(candidates, list):
        raise ValueError(f"Candidate column must be a list, got {type(candidates).__name__}")
    words: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("word"), str):
            raise ValueError(f"Malformed candidate record: {candidate!r}")
        word = normalize_nfc(candidate["word"].strip())
        if not word:
            raise ValueError(f"Empty candidate word: {candidate!r}")
        key = word.casefold()
        if key not in seen:
            seen.add(key)
            words.append(word)
    return words


def summed_vi_senses(candidate: dict[str, Any]) -> int:
    """Sum sense counts over every Vietnamese homograph entry attached to a candidate."""
    values = candidate.get("n_senses_vi")
    n_entries = candidate.get("n_vi_entries")
    if not isinstance(values, list) or not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"Malformed n_senses_vi for candidate {candidate!r}")
    if type(n_entries) is not int or n_entries != len(values):
        raise ValueError(f"n_vi_entries does not match n_senses_vi homograph counts: {candidate!r}")
    return sum(values)


def merge_vi_candidates(
    candidates: list[dict[str, Any]],
    zipf: dict[str, float],
    attestation_sources: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Merge spelling-equivalent VI candidates and select a frequency-ranked display form."""
    source_flags = {
        "wiktextract_table": "in_wiktextract", "vi_gloss": "in_vi_gloss",
        "muse": "in_muse", "wikidata": "in_wikidata",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("word"), str):
            raise ValueError(f"Malformed Vietnamese candidate: {candidate!r}")
        word = normalize_nfc(candidate["word"].strip())
        if not word:
            raise ValueError(f"Empty Vietnamese candidate: {candidate!r}")
        grouped[vi_orth_key(word)].append({**candidate, "word": word})

    merged: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    merged_count = 0
    for key in sorted(grouped):
        variants = grouped[key]
        unique_words = sorted({candidate["word"] for candidate in variants})
        if not key:
            raise ValueError(f"VI candidate normalizes to an empty orthographic key: {variants!r}")

        def display_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
            word = candidate["word"]
            frequency = zipf.setdefault(word, float(zipf_frequency(word, "vi")))
            return (-frequency, int(word.casefold() != key), word.casefold(), word)

        variants.sort(key=display_rank)
        display_candidate = dict(variants[0])
        if len(variants) > 1:
            merged_count += len(variants) - 1
            examples.append({"orth_key": key, "display": display_candidate["word"], "variants": unique_words})
            for field in ("n_senses_vi", "categories", "etymology_templates", "etymology_text", "cjk_forms"):
                values = [candidate.get(field) for candidate in variants]
                if any(not isinstance(value, list) for value in values):
                    raise ValueError(f"VI orthographic variants have malformed {field}: {variants!r}")
                display_candidate[field] = [item for value in values for item in value]
            entry_counts = [candidate.get("n_vi_entries") for candidate in variants]
            if any(type(count) is not int or count < 1 for count in entry_counts):
                raise ValueError(f"VI orthographic variants have malformed n_vi_entries: {variants!r}")
            display_candidate["n_vi_entries"] = sum(entry_counts)
            display_candidate["vi_pos"] = sorted({pos for candidate in variants for pos in candidate.get("vi_pos", [])})
            for flag in ("in_wiktextract", "in_vi_gloss", "in_muse", "in_wikidata"):
                if any(flag in candidate for candidate in variants):
                    if any(type(candidate.get(flag)) is not bool for candidate in variants):
                        raise ValueError(f"VI orthographic variant has malformed {flag}: {variants!r}")
                    display_candidate[flag] = any(candidate[flag] for candidate in variants)
            if any("external_attested" in candidate for candidate in variants):
                display_candidate["external_attested"] = any(bool(candidate.get("external_attested")) for candidate in variants)
            if any("n_sources" in candidate for candidate in variants):
                if not attestation_sources:
                    raise ValueError("Cannot merge attestation counts without filters.attestation_sources")
                display_candidate["n_sources"] = sum(
                    int(bool(display_candidate.get(source_flags[name]))) for name in attestation_sources
                )
            locations = {str(candidate.get("source_location")) for candidate in variants}
            if "both" in locations or {"top", "senses"}.issubset(locations):
                display_candidate["source_location"] = "both"
            elif len(locations) == 1:
                display_candidate["source_location"] = next(iter(locations))

        senses = display_candidate.get("n_senses_vi")
        if not isinstance(senses, list) or not senses or display_candidate.get("n_vi_entries") != len(senses):
            raise ValueError(f"Merged VI candidate has malformed entry/sense counts: {display_candidate!r}")
        merged.append(display_candidate)
        variant_records.append({
            "orth_key": key, "display": display_candidate["word"], "variants": unique_words,
            "n_senses_vi": list(senses), "n_vi_entries": display_candidate["n_vi_entries"],
            "n_sources": display_candidate["n_sources"], "external_attested": display_candidate["external_attested"],
            "loan_templates_found": [],
        })
    return merged, variant_records, merged_count, examples


def canonical_vi(candidates: list[dict[str, Any]], zipf: dict[str, float]) -> tuple[dict[str, Any], list[str], int]:
    """Choose VI form by source evidence, polysemy, frequency, length, and spelling."""
    if not candidates:
        raise ValueError("Cannot canonicalize an empty Vietnamese candidate list")
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("word"), str):
            raise ValueError(f"Malformed Vietnamese candidate: {candidate!r}")
        word = normalize_nfc(candidate["word"].strip())
        if not word:
            raise ValueError(f"Empty Vietnamese candidate: {candidate!r}")
        unique.setdefault(vi_orth_key(word), candidate)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any], str, int]] = []
    for candidate in unique.values():
        word = normalize_nfc(candidate["word"].strip())
        n_sources = candidate.get("n_sources")
        external = candidate.get("external_attested")
        if type(n_sources) is not int or not isinstance(external, bool):
            raise ValueError(f"Missing attestation flags on VI candidate: {candidate!r}")
        sense_count = summed_vi_senses(candidate)
        frequency = zipf.setdefault(word, float(zipf_frequency(word, "vi")))
        ranked.append(((-n_sources, -int(external), sense_count, -frequency, len(word), word.casefold(), word), candidate, word, sense_count))
    ranked.sort(key=lambda item: item[0])
    _, selected, word, senses = ranked[0]
    alts = [item[2] for item in ranked[1:]]
    return selected, alts, senses


def has_latin_or_digit(value: str) -> bool:
    """Whether a candidate contains any Latin-script letter or Unicode digit."""
    import unicodedata

    return any(char.isdigit() or (char.isalpha() and "LATIN" in unicodedata.name(char, "")) for char in value)


def canonical_language(words: list[str], *, language: str, pos: str, config: dict[str, Any], zipf: dict[str, float]) -> tuple[str | None, list[str], bool, int]:
    """Filter and provisionally rank one non-Vietnamese language's candidates."""
    filters = config["filters"]
    zh_removed = 0
    candidates = list(words)
    derived = False
    if language == "zh" and filters["zh_forbid_latin"]:
        kept = [word for word in candidates if not has_latin_or_digit(word)]
        zh_removed = len(candidates) - len(kept)
        candidates = kept
    alternate_candidates = list(candidates)
    if language == "zh" and pos == "adj" and filters["zh_adj_strip_de"] and candidates:
        plain = [word for word in candidates if not word.endswith("的")]
        if plain:
            candidates = plain
        else:
            candidates = [word[:-1] for word in candidates if word.endswith("的") and word[:-1]]
            derived = True
    candidates = list(dict.fromkeys(normalize_nfc(word.strip()) for word in candidates if word.strip()))
    if not candidates:
        return None, [], derived, zh_removed
    frequency_language = language
    ranked = sorted(candidates, key=lambda word: (-zipf.setdefault(f"{language}\0{word}", float(zipf_frequency(word, frequency_language))), len(word), word))
    if language == "zh" and pos == "adj" and not derived:
        alt_words = [word for word in alternate_candidates if word != ranked[0]]
        alt_words.extend(word for word in ranked[1:] if word not in alt_words)
        return ranked[0], alt_words, derived, zh_removed
    return ranked[0], ranked[1:], derived, zh_removed


def loan_templates_found(candidate: dict[str, Any], template_names: set[str]) -> list[dict[str, str | None]]:
    """Capture configured loan-template name, source language, and source term."""
    found: list[dict[str, str | None]] = []
    groups = candidate.get("etymology_templates")
    if not isinstance(groups, list):
        raise ValueError(f"Malformed etymology_templates: {groups!r}")
    for group in groups:
        if not isinstance(group, list):
            raise ValueError(f"Malformed homograph template group: {group!r}")
        for template in group:
            if not isinstance(template, dict) or not isinstance(template.get("name"), str):
                raise ValueError(f"Malformed etymology template: {template!r}")
            if template["name"] not in template_names:
                continue
            args_json = template.get("args_json")
            if not isinstance(args_json, str):
                raise ValueError(f"Loan template is missing args_json: {template!r}")
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid args_json on loan template: {template!r}") from exc
            if not isinstance(args, dict) or not isinstance(args.get("2"), str) or not args["2"].strip():
                raise ValueError(f"Loan template has no string source-language argument at position 2: {template!r}")
            if args.get("1") != "vi":
                raise ValueError(f"Expected Vietnamese target language in loan template argument 1: {template!r}")
            source = normalize_nfc(args["2"].strip()).lower()
            source_term = args.get("3")
            if source_term is not None and not isinstance(source_term, str):
                raise ValueError(f"Loan template argument 3 must be a string or missing: {template!r}")
            found.append({
                "template": normalize_nfc(template["name"]),
                "source_lang": source,
                "args3": normalize_nfc(source_term) if source_term is not None else None,
            })
    return sorted(found, key=lambda item: (item["template"] or "", item["source_lang"] or "", item["args3"] or ""))


def source_matches_drop_list(source: str, exact: set[str], prefixes: list[str]) -> bool:
    """Match a normalized source code against exact codes and configured prefixes."""
    normalized = normalize_nfc(source).strip().lower()
    return normalized in exact or any(normalized.startswith(prefix.lower()) for prefix in prefixes)


def is_hyphenated_transliteration(word: str) -> bool:
    """Return whether a canonical Vietnamese form contains an ASCII hyphen."""
    return "-" in normalize_nfc(word)


def old_rule_dropped(records: list[dict[str, str | None]], *, template_names: set[str], chinese_codes: set[str], chinese_prefix: str) -> list[str]:
    """Return source codes the previous non-Chinese rule would have dropped."""
    return sorted({
        str(record["source_lang"])
        for record in records
        if record["template"] in template_names
        and record["source_lang"] not in chinese_codes
        and not str(record["source_lang"]).startswith(chinese_prefix)
    })


def surface_comparisons(vi: str, words_by_lang: dict[str, list[str]], *, norm_config: dict[str, Any]) -> tuple[float, str, str, float]:
    """Return best configured distance and its paired raw diacritic-only distance."""
    vi_norm = surface_norm(vi, vietnamese=True, **norm_config)
    comparisons: list[tuple[float, int, str, str, float]] = []
    for lang_index, language in enumerate(("en", "fr", "id")):
        for form in words_by_lang[language]:
            form_norm = surface_norm(form, vietnamese=False, **norm_config)
            distance = normalized_levenshtein(vi_norm, form_norm)
            vi_raw = strip_diacritics(vi).lower()
            form_raw = strip_diacritics(form).lower()
            raw_distance = normalized_levenshtein(vi_raw, form_raw)
            comparisons.append((distance, lang_index, language, form, raw_distance))
    if not comparisons:
        raise ValueError("Surface comparison requires at least one en/fr/id candidate")
    distance, _, language, form, raw_distance = min(comparisons)
    return distance, language, form, raw_distance


def _quantile(values: list[int], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate polysemy cutoff from an empty test POS stratum")
    return float(np.quantile(np.asarray(values, dtype=np.float64), percentile / 100.0))


def syllable_bucket(word: str, edges: list[int]) -> str:
    """Bucket a form by whitespace-delimited syllable count using configured edges."""
    if len(edges) != 2 or any(type(edge) is not int for edge in edges) or edges != sorted(set(edges)) or edges[0] < 1:
        raise ValueError(f"split.syllable_bins must be two ascending positive integers, got {edges!r}")
    count = len(normalize_nfc(word).strip().split())
    if count <= edges[0]:
        return str(edges[0])
    if count <= edges[1]:
        return str(edges[1])
    return f"{edges[1] + 1}+"


def _counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row["pos"] for row in rows)


def _dropflow_without_step(path: Path, step: str) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = []
    for line_no, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed dropflow JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("step"), str):
            raise ValueError(f"Unexpected dropflow record at {path}:{line_no}")
        if record["step"] != step:
            kept.append(line if line.endswith("\n") else line + "\n")
    path.write_text("".join(kept), encoding="utf-8", newline="\n")


def _output_schema(schema: pa.Schema) -> pa.Schema:
    vi_variant = pa.struct([
        pa.field("orth_key", pa.string()), pa.field("display", pa.string()),
        pa.field("variants", pa.list_(pa.string())), pa.field("n_senses_vi", pa.list_(pa.int32())),
        pa.field("n_vi_entries", pa.int32()), pa.field("n_sources", pa.int8()),
        pa.field("external_attested", pa.bool_()),
        pa.field("loan_templates_found", pa.list_(pa.struct([
            pa.field("template", pa.string()), pa.field("source_lang", pa.string()), pa.field("args3", pa.string()),
        ]))),
    ])
    additions = [
        pa.field("vi_canonical", pa.string()), pa.field("vi_alts", pa.list_(pa.string())),
        pa.field("vi_variants", pa.list_(vi_variant)), pa.field("polysemy_cutoff", pa.float64()),
        pa.field("n_senses_vi", pa.int32()), pa.field("n_vi_entries", pa.int32()),
        pa.field("n_sources", pa.int8()), pa.field("external_attested", pa.bool_()),
        pa.field("zh_canonical", pa.string()), pa.field("zh_alts", pa.list_(pa.string())),
        pa.field("zh_derived", pa.bool_()), pa.field("zh_canonical_provisional", pa.bool_()),
        pa.field("fr_canonical", pa.string()), pa.field("fr_alts", pa.list_(pa.string())),
        pa.field("fr_canonical_provisional", pa.bool_()),
        pa.field("id_canonical", pa.string()), pa.field("id_alts", pa.list_(pa.string())),
        pa.field("id_canonical_provisional", pa.bool_()),
        pa.field("min_surface_dist", pa.float64()), pa.field("closest_lang", pa.string()),
        pa.field("closest_form", pa.string()), pa.field("raw_surface_dist", pa.float64()),
        pa.field("loan_templates_found", pa.list_(pa.struct([
            pa.field("template", pa.string()), pa.field("source_lang", pa.string()), pa.field("args3", pa.string()),
        ]))),
    ]
    existing = set(schema.names)
    return pa.schema([*schema, *(field for field in additions if field.name not in existing)])


def _loan_audit_schema() -> pa.Schema:
    return pa.schema([
        pa.field("concept_id", pa.string()), pa.field("split", pa.string()),
        pa.field("en_lemma", pa.string()), pa.field("pos", pa.string()),
        pa.field("vi_canonical", pa.string()),
        pa.field("loan_templates_found", pa.list_(pa.struct([
            pa.field("template", pa.string()), pa.field("source_lang", pa.string()), pa.field("args3", pa.string()),
        ]))),
        pa.field("reached_filter4b", pa.bool_()), pa.field("loanword_dropped", pa.bool_()),
    ])


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path, row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as temp:
            temp_name = temp.name
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temp_name, compression="zstd", version="2.6", row_group_size=row_group_size, write_statistics=True)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _rank_dedup(row: dict[str, Any], en_zipf: dict[str, float]) -> tuple[Any, ...]:
    lemma = normalize_nfc(row["en_lemma"])
    return (-row["n_sources"], -int(row["external_attested"]), row["n_senses_vi"], -en_zipf.setdefault(lemma, float(zipf_frequency(lemma, "en"))), row["concept_id"])


def _deduplicate(rows: list[dict[str, Any]], field: str, en_zipf: dict[str, float]) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row["vi_canonical"] if field == "vi_canonical" else row["en_lemma"]
        key = vi_orth_key(value) if field == "vi_canonical" else normalize_nfc(value.strip()).casefold()
        grouped[key].append(row)
    keep: list[dict[str, Any]] = []
    actions: list[tuple[str, str, str]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: _rank_dedup(row, en_zipf))
        keep.append(group[0])
        actions.extend((field, group[0]["concept_id"], row["concept_id"]) for row in group[1:])
    return sorted(keep, key=lambda row: row["concept_id"]), actions


def assert_split_disjointness(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    """Assert concept IDs, all VI candidates, and English lemmas remain split-exclusive."""
    seen: dict[str, dict[str, str]] = {"concept_id": {}, "vi": {}, "en": {}}
    for split in SPLITS:
        for row in rows_by_split[split]:
            terms = {
                "concept_id": [normalize_nfc(row["concept_id"]).casefold()],
                "en": [normalize_nfc(row["en_lemma"]).casefold()],
                "vi": [vi_orth_key(candidate["word"]) for candidate in row["vi_cands"]],
            }
            for kind, values in terms.items():
                for value in values:
                    previous = seen[kind].setdefault(value, split)
                    if previous != split:
                        raise AssertionError(f"{kind} value {value!r} appears in splits {previous!r} and {split!r}")


def run(config_path: str) -> dict[str, Any]:
    """Run the canonicalization and filtering stages for every held-out split."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    filters = config["filters"]
    settings = config["filter"]
    try:
        progress_every = int(config["logging"]["progress_every"])
        threshold = float(filters["surface_edit_threshold"])
        percentile = float(filters["polysemy_percentile"])
        row_group_size = int(config["split"]["parquet_row_group_size"])
        syllable_edges = config["split"]["syllable_bins"]
        report = filters["report"]
        report_bin = float(report["surface_histogram_bin_width"])
        near_n = int(report["surface_near_threshold_n"])
        dedup_examples_n = int(report["dedup_examples_n"])
        hyphen_examples_n = int(report["hyphen_examples_n"])
        orth_merge_examples_n = int(report["orth_merge_examples_n"])
        if progress_every < 1 or row_group_size < 1 or not 0 <= threshold <= 1 or not 0 <= percentile <= 100 or report_bin <= 0 or near_n < 0 or dedup_examples_n < 0 or hyphen_examples_n < 0 or orth_merge_examples_n < 0:
            raise ValueError("Invalid step-06 numeric setting in config")
        if filters["polysemy_count"] != "sum_over_homograph_entries":
            raise ValueError(f"Unsupported filters.polysemy_count: {filters['polysemy_count']!r}")
        if filters["surface_compare"] != "all_candidates":
            raise ValueError(f"Unsupported filters.surface_compare: {filters['surface_compare']!r}")
        if tuple(filters["surface_compare_langs"]) != ("en", "fr", "id"):
            raise ValueError("filters.surface_compare_langs must be [en, fr, id]")
        norm_config = filters["surface_norm"]
        if not isinstance(norm_config.get("remove_chars"), list) or not isinstance(norm_config.get("vi_replace"), dict):
            raise ValueError("filters.surface_norm requires remove_chars list and vi_replace mapping")
        loan_settings = filters["loanword"]
        loan_templates = set(loan_settings["loan_templates"])
        drop_source_langs = {code.lower() for code in loan_settings["drop_source_langs"]}
        drop_source_prefixes = [prefix.lower() for prefix in loan_settings["drop_source_lang_prefixes"]]
        previous_rule = loan_settings["previous_rule_for_comparison"]
        previous_templates = set(previous_rule["loan_templates"])
        previous_chinese = {code.lower() for code in previous_rule["chinese_codes"]}
        previous_chinese_prefix = previous_rule["chinese_prefix"].lower()
        if not loan_templates or not drop_source_langs or not drop_source_prefixes:
            raise ValueError("Loanword templates, drop source codes, and prefixes must be non-empty")

        input_path = Path(settings["paths"]["input"])
        table = pq.read_table(input_path)
        required = {"concept_id", "en_lemma", "pos", "split", "vi_cands", "zh_cands", "fr_cands", "id_cands"}
        missing = sorted(required - set(table.column_names))
        if missing:
            raise ValueError(f"Step-05 parquet is missing required columns: {missing}")
        rows = [normalize_strings(row) for row in table.to_pylist()]
        if not rows:
            raise ValueError("Step-05 parquet has no concepts")
        if any(row["split"] not in SPLITS for row in rows):
            raise ValueError(f"Unexpected split labels: {sorted({row['split'] for row in rows})}")
        if len({row["concept_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate concept_id in step-05 input")
        logger.info("Loaded %d concepts from %s", len(rows), input_path)

        zipf_cache: dict[str, float] = {}
        prepared: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        zh_latin_removed: Counter[str] = Counter()
        loan_info_by_id: dict[str, tuple[str, list[dict[str, str | None]], list[dict[str, str | None]]]] = {}
        vi_merged_candidate_count = 0
        vi_merge_examples: list[dict[str, Any]] = []
        for index, original in enumerate(sorted(rows, key=lambda row: row["concept_id"]), start=1):
            row = dict(original)
            split = row["split"]
            try:
                merged_vi, variant_records, merged_count, merge_examples = merge_vi_candidates(
                    row["vi_cands"], zipf_cache, filters["attestation_sources"],
                )
                vi_merged_candidate_count += merged_count
                vi_merge_examples.extend({
                    "concept_id": row["concept_id"], "en_lemma": row["en_lemma"], "split": split, **example,
                } for example in merge_examples)
                selected, vi_alts, vi_senses = canonical_vi(merged_vi, zipf_cache)
                vi_word = normalize_nfc(selected["word"].strip())
                row.update({
                    "vi_canonical": vi_word, "vi_alts": vi_alts, "n_senses_vi": vi_senses,
                    "n_vi_entries": selected["n_vi_entries"], "n_sources": selected["n_sources"],
                    "external_attested": selected["external_attested"], "vi_variants": variant_records,
                })
                for group, candidate in zip(variant_records, merged_vi, strict=True):
                    group["loan_templates_found"] = loan_templates_found(candidate, loan_templates)
                selected_group = next(group for group in variant_records if group["orth_key"] == vi_orth_key(vi_word))
                row["loan_templates_found"] = selected_group["loan_templates_found"]
                selected_raw = next(candidate for candidate in merged_vi if vi_orth_key(candidate["word"]) == vi_orth_key(vi_word))
                previous_records = loan_templates_found(selected_raw, previous_templates)
                loan_info_by_id[row["concept_id"]] = (vi_word, row["loan_templates_found"], previous_records)
                missing = []
                for language in LANGUAGES:
                    words = candidate_words(row[f"{language}_cands"])
                    canonical, alts, derived, removed = canonical_language(words, language=language, pos=row["pos"], config=config, zipf=zipf_cache)
                    zh_latin_removed[split] += removed
                    row[f"{language}_canonical"] = canonical
                    row[f"{language}_alts"] = alts
                    row[f"{language}_canonical_provisional"] = language != "zh" or canonical is not None
                    if language == "zh":
                        row["zh_derived"] = derived
                    if canonical is None:
                        missing.append(language)
                if missing:
                    continue
                row["zh_canonical_provisional"] = True
                prepared[split].append(row)
            except Exception as exc:
                raise ValueError(f"Step-06 canonicalization failed for concept {row.get('concept_id')!r}: {exc}") from exc
            if index % progress_every == 0 or index == len(rows):
                logger.info("Canonicalized concepts: %d/%d", index, len(rows))

        logger.info("Vietnamese candidates merged by vi_orth_key: %d", vi_merged_candidate_count)
        logger.info("VI orthographic-merge examples (en | orth_key | display | merged variants):")
        for example in sorted(vi_merge_examples, key=lambda item: (item["concept_id"], item["orth_key"]))[:orth_merge_examples_n]:
            logger.info("%s | %s | %s | %s", example["en_lemma"], example["orth_key"], example["display"], json.dumps(example["variants"], ensure_ascii=False))

        stages: dict[str, dict[str, list[dict[str, Any]]]] = {stage: {} for stage in STAGES}
        for split in SPLITS:
            input_rows = [row for row in rows if row["split"] == split]
            stages["canonical_missing_lang"][split] = prepared[split]
            logger.info("zh Latin/digit candidates removed (%s): %d", split, zh_latin_removed[split])

        # Proper-noun filter precedes TEST-only threshold estimation as specified.
        for split in SPLITS:
            stages["filter5_proper_noun"][split] = [
                row for row in prepared[split]
                if not row["vi_canonical"][:1].isupper()
                and not row["en_lemma"][:1].isupper()
                and row["pos"].casefold() not in {"name", "proper noun"}
            ]

        test_for_cutoff = stages["filter5_proper_noun"]["test"]
        scores_by_pos: dict[str, list[int]] = defaultdict(list)
        for row in test_for_cutoff:
            scores_by_pos[row["pos"]].append(row["n_senses_vi"])
        cutoffs: dict[str, float] = {}
        cutoff_n: dict[str, int] = {}
        for pos in sorted({row["pos"] for split in SPLITS for row in stages["filter5_proper_noun"][split]}):
            if pos not in scores_by_pos:
                raise ValueError(f"No test concepts for POS {pos!r}; cannot compute TEST-only polysemy cutoff")
            cutoffs[pos] = _quantile(scores_by_pos[pos], percentile)
            cutoff_n[pos] = sum(value == cutoffs[pos] for value in scores_by_pos[pos])
            logger.info("Polysemy TEST cutoff | pos=%s | percentile=%s | cutoff=%s | exactly_at_cutoff=%d", pos, percentile, cutoffs[pos], cutoff_n[pos])
        for split in SPLITS:
            for row in stages["filter5_proper_noun"][split]:
                row["polysemy_cutoff"] = cutoffs[row["pos"]]
        all_sense_distribution = Counter(row["n_senses_vi"] for split in SPLITS for row in stages["filter5_proper_noun"][split])
        logger.info("Canonical Vietnamese n_senses_vi distribution before polysemy filter: %s", json.dumps(dict(sorted(all_sense_distribution.items())), sort_keys=True))
        for split in SPLITS:
            stages["filter3_polysemy"][split] = [row for row in stages["filter5_proper_noun"][split] if row["n_senses_vi"] <= cutoffs[row["pos"]]]

        surface_details: dict[str, dict[str, tuple[float, str, str, float]]] = defaultdict(dict)
        for split in SPLITS:
            kept = []
            for row in stages["filter3_polysemy"][split]:
                words = {"en": [normalize_nfc(row["en_lemma"])], **{
                    language: candidate_words(row[f"{language}_cands"]) for language in ("fr", "id")
                }}
                comparison = surface_comparisons(row["vi_canonical"], words, norm_config=norm_config)
                row.update({"min_surface_dist": comparison[0], "closest_lang": comparison[1], "closest_form": comparison[2], "raw_surface_dist": comparison[3]})
                surface_details[split][row["concept_id"]] = comparison
                if comparison[0] < threshold:
                    continue
                else:
                    kept.append(row)
            stages["filter4_surface"][split] = kept

        loan_counts: Counter[str] = Counter()
        recovered_at_loan_stage: Counter[str] = Counter()
        loan_examples: list[dict[str, Any]] = []
        loanword_dropped_ids: set[str] = set()
        e2_eligible_ids = {row["concept_id"] for split in SPLITS for row in stages["filter4_surface"][split]}
        for split in SPLITS:
            kept = []
            for row in stages["filter4_surface"][split]:
                records = row["loan_templates_found"]
                sources = sorted({str(record["source_lang"]) for record in records if source_matches_drop_list(str(record["source_lang"]), drop_source_langs, drop_source_prefixes)})
                old_sources = old_rule_dropped(
                    loan_info_by_id[row["concept_id"]][2], template_names=previous_templates,
                    chinese_codes=previous_chinese, chinese_prefix=previous_chinese_prefix,
                )
                for source in sources:
                    loan_counts[source] += 1
                if old_sources and not sources:
                    for source in old_sources:
                        recovered_at_loan_stage[source] += 1
                row["loanword_dropped"] = bool(sources)
                if sources:
                    loanword_dropped_ids.add(row["concept_id"])
                    loan_examples.append(row)
                else:
                    kept.append(row)
            stages["filter4b_loanword"][split] = kept

        hyphen_dropped: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        for split in SPLITS:
            kept = []
            for row in stages["filter4b_loanword"][split]:
                if is_hyphenated_transliteration(row["vi_canonical"]):
                    hyphen_dropped[split].append(row)
                else:
                    kept.append(row)
            stages["filter4c_hyphen"][split] = kept

        loan_audit_rows = []
        for row in sorted(rows, key=lambda item: item["concept_id"]):
            info = loan_info_by_id.get(row["concept_id"])
            vi_word, records = (info[0], info[1]) if info else (None, [])
            loan_audit_rows.append({
                "concept_id": row["concept_id"], "split": row["split"],
                "en_lemma": row["en_lemma"], "pos": row["pos"],
                "vi_canonical": vi_word, "loan_templates_found": records,
                "reached_filter4b": row["concept_id"] in e2_eligible_ids,
                "loanword_dropped": row["concept_id"] in loanword_dropped_ids,
            })

        dedup_actions: list[tuple[str, str, str, str]] = []
        final_by_split: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            after_vi, vi_actions = _deduplicate(stages["filter4c_hyphen"][split], "vi_canonical", zipf_cache)
            dedup_actions.extend((split, *action) for action in vi_actions)
            after_en, en_actions = _deduplicate(after_vi, "en_lemma", zipf_cache)
            dedup_actions.extend((split, *action) for action in en_actions)
            final_by_split[split] = after_en

        assert_split_disjointness(final_by_split)

        flow = DropflowLogger(config["paths"]["dropflow"])
        flow_counts: dict[str, dict[str, tuple[int, Counter[str], int, Counter[str]]]] = {}
        for split in SPLITS:
            stage_inputs = {
                "canonical_missing_lang": [row for row in rows if row["split"] == split],
                "filter5_proper_noun": stages["canonical_missing_lang"][split],
                "filter3_polysemy": stages["filter5_proper_noun"][split],
                "filter4_surface": stages["filter3_polysemy"][split],
                "filter4b_loanword": stages["filter4_surface"][split],
                "filter4c_hyphen": stages["filter4b_loanword"][split],
                "dedup_vi_form": stages["filter4c_hyphen"][split],
                "dedup_en_lemma": _deduplicate(stages["filter4c_hyphen"][split], "vi_canonical", zipf_cache)[0],
            }
            stage_outputs = {
                **stages,
                "dedup_vi_form": {split: _deduplicate(stages["filter4c_hyphen"][split], "vi_canonical", zipf_cache)[0]},
                "dedup_en_lemma": {split: final_by_split[split]},
            }
            for stage in STAGES:
                before, after = stage_inputs[stage], stage_outputs[stage][split]
                flow_counts.setdefault(split, {})[stage] = (len(before), _counts(before), len(after), _counts(after))

        output_rows = sorted((row for split in SPLITS for row in final_by_split[split]), key=lambda row: row["concept_id"])
        output_schema = _output_schema(table.schema)
        output_path = Path(settings["paths"]["output"])
        audit_path = Path(settings["paths"]["loanword_audit"])
        previous_output_ids: set[str] | None = None
        if output_path.exists():
            previous_output_ids = set(pq.read_table(output_path, columns=["concept_id"]).column("concept_id").to_pylist())
        _write_parquet(loan_audit_rows, _loan_audit_schema(), audit_path, row_group_size)
        _write_parquet(output_rows, output_schema, output_path, row_group_size)
        recovered_final = [] if previous_output_ids is None else [row for row in output_rows if row["concept_id"] not in previous_output_ids]
        recovered_final_by_source: Counter[str] = Counter()
        for row in recovered_final:
            previous_sources = old_rule_dropped(
                loan_info_by_id[row["concept_id"]][2], template_names=previous_templates,
                chinese_codes=previous_chinese, chinese_prefix=previous_chinese_prefix,
            )
            if previous_sources:
                recovered_final_by_source.update(previous_sources)
            else:
                recovered_final_by_source["not-attributed-to-previous-e2"] += 1

        dropflow_path = Path(config["paths"]["dropflow"])
        _dropflow_without_step(dropflow_path, "06")
        for split in SPLITS:
            for stage in STAGES:
                n_in, in_pos, n_out, out_pos = flow_counts[split][stage]
                flow.record(step="06", stage=f"{stage}_{split}", unit="concepts", n_in=n_in, n_out=n_out, n_out_by_pos=dict(sorted(out_pos.items())))

        logger.info("Dropflow (unit=concepts) by split and stage:")
        for split in SPLITS:
            for stage in STAGES:
                n_in, _, n_out, out_pos = flow_counts[split][stage]
                logger.info("%s | %s | n_in=%d | n_out=%d | n_out_by_pos=%s", split, stage, n_in, n_out, json.dumps(dict(sorted(out_pos.items())), sort_keys=True))
        logger.info("filter4c_hyphen dropped concepts by split: %s", json.dumps({split: len(hyphen_dropped[split]) for split in SPLITS}, sort_keys=True))
        logger.info("Hyphen-filter examples (vi | en | pos | split):")
        hyphen_examples = sorted(
            ((split, row) for split in SPLITS for row in hyphen_dropped[split]),
            key=lambda item: item[1]["concept_id"],
        )[:hyphen_examples_n]
        for split, row in hyphen_examples:
            logger.info("%s | %s | %s | %s", row["vi_canonical"], row["en_lemma"], row["pos"], split)
        logger.info("filter4b_loanword dropped concepts by source language: %s", json.dumps(dict(sorted(loan_counts.items())), ensure_ascii=False, sort_keys=True))
        logger.info("Recovered at filter4b vs previous rule by source language: %s", json.dumps(dict(sorted(recovered_at_loan_stage.items())), ensure_ascii=False, sort_keys=True))
        logger.info("Additional final-output concepts vs previous parquet: %d; by previous-rule source language: %s", len(recovered_final), json.dumps(dict(sorted(recovered_final_by_source.items())), ensure_ascii=False, sort_keys=True))
        logger.info("Loanword audit: %s (%d input concepts)", audit_path, len(loan_audit_rows))
        logger.info("Concepts dropped by e2 that surface filter alone would keep (up to 15): vi | source | distance")
        for row in sorted(loan_examples, key=lambda item: item["concept_id"])[:int(report["loanword_examples_n"])]:
            sources = sorted({str(item["source_lang"]) for item in row["loan_templates_found"] if source_matches_drop_list(str(item["source_lang"]), drop_source_langs, drop_source_prefixes)})
            logger.info("%s | %s | %.6f", row["vi_canonical"], ",".join(sources), row["min_surface_dist"])

        # Surface diagnostic on TEST rows after proper-noun/polysemy stages and before surface filtering.
        test_surface = [row for row in stages["filter3_polysemy"]["test"]]
        max_bin = math.ceil(1.0 / report_bin)
        hist = Counter(min(max_bin - 1, int(row["min_surface_dist"] / report_bin)) for row in test_surface)
        logger.info("TEST min_surface_dist histogram (bin_width=%s):", report_bin)
        for bucket in range(max_bin):
            low, high = bucket * report_bin, min(1.0, (bucket + 1) * report_bin)
            logger.info("[%.2f, %.2f%s | %d", low, high, "]" if high >= 1 else ")", hist[bucket])
        at_threshold = sum(row["min_surface_dist"] == threshold for row in test_surface)
        logger.info("TEST concepts exactly at surface threshold %.2f: %d", threshold, at_threshold)
        above = sorted((row for row in test_surface if row["min_surface_dist"] > threshold), key=lambda row: (row["min_surface_dist"] - threshold, row["concept_id"]))[:near_n]
        below = sorted((row for row in test_surface if row["min_surface_dist"] < threshold), key=lambda row: (threshold - row["min_surface_dist"], row["concept_id"]))[:near_n]
        for label, selected in (("ABOVE", above), ("BELOW", below)):
            logger.info("%d TEST concepts %s threshold %.2f (vi | closest lang:form | min_distance | raw_distance):", len(selected), label, threshold)
            for row in selected:
                logger.info("%s | %s:%s | %.6f | %.6f", row["vi_canonical"], row["closest_lang"], row["closest_form"], row["min_surface_dist"], row["raw_surface_dist"])

        logger.info("Dedup removals by split:")
        for split in SPLITS:
            logger.info("%s | dedup_vi_form=%d | dedup_en_lemma=%d", split,
                        sum(action[0] == split and action[1] == "vi_canonical" for action in dedup_actions),
                        sum(action[0] == split and action[1] == "en_lemma" for action in dedup_actions))
        logger.info("Dedup examples (split | field | kept | dropped):")
        for split, field, kept, dropped in dedup_actions[:dedup_examples_n]:
            logger.info("%s | %s | %s | %s", split, field, kept, dropped)

        logger.info("TEST split size after step 06 by POS x canonical-VI syllable bucket:")
        syllable_edges = config["split"]["syllable_bins"]
        stratum_counts = Counter((row["pos"], syllable_bucket(row["vi_canonical"], syllable_edges)) for row in final_by_split["test"])
        for (pos, bucket), count in sorted(stratum_counts.items()):
            logger.info("%s | %s | %d", pos, bucket, count)
        reserve = final_by_split["fewshot_reservoir"]
        logger.info("Few-shot reservoir after step 06: %d concepts", len(reserve))
        logger.info("Few-shot reservoir: en | vi | zh | fr | id")
        for row in sorted(reserve, key=lambda item: item["concept_id"]):
            logger.info("%s | %s | %s | %s | %s", row["en_lemma"], row["vi_canonical"], row["zh_canonical"], row["fr_canonical"], row["id_canonical"])
        logger.info("Output: %s (%d concepts)", output_path, len(output_rows))
        runtime = time.monotonic() - started
        logger.info("Runtime seconds: %.3f", runtime)
        return {"rows": output_rows, "dropflow": flow_counts, "cutoffs": cutoffs, "loanword_counts": loan_counts, "dedup_actions": dedup_actions, "runtime_seconds": runtime}
    except Exception:
        logger.exception("Step 06 failed; output was not intentionally advanced")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
