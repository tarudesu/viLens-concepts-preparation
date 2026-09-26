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
    from common import DropflowLogger, load_config, normalize_nfc, normalize_strings, normalized_levenshtein, setup_logging, strip_diacritics
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, load_config, normalize_nfc, normalize_strings, normalized_levenshtein, setup_logging, strip_diacritics


STEP = "06_filter"
SPLITS = ("fewshot_reservoir", "directions", "test")
LANGUAGES = ("zh", "fr", "id")
STAGES = (
    "canonical_missing_lang", "filter5_proper_noun", "filter3_polysemy",
    "filter4_surface", "filter4b_loanword", "dedup_vi_form", "dedup_en_lemma",
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
        unique.setdefault(word.casefold(), candidate)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any], str, int]] = []
    for candidate in unique.values():
        word = normalize_nfc(candidate["word"].strip())
        n_sources = candidate.get("n_sources")
        external = candidate.get("external_attested")
        if type(n_sources) is not int or not isinstance(external, bool):
            raise ValueError(f"Missing attestation flags on VI candidate: {candidate!r}")
        sense_count = summed_vi_senses(candidate)
        frequency = zipf.setdefault(word, float(zipf_frequency(word, "vi")))
        ranked.append(((-n_sources, -int(external), sense_count, -frequency, len(word), word), candidate, word, sense_count))
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


def loan_source_languages(candidate: dict[str, Any], template_names: set[str], chinese_codes: set[str]) -> list[str]:
    """Return distinct non-Chinese source languages in configured VI loan templates."""
    output: set[str] = set()
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
            if source not in chinese_codes and not source.startswith("zh-"):
                output.add(source)
    return sorted(output)


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
    additions = [
        pa.field("vi_canonical", pa.string()), pa.field("vi_alts", pa.list_(pa.string())),
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
        pa.field("loan_source_lang", pa.list_(pa.string())),
    ]
    existing = set(schema.names)
    return pa.schema([*schema, *(field for field in additions if field.name not in existing)])


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
        key = normalize_nfc(value.strip()).casefold()
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
                "vi": [normalize_nfc(candidate["word"]).casefold() for candidate in row["vi_cands"]],
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
        if progress_every < 1 or row_group_size < 1 or not 0 <= threshold <= 1 or not 0 <= percentile <= 100 or report_bin <= 0 or near_n < 0 or dedup_examples_n < 0:
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
        loan_templates = set(filters["loanword_templates"])
        chinese_codes = {code.lower() for code in filters["loanword_chinese_codes"]}
        if not loan_templates or not chinese_codes:
            raise ValueError("Loanword template and Chinese-code config lists must be non-empty")

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
        for index, original in enumerate(sorted(rows, key=lambda row: row["concept_id"]), start=1):
            row = dict(original)
            split = row["split"]
            try:
                selected, vi_alts, vi_senses = canonical_vi(row["vi_cands"], zipf_cache)
                vi_word = normalize_nfc(selected["word"].strip())
                row.update({
                    "vi_canonical": vi_word, "vi_alts": vi_alts, "n_senses_vi": vi_senses,
                    "n_vi_entries": selected["n_vi_entries"], "n_sources": selected["n_sources"],
                    "external_attested": selected["external_attested"],
                })
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
        loan_examples: list[dict[str, Any]] = []
        for split in SPLITS:
            kept = []
            for row in stages["filter4_surface"][split]:
                # Canonical VI deduplicates by spelling; scan every same-spelling homograph candidate as required.
                same_form = [candidate for candidate in row["vi_cands"] if candidate["word"].casefold() == row["vi_canonical"].casefold()]
                sources = sorted({source for candidate in same_form for source in loan_source_languages(candidate, loan_templates, chinese_codes)})
                row["loan_source_lang"] = sources
                if sources:
                    for source in sources:
                        loan_counts[source] += 1
                    loan_examples.append(row)
                else:
                    kept.append(row)
            stages["filter4b_loanword"][split] = kept

        dedup_actions: list[tuple[str, str, str, str]] = []
        final_by_split: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            after_vi, vi_actions = _deduplicate(stages["filter4b_loanword"][split], "vi_canonical", zipf_cache)
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
                "dedup_vi_form": stages["filter4b_loanword"][split],
                "dedup_en_lemma": _deduplicate(stages["filter4b_loanword"][split], "vi_canonical", zipf_cache)[0],
            }
            stage_outputs = {
                **stages,
                "dedup_vi_form": {split: _deduplicate(stages["filter4b_loanword"][split], "vi_canonical", zipf_cache)[0]},
                "dedup_en_lemma": {split: final_by_split[split]},
            }
            for stage in STAGES:
                before, after = stage_inputs[stage], stage_outputs[stage][split]
                flow_counts.setdefault(split, {})[stage] = (len(before), _counts(before), len(after), _counts(after))

        output_rows = sorted((row for split in SPLITS for row in final_by_split[split]), key=lambda row: row["concept_id"])
        output_schema = _output_schema(table.schema)
        output_path = Path(settings["paths"]["output"])
        _write_parquet(output_rows, output_schema, output_path, row_group_size)

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
        logger.info("filter4b_loanword dropped concepts by loan_source_lang: %s", json.dumps(dict(sorted(loan_counts.items())), ensure_ascii=False, sort_keys=True))
        logger.info("Concepts dropped by e2 that surface filter alone would keep (up to 15): vi | source | distance")
        for row in sorted(loan_examples, key=lambda item: item["concept_id"])[:int(report["loanword_examples_n"])]:
            logger.info("%s | %s | %.6f", row["vi_canonical"], ",".join(row["loan_source_lang"]), row["min_surface_dist"])

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
            logger.info("%d TEST concepts %s threshold %.2f (vi | closest lang:form | distance):", len(selected), label, threshold)
            for row in selected:
                logger.info("%s | %s:%s | %.6f", row["vi_canonical"], row["closest_lang"], row["closest_form"], row["min_surface_dist"])

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
