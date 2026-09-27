"""Build auditable v1.3 Vietnamese cloze candidates and prompt records."""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

try:
    from common import iter_jsonl, normalize_nfc, strip_diacritics, vi_orth_key
except ModuleNotFoundError:
    from data.build.common import iter_jsonl, normalize_nfc, strip_diacritics, vi_orth_key


RULES = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
FORBIDDEN = frozenset("()[]/|~")
LEGACY_SAMPLE_ITEMS = (
    ("#4", "chỉ", "point"),
    ("#8", "hôm qua", "yesterday"),
    ("#12", "hội đồng", "council"),
    ("#14", "núi lửa", "volcano"),
    ("#15", "trinh nữ", "virgin"),
    ("#16", "phương pháp", "method"),
    ("#19", "tượng đài", "monument"),
    ("#22", "hội chứng", "syndrome"),
)


def clean_example_text(value: str) -> str:
    """Normalize NFC and collapse whitespace to one line with single spaces."""
    return " ".join(normalize_nfc(value).split())


def is_single_line_raw_example(value: str) -> bool:
    """Reject source examples containing a physical line break before cleanup."""
    return "\n" not in value and "\r" not in value


def count_query_tokens(query: str) -> int:
    """Count letter-bearing whitespace tokens plus the single blank marker."""
    return sum(
        "___" in token or any(unicodedata.category(char).startswith("L") for char in token)
        for token in query.split()
    )


def _syllable_key(token: str) -> str:
    """Normalize a single Vietnamese syllable with the project's orthographic key."""
    return vi_orth_key(token.strip())


def vietnamese_syllables(headwords: set[str]) -> set[str]:
    """Return all whitespace/hyphen-delimited syllables attested in VI headwords."""
    syllables: set[str] = set()
    for headword in headwords:
        for token in re.split(r"[\s-]+", headword):
            core, _, _ = _punctuation_edges(token)
            if core:
                syllables.add(_syllable_key(core))
    return syllables


def c5b_unknown_tokens(query: str, syllables: set[str]) -> list[str]:
    """Return letter-bearing query tokens absent from VI Wiktextract syllables."""
    unknown: list[str] = []
    for raw_token in re.split(r"[\s-]+", query):
        if "___" in raw_token:
            raw_token = raw_token.replace("___", "")
        token, _, _ = _punctuation_edges(raw_token)
        if not token or not any(unicodedata.category(char).startswith("L") for char in token):
            continue
        if _syllable_key(token) not in syllables:
            unknown.append(token)
    return unknown


def query_token_syllables(query: str, canonical: str) -> list[str]:
    """Fill the blank and return case-folded query syllables split on spaces/hyphens."""
    filled = query.replace("___", canonical)
    result: list[str] = []
    for raw_token in re.split(r"[\s-]+", filled):
        core, _, _ = _punctuation_edges(raw_token)
        if core:
            result.append(_syllable_key(core))
    return result


def _punctuation_edges(token: str) -> tuple[str, int, int]:
    """Return a token's non-punctuation core and its offsets."""
    start, end = 0, len(token)
    while start < end and unicodedata.category(token[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(token[end - 1]).startswith("P"):
        end -= 1
    return token[start:end], start, end


def exact_occurrences(text: str, canonical: str) -> list[tuple[int, int]]:
    """Find case-insensitive canonical forms on whitespace-delimited syllable boundaries."""
    normalized = normalize_nfc(text)
    syllables = normalize_nfc(canonical).casefold().split()
    if not syllables:
        raise ValueError("Cannot search for an empty Vietnamese canonical form")
    tokens = list(re.finditer(r"\S+", normalized))
    cores = [_punctuation_edges(match.group(0)) for match in tokens]
    found: list[tuple[int, int]] = []
    for start in range(len(tokens) - len(syllables) + 1):
        window = cores[start:start + len(syllables)]
        if [core.casefold() for core, _, _ in window] != syllables or any(not core for core, _, _ in window):
            continue
        left = tokens[start].start() + window[0][1]
        right = tokens[start + len(syllables) - 1].start() + window[-1][2]
        found.append((left, right))
    return found


def mask_exactly_once(
    text: str,
    canonical: str,
    *,
    min_occurrences: int = 1,
    max_occurrences: int = 1,
) -> tuple[str | None, int]:
    """Mask a canonical occurrence when its count is within the configured range."""
    if min_occurrences < 0 or max_occurrences < min_occurrences:
        raise ValueError("Invalid canonical occurrence bounds")
    normalized = normalize_nfc(text)
    occurrences = exact_occurrences(normalized, canonical)
    if not min_occurrences <= len(occurrences) <= max_occurrences:
        return None, len(occurrences)
    if len(occurrences) != 1:
        raise ValueError("C3 can mask only one occurrence")
    start, end = occurrences[0]
    return normalized[:start] + "___" + normalized[end:], 1


def compound_headword_conflict(query: str, canonical: str, headwords: set[str]) -> str | None:
    """Return a headword window containing the filled target and a neighbor syllable."""
    if query.count("___") != 1:
        raise ValueError("Compound check requires a query with exactly one blank")
    syllables = query_token_syllables(query, canonical)
    target = [_syllable_key(item) for item in canonical.split()]
    target_start = None
    for index in range(len(syllables) - len(target) + 1):
        if syllables[index:index + len(target)] == target:
            target_start = index
            break
    if target_start is None:
        raise ValueError("Compound check could not locate the filled canonical form")

    headwords_by_length: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for headword in headwords:
        parts = tuple(
            _syllable_key(core)
            for item in re.split(r"[\s-]+", headword)
            for core, _, _ in [_punctuation_edges(item)]
            if core
        )
        if len(parts) > len(target):
            headwords_by_length[len(parts)].add(parts)

    target_end = target_start + len(target)
    for window_size in sorted(headwords_by_length):
        for start in range(max(0, target_end - window_size), min(target_start, len(syllables) - window_size) + 1):
            end = start + window_size
            if start <= target_start and end >= target_end:
                window = tuple(syllables[start:end])
                if window in headwords_by_length[window_size]:
                    return normalize_nfc(" ".join(window))
    return None


def lemma_tokens(doc: Any) -> tuple[str, ...]:
    """Return lowercased spaCy lemmas with punctuation and spaces removed."""
    return tuple(
        normalize_nfc(token.lemma_).casefold()
        for token in doc
        if not token.is_space and not token.is_punct and token.lemma_
    )


def contains_lemma_sequence(output: Iterable[str], phrase: Iterable[str]) -> bool:
    """Return whether one lemma sequence occurs contiguously in another."""
    words, target = tuple(output), tuple(phrase)
    if not target or len(target) > len(words):
        return False
    return any(words[index:index + len(target)] == target for index in range(len(words) - len(target) + 1))


def _scope_vi_entries(entries: list[dict[str, Any]], pos: str) -> list[dict[str, Any]]:
    """Use exact-POS homographs when available, otherwise all matching entries."""
    matching = [entry for entry in entries if normalize_nfc(entry["pos"]) == normalize_nfc(pos)]
    return matching if matching else entries


def stream_vi_entries_and_headwords(
    path: str | Path,
    wanted_keys: set[str],
    *,
    progress_every: int,
    logger: logging.Logger,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    """Stream the Vietnamese dump, retaining requested entries and all headword keys."""
    if progress_every < 1:
        raise ValueError("progress_every must be at least one")
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    headwords: set[str] = set()
    entry_count = 0
    for entry_count, entry in enumerate(iter_jsonl(path), start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Vietnamese dump record {entry_count} is not an object")
        word, pos = entry.get("word"), entry.get("pos")
        if not isinstance(word, str) or not word.strip() or not isinstance(pos, str) or not pos.strip():
            raise ValueError(f"Vietnamese dump record {entry_count} lacks a non-empty word/pos")
        key = vi_orth_key(word)
        headwords.add(key)
        if key in wanted_keys:
            found[key].append(entry)
        if entry_count % progress_every == 0:
            logger.info("Streamed Vietnamese entries for v1.3 cloze: %d", entry_count)
    logger.info("Streamed Vietnamese entries for v1.3 cloze: %d (end of dump)", entry_count)
    missing = sorted(wanted_keys - set(found))
    if missing:
        raise ValueError(f"Requested Vietnamese forms have no Wiktextract entries: {missing[:20]!r}")
    return dict(found), headwords


def collect_example_candidates(
    rows: list[dict[str, Any]],
    entries_by_key: dict[str, list[dict[str, Any]]],
    *,
    contains_whole_word: Callable[[str, str], bool],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect all example records and mark the C1 aligned-sense eligibility."""
    candidates: list[dict[str, Any]] = []
    empty_concepts: dict[str, int] = {}
    for row in sorted(rows, key=lambda item: item["concept_id"]):
        concept_id = row["concept_id"]
        canonical = normalize_nfc(row["vi_canonical"]).strip()
        english = normalize_nfc(row["en_lemma"]).strip()
        pos = normalize_nfc(row["pos"]).strip()
        key = vi_orth_key(canonical)
        entries = entries_by_key[key]
        scoped = _scope_vi_entries(entries, pos)
        per_entry_senses: list[list[dict[str, Any]]] = []
        matched_senses: set[tuple[int, int]] = set()
        for entry_index, entry in enumerate(scoped):
            senses = entry.get("senses", [])
            if senses is None:
                senses = []
            if not isinstance(senses, list) or any(not isinstance(sense, dict) for sense in senses):
                raise ValueError(f"Malformed Vietnamese senses on entry {entry.get('word')!r}")
            per_entry_senses.append(senses)
            for sense_index, sense in enumerate(senses):
                glosses = sense.get("glosses", [])
                if glosses is None:
                    glosses = []
                if not isinstance(glosses, list) or any(not isinstance(gloss, str) for gloss in glosses):
                    raise ValueError(f"Malformed senses[].glosses on entry {entry.get('word')!r}")
                if any(contains_whole_word(english, gloss) for gloss in glosses):
                    matched_senses.add((entry_index, sense_index))

        if matched_senses:
            allowed_senses = matched_senses
        else:
            allowed_senses = {
                (entry_index, 0)
                for entry_index, senses in enumerate(per_entry_senses)
                if len(senses) == 1
            }

        before_count = len(candidates)
        for entry_index, (entry, senses) in enumerate(zip(scoped, per_entry_senses)):
            for sense_index, sense in enumerate(senses):
                examples = sense.get("examples", [])
                if examples is None:
                    examples = []
                if not isinstance(examples, list):
                    raise ValueError(f"Unexpected examples on Vietnamese sense of {entry.get('word')!r}")
                for example_index, example in enumerate(examples):
                    if not isinstance(example, dict):
                        raise ValueError(f"Malformed Wiktionary example on {entry.get('word')!r}: {example!r}")
                    text = example.get("text")
                    if text is not None and not isinstance(text, str):
                        raise ValueError(f"Vietnamese example text is not a string on {entry.get('word')!r}: {example!r}")
                    candidates.append({
                        "candidate_id": f"{concept_id}:{entry_index}:{sense_index}:{example_index}",
                        "concept_id": concept_id,
                        "vi": canonical,
                        "en": english,
                        "en_sense_gloss": normalize_nfc(row.get("sense_gloss", "")),
                        "pos": pos,
                        "stratum": row.get("stratum", ""),
                        "m1_extension": bool(row.get("m1_extension", False)),
                        "split": row["split"],
                        "fewshot_set": row.get("fewshot_set"),
                        "entry_word": normalize_nfc(entry["word"]),
                        "sense_index": sense_index,
                        "example_index": example_index,
                        "example_text": text if text is not None else "",
                        "c1_allowed": (
                            (entry_index, sense_index) in allowed_senses
                            and "ref" not in example
                            and example.get("type") != "quotation"
                            and isinstance(text, str)
                        ),
                        "c1_reason": (
                            "example has a ref field" if "ref" in example
                            else "example type is quotation" if example.get("type") == "quotation"
                            else "example has no text field" if not isinstance(text, str)
                            else "no English-gloss-matched sense; entry is not single-sense"
                        ),
                    })
        empty_concepts[concept_id] = len(candidates) - before_count
    return candidates, empty_concepts


def _failure(candidate: dict[str, Any], rule: str, reason: str) -> dict[str, Any]:
    """Render one rejected-example audit record."""
    return {
        "candidate_id": candidate["candidate_id"],
        "concept_id": candidate["concept_id"],
        "vi": candidate.get("vi", ""),
        "en": candidate.get("en", ""),
        "entry_word": candidate.get("entry_word", ""),
        "sense_index": candidate.get("sense_index"),
        "example_index": candidate.get("example_index"),
        "example_text": candidate.get("example_text", ""),
        "query": candidate.get("query", ""),
        "failed_rule": rule,
        "reason": reason,
        "nllb_english": candidate.get("nllb_english", []),
        "nllb_limit_hits": candidate.get("nllb_limit_hits", []),
        "c7_match_type": candidate.get("c7_match_type"),
        "c7_match_word": candidate.get("c7_match_word"),
        "c7_beam_match_count": candidate.get("c7_beam_match_count", 0),
        "c8_conflicts": candidate.get("c8_conflicts", []),
    }


def _scope_name(candidate: dict[str, Any]) -> str:
    """Return the reporting group for test or demonstration candidates."""
    if candidate["split"] == "test":
        return "extension" if candidate["m1_extension"] else "main"
    return f"fewshot_set_{candidate.get('fewshot_set')}"


def _stage_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(_scope_name(candidate) for candidate in candidates))


def english_match_targets(
    english: str,
    *,
    pos: str,
    nlp: Any,
    wordnet_reader: Any,
) -> list[dict[str, Any]]:
    """Build the only permitted C7 targets: exact, two-word head, and same-POS WordNet."""
    doc = nlp(english)
    target_words = lemma_tokens(doc)
    if not target_words:
        raise ValueError(f"spaCy produced no English lemmas for target {english!r}")
    targets: list[dict[str, Any]] = [{
        "type": "target_lemma", "word": english, "sequence": target_words,
    }]
    if len(target_words) == 2:
        lexical = [token for token in doc if not token.is_space and not token.is_punct]
        roots = [token for token in lexical if token.head == token]
        head = roots[-1] if roots else lexical[-1]
        targets.append({
            "type": "two_word_head_lemma",
            "word": normalize_nfc(head.lemma_),
            "sequence": (normalize_nfc(head.lemma_).casefold(),),
        })

    wordnet_pos = {"noun": "n", "verb": "v", "adj": "a"}.get(pos)
    if wordnet_pos is None:
        raise ValueError(f"Unsupported POS for same-POS WordNet matching: {pos!r}")
    phrase_key = normalize_nfc(english).replace(" ", "_")
    for synset in wordnet_reader.synsets(phrase_key, pos=wordnet_pos):
        for lemma in synset.lemmas():
            word = normalize_nfc(lemma.name().replace("_", " "))
            sequence = lemma_tokens(nlp(word))
            if sequence:
                targets.append({"type": "wordnet_synset_lemma", "word": word, "sequence": sequence})

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for target in targets:
        key = (target["type"], target["sequence"])
        if key not in seen:
            seen.add(key)
            unique.append(target)
    return unique


def match_c7_output(output_lemmas: tuple[str, ...], targets: list[dict[str, Any]]) -> dict[str, str] | None:
    """Return the first allowed exact C7 match and its auditable type/word."""
    for target in targets:
        if contains_lemma_sequence(output_lemmas, target["sequence"]):
            return {"type": target["type"], "word": target["word"]}
    return None


def c7_beam_match_results(
    output_lemmas: list[tuple[str, ...]], limit_hits: list[bool], targets: list[dict[str, Any]],
) -> list[dict[str, str] | None]:
    """Return an allowed C7 match for each output, excluding token-limit hits."""
    if len(output_lemmas) != len(limit_hits):
        raise ValueError("C7 output lemmas and token-limit flags must have equal lengths")
    return [
        None if reached_limit else match_c7_output(lemmas, targets)
        for lemmas, reached_limit in zip(output_lemmas, limit_hits, strict=True)
    ]


def match_untruncated_c7_beams(
    output_lemmas: list[tuple[str, ...]],
    limit_hits: list[bool],
    targets: list[dict[str, Any]],
    *,
    minimum_matches: int,
) -> dict[str, str] | None:
    """Require a top-beam match plus minimum agreement across untruncated beams."""
    if minimum_matches < 1:
        raise ValueError("C7 minimum beam matches must be positive")
    matches = c7_beam_match_results(output_lemmas, limit_hits, targets)
    if not matches or matches[0] is None or sum(match is not None for match in matches) < minimum_matches:
        return None
    return matches[0]


def recheck_point_sample(
    candidates: list[dict[str, Any]],
    *,
    translator: Any,
    nlp: Any,
    wordnet_reader: Any,
    source_code: str,
    target_code: str,
    minimum_beam_matches: int,
    expected_beam_count: int,
) -> dict[str, Any]:
    """Independently re-evaluate the legacy #4 point example under C7's output rule."""
    matches = [candidate for candidate in candidates if candidate["vi"] == "chỉ" and candidate["en"] == "point"]
    if len(matches) != 1:
        raise ValueError(f"Expected one legacy point example for C7 diagnostic; found {len(matches)}")
    candidate = matches[0]
    raw = candidate["example_text"]
    if not is_single_line_raw_example(raw):
        return {"status": "not_rechecked_raw_multiline", "concept_id": candidate["concept_id"]}
    text = clean_example_text(raw)
    occurrences = exact_occurrences(text, candidate["vi"])
    if len(occurrences) != 1:
        return {
            "status": "not_rechecked_target_occurrence_count",
            "concept_id": candidate["concept_id"],
            "occurrences": len(occurrences),
        }
    start, end = occurrences[0]
    filled = text[:start] + candidate["vi"] + text[end:]
    request = (source_code, target_code, filled)
    details = translator.translate_many_with_limit_hits([request])[request]
    if len(details["outputs"]) != expected_beam_count or len(details["limit_hits"]) != expected_beam_count:
        raise ValueError(
            f"C7 expected {expected_beam_count} return beams for point diagnostic; "
            f"got {len(details['outputs'])} outputs and {len(details['limit_hits'])} limit flags"
        )
    output_lemmas = [lemma_tokens(doc) for doc in nlp.pipe(details["outputs"], batch_size=32)]
    targets = english_match_targets("point", pos=candidate["pos"], nlp=nlp, wordnet_reader=wordnet_reader)
    beam_count = len(details["outputs"])
    matches = c7_beam_match_results(output_lemmas, details["limit_hits"], targets)
    match = match_untruncated_c7_beams(
        output_lemmas, details["limit_hits"], targets,
        minimum_matches=minimum_beam_matches,
    )
    return {
        "status": "evaluated",
        "concept_id": candidate["concept_id"],
        "NLLB English": details["outputs"],
        "NLLB limit hits": details["limit_hits"],
        "evaluated_beam_count": beam_count,
        "C7 matching beam count": sum(item is not None for item in matches),
        "C7 matched type": match["type"] if match else None,
        "C7 matched word": match["word"] if match else None,
        "passes_C7_beam_agreement": match is not None and match["word"].casefold() == "point",
    }


def filter_cloze_candidates_c1_to_c7(
    candidates: list[dict[str, Any]],
    *,
    headwords: set[str],
    forbidden_characters: Iterable[str],
    min_query_tokens: int,
    min_occurrences: int,
    max_occurrences: int,
    minimum_beam_matches: int,
    expected_beam_count: int,
    spacy_batch_size: int,
    translator: Any,
    nlp: Any,
    wordnet_reader: Any,
    step08: Any,
    source_code: str,
    target_code: str,
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, int]], list[dict[str, Any]],
    dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, Counter[str]]],
    list[dict[str, Any]],
]:
    """Apply the ordered content rules C1–C7 and return survivors and audit counts."""
    current = sorted(candidates, key=lambda row: row["candidate_id"])
    counts: dict[str, dict[str, int]] = {"candidates": _stage_counts(current)}
    pos_counts: dict[str, dict[str, Counter[str]]] = {"candidates": defaultdict(Counter)}
    failures: list[dict[str, Any]] = []
    per_concept: dict[str, dict[str, int]] = defaultdict(dict)
    for candidate in current:
        per_concept[candidate["concept_id"]]["candidates"] = per_concept[candidate["concept_id"]].get("candidates", 0) + 1
        pos_counts["candidates"][_scope_name(candidate)][candidate["pos"]] += 1

    def apply_rule(rule: str, predicate: Callable[[dict[str, Any]], tuple[bool, str]]) -> None:
        nonlocal current
        passed: list[dict[str, Any]] = []
        for candidate in current:
            keep, reason = predicate(candidate)
            if keep:
                passed.append(candidate)
            else:
                failures.append(_failure(candidate, rule, reason))
        current = passed
        counts[f"after_{rule}"] = _stage_counts(current)
        pos_counts[f"after_{rule}"] = defaultdict(Counter)
        for candidate in candidates:
            per_concept[candidate["concept_id"]][f"after_{rule}"] = 0
        for candidate in current:
            per_concept[candidate["concept_id"]][f"after_{rule}"] = per_concept[candidate["concept_id"]].get(f"after_{rule}", 0) + 1
            pos_counts[f"after_{rule}"][_scope_name(candidate)][candidate["pos"]] += 1

    apply_rule("C1", lambda candidate: (bool(candidate["c1_allowed"]), candidate.get("c1_reason", "example source is not eligible")))

    def c2(candidate: dict[str, Any]) -> tuple[bool, str]:
        raw = candidate["example_text"]
        if not is_single_line_raw_example(raw):
            return False, "raw source example contains a newline or carriage return"
        cleaned = clean_example_text(candidate["example_text"])
        candidate["clean_text"] = cleaned
        return bool(cleaned), "empty example after whitespace cleanup"

    apply_rule("C2", c2)

    def c3(candidate: dict[str, Any]) -> tuple[bool, str]:
        query, occurrence_count = mask_exactly_once(
            candidate["clean_text"], candidate["vi"],
            min_occurrences=min_occurrences, max_occurrences=max_occurrences,
        )
        candidate["occurrence_count"] = occurrence_count
        candidate["query"] = query or ""
        if occurrence_count != 1:
            return False, f"canonical Vietnamese form occurs {occurrence_count} times; expected exactly one"
        start, end = exact_occurrences(candidate["clean_text"], candidate["vi"])[0]
        candidate["filled_sentence"] = candidate["clean_text"][:start] + candidate["vi"] + candidate["clean_text"][end:]
        return True, ""

    apply_rule("C3", c3)
    # C8 is displayed last in the ordered funnel, but its query context is fixed
    # here: every candidate with a valid blank, before C4–C7 can filter it out.
    c3_context_candidates = [dict(candidate) for candidate in current]

    def c4(candidate: dict[str, Any]) -> tuple[bool, str]:
        candidate["token_count"] = count_query_tokens(candidate["query"])
        return candidate["token_count"] >= min_query_tokens, f"query has {candidate['token_count']} letter-bearing tokens including blank; minimum is {min_query_tokens}"

    apply_rule("C4", c4)

    forbidden = frozenset(forbidden_characters)
    vi_syllables = vietnamese_syllables(headwords)

    def c5(candidate: dict[str, Any]) -> tuple[bool, str]:
        found = sorted(set(candidate["clean_text"]).intersection(forbidden))
        if found:
            return False, f"query contains forbidden character(s): {''.join(found)}"
        unknown = c5b_unknown_tokens(candidate["query"], vi_syllables)
        candidate["c5b_unknown_tokens"] = unknown
        return not unknown, f"query contains token(s) absent from Vietnamese Wiktextract syllables: {', '.join(unknown)}" if unknown else ""

    apply_rule("C5", c5)

    def c6(candidate: dict[str, Any]) -> tuple[bool, str]:
        conflict = compound_headword_conflict(candidate["query"], candidate["vi"], headwords)
        candidate["compound_headword"] = conflict
        return conflict is None, f"blank plus adjacent syllable(s) forms Wiktextract headword {conflict!r}" if conflict else ""

    apply_rule("C6", c6)

    english_requests = [(source_code, target_code, candidate["filled_sentence"]) for candidate in current]
    detailed_translations = translator.translate_many_with_limit_hits(english_requests) if english_requests else {}
    request_key_by_candidate = {
        candidate["candidate_id"]: (source_code, target_code, normalize_nfc(candidate["filled_sentence"]))
        for candidate in current
    }
    output_texts = sorted({
        output
        for detail in detailed_translations.values()
        for output in detail["outputs"]
    })

    targets_by_candidate = {
        candidate["candidate_id"]: english_match_targets(
            candidate["en"], pos=candidate["pos"], nlp=nlp, wordnet_reader=wordnet_reader,
        )
        for candidate in current
    }

    # Batch parse translations for throughput while retaining the same spaCy lemmas per text.
    output_lemmas: dict[str, tuple[str, ...]] = {}
    if spacy_batch_size < 1:
        raise ValueError("spacy_batch_size must be positive")
    for text, doc in zip(output_texts, nlp.pipe(output_texts, batch_size=spacy_batch_size)):
        output_lemmas[text] = lemma_tokens(doc)

    semantic_survivors: list[dict[str, Any]] = []
    for candidate in current:
        key = request_key_by_candidate[candidate["candidate_id"]]
        detail = detailed_translations[key]
        outputs = detail["outputs"]
        limit_hits = detail["limit_hits"]
        if len(outputs) != expected_beam_count or len(limit_hits) != expected_beam_count:
            raise ValueError(
                f"C7 expected {expected_beam_count} return beams for {candidate['candidate_id']}; "
                f"got {len(outputs)} outputs and {len(limit_hits)} limit flags"
            )
        candidate["nllb_english"] = outputs
        candidate["nllb_limit_hits"] = limit_hits
        selected_lemmas = [output_lemmas[output] for output in outputs]
        beam_matches = c7_beam_match_results(selected_lemmas, limit_hits, targets_by_candidate[candidate["candidate_id"]])
        beam_match_count = sum(match is not None for match in beam_matches)
        candidate["c7_beam_match_count"] = beam_match_count
        matched = match_untruncated_c7_beams(
            selected_lemmas, limit_hits, targets_by_candidate[candidate["candidate_id"]],
            minimum_matches=minimum_beam_matches,
        )
        if matched is not None:
            candidate["c7_match_type"] = matched["type"]
            candidate["c7_match_word"] = matched["word"]
            semantic_survivors.append(candidate)
        else:
            candidate["c7_match_type"] = None
            candidate["c7_match_word"] = None
            untruncated = sum(not value for value in limit_hits)
            failures.append(_failure(
                candidate, "C7",
                "C7 requires a matching untruncated top beam and at least "
                f"{minimum_beam_matches} matching untruncated return beams "
                f"(matching beams={beam_match_count}/{len(outputs)}; "
                f"untruncated beams={untruncated}; token-limit hits={sum(limit_hits)})",
            ))
    current = semantic_survivors
    counts["after_C7"] = _stage_counts(current)
    pos_counts["after_C7"] = defaultdict(Counter)
    for candidate in candidates:
        per_concept[candidate["concept_id"]]["after_C7"] = 0
    for candidate in current:
        per_concept[candidate["concept_id"]]["after_C7"] = per_concept[candidate["concept_id"]].get("after_C7", 0) + 1
        pos_counts["after_C7"][_scope_name(candidate)][candidate["pos"]] += 1
    nllb_stats = {
        "requests": len(english_requests),
        "unique_requests": len(set(request_key_by_candidate.values())),
        "outputs": len(output_texts),
    }
    return (
        current, counts, failures, {key: dict(value) for key, value in per_concept.items()},
        nllb_stats, pos_counts, c3_context_candidates,
    )


def query_owner_map(candidates: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Map each masked query to its distinct concept owners."""
    owners: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        owners[candidate["query"]].add(candidate["concept_id"])
    return dict(owners)


def c8_conflicts(
    candidate: dict[str, Any], context_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Find same-query candidates with a different normalized Vietnamese fill."""
    query = candidate["query"]
    fill = normalize_nfc(candidate.get("vi", "")).strip().casefold()
    conflicts = []
    for other in context_candidates:
        if other.get("candidate_id") == candidate.get("candidate_id"):
            continue
        if other.get("query") != query:
            continue
        other_fill = normalize_nfc(other.get("vi", "")).strip().casefold()
        if other_fill == fill:
            continue
        conflicts.append({
            "candidate_id": str(other.get("candidate_id", "")),
            "concept_id": str(other.get("concept_id", "")),
            "vi": str(other.get("vi", "")),
        })
    return sorted(conflicts, key=lambda item: (item["concept_id"], item["candidate_id"], item["vi"]))


def select_demo_concept_ids(
    *,
    candidate_rows: list[dict[str, Any]],
    fewshot_sets: dict[int, list[dict[str, Any]]],
    directions_rows: list[dict[str, Any]],
    preferred_vi: list[str],
    demo_count: int,
    preferred_min_tokens: int,
    test_rows: list[dict[str, Any]],
    context_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose noncollapsed passing demos by set tier, then directions, balancing POS."""
    test_ids = {row["concept_id"] for row in test_rows}
    selected: list[str] = []
    selected_candidates: list[dict[str, Any]] = []
    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        candidates_by_id[candidate["concept_id"]].append(candidate)
    rows_by_id = {
        row["concept_id"]: row
        for row in [*(item for group in fewshot_sets.values() for item in group), *directions_rows, *test_rows]
    }
    selected_pos: set[str] = set()

    def candidate_for(concept_id: str) -> dict[str, Any] | None:
        row = rows_by_id.get(concept_id)
        if row is None or row.get("collapsed") is not False or concept_id in test_ids:
            return None
        options = [
            candidate for candidate in candidates_by_id.get(concept_id, [])
            if not c8_conflicts(candidate, context_candidates)
        ]
        if not options:
            return None
        return choose_examples(options, preferred_min_tokens=preferred_min_tokens)[concept_id]

    preferred_forms = [normalize_nfc(value).casefold() for value in preferred_vi]
    for set_number in (*range(1, 7), 0):
        if set_number == 0:
            tier_rows = directions_rows
        else:
            tier_rows = fewshot_sets.get(set_number, [])
        by_id = {row["concept_id"]: row for row in tier_rows}
        if not by_id:
            continue

        preferred_ids: list[str] = []
        if set_number == 1:
            for form in preferred_forms:
                preferred_ids.extend(
                    concept_id for concept_id, row in sorted(by_id.items())
                    if normalize_nfc(row["vi_canonical"]).casefold() == form
                    and concept_id not in preferred_ids
                )

        # Preferred set-1 concepts are tried first. Remaining items are greedily
        # selected to introduce a new POS before reusing one, with concept_id ties.
        remaining = [concept_id for concept_id in sorted(by_id) if concept_id not in preferred_ids]
        ordered = [*preferred_ids]
        while remaining:
            available = [concept_id for concept_id in remaining if candidate_for(concept_id) is not None]
            if not available:
                break
            new_pos = [
                concept_id for concept_id in available
                if candidate_for(concept_id)["pos"] not in selected_pos
            ]
            chosen = min(new_pos or available)
            ordered.append(chosen)
            remaining.remove(chosen)

        for concept_id in ordered:
            if concept_id in selected:
                continue
            candidate = candidate_for(concept_id)
            if candidate is None:
                continue
            selected.append(concept_id)
            selected_candidates.append(candidate)
            selected_pos.add(candidate["pos"])
            if len(selected) == demo_count:
                return selected_candidates
    return selected_candidates


def apply_c8(
    candidates: list[dict[str, Any]],
    *,
    context_candidates: list[dict[str, Any]],
    per_concept_counts: dict[str, dict[str, int]],
    prior_failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Drop candidates colliding with a different fill anywhere in the C3 pool."""
    kept: list[dict[str, Any]] = []
    failures = list(prior_failures)
    for candidate in candidates:
        conflicts = c8_conflicts(candidate, context_candidates)
        if conflicts:
            candidate["c8_conflicts"] = conflicts
            colliders = ", ".join(
                f"{item['concept_id']} ({item['candidate_id']}; vi={item['vi']!r})"
                for item in conflicts
            )
            failures.append(_failure(
                candidate, "C8",
                f"masked query collides with different Vietnamese fill(s): {colliders}",
            ))
        else:
            kept.append(candidate)
    for values in per_concept_counts.values():
        values["after_C8"] = 0
    for candidate in kept:
        concept_counts = per_concept_counts.setdefault(candidate["concept_id"], {})
        concept_counts["after_C8"] = concept_counts.get("after_C8", 0) + 1
    return kept, failures, per_concept_counts


def choose_examples(candidates: list[dict[str, Any]], *, preferred_min_tokens: int) -> dict[str, dict[str, Any]]:
    """Choose the shortest ≥ configured preference, else longest; lexical query breaks ties."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["concept_id"]].append(candidate)
    selected: dict[str, dict[str, Any]] = {}
    for concept_id, group in grouped.items():
        preferred = [candidate for candidate in group if candidate["token_count"] >= preferred_min_tokens]
        if preferred:
            selected[concept_id] = min(preferred, key=lambda item: (item["token_count"], item["query"]))
        else:
            longest = max(candidate["token_count"] for candidate in group)
            selected[concept_id] = min(
                (candidate for candidate in group if candidate["token_count"] == longest),
                key=lambda item: item["query"],
            )
    return selected


def first_empty_rule(stage_counts: dict[str, int]) -> str:
    """Return the first C1–C8 stage that removes a concept's final candidate."""
    if stage_counts.get("candidates", 0) == 0:
        return "C1"
    for rule in RULES:
        if stage_counts.get(f"after_{rule}", 0) == 0:
            return rule
    return ""


def old_query_from_prompt(prompt: str, answer_label: str) -> str:
    """Extract the final v1.2 query, preserving old multiline queries for the audit CSV."""
    lines = prompt.splitlines(keepends=True)
    answer_prefix = f"{answer_label} "
    answer_indexes = [index for index, line in enumerate(lines) if line.rstrip("\r\n").startswith(answer_prefix)]
    if len(answer_indexes) != 3 or not lines or lines[-1].rstrip("\r\n") != answer_label:
        raise ValueError("Unexpected v1.2 cloze prompt layout while reading its old query")
    query_start = sum(len(line) for line in lines[:answer_indexes[-1] + 1])
    query_end = sum(len(line) for line in lines[:-1])
    return prompt[query_start:query_end].rstrip("\r\n")


def write_failure_jsonl(path: str | Path, failures: list[dict[str, Any]]) -> None:
    """Write candidate-level first-failure records as stable UTF-8 JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(failures, key=lambda item: (item["concept_id"], item["candidate_id"], item["failed_rule"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(output)


def write_dropped_csv(
    path: str | Path,
    *,
    old_rows: list[dict[str, Any]],
    selected_by_id: dict[str, dict[str, Any]],
    per_concept_counts: dict[str, dict[str, int]],
    answer_label: str,
) -> list[dict[str, str]]:
    """Write one row for each v1.2 cloze concept that no longer has a C1–C8 candidate."""
    dropped: list[dict[str, str]] = []
    for row in sorted(old_rows, key=lambda item: item["concept_id"]):
        concept_id = row["concept_id"]
        if concept_id in selected_by_id:
            continue
        counts = per_concept_counts.get(concept_id, {"candidates": 0})
        rule = first_empty_rule(counts) or "C8"
        dropped.append({
            "concept_id": concept_id,
            "vi": row.get("target_vi", "").strip(),
            "en": row.get("target_en", "").strip(),
            "old query": old_query_from_prompt(row["prompt"], answer_label),
            "failed rule": rule,
        })
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["concept_id", "vi", "en", "old query", "failed rule"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(dropped)
    return dropped


def build_record(
    row: dict[str, Any],
    *,
    prompt: str,
    condition: str,
    fewshot_set: int | None,
    vi_target: str,
    demo_concept_ids: list[str],
) -> dict[str, Any]:
    """Build a v1.2 prompt row with the v1.3 demonstration provenance field."""
    targets = {
        "target_vi": vi_target,
        "target_en": row["en_lemma"],
        "target_zh": row["zh_canonical"],
        "target_fr": row["fr_canonical"],
        "target_id": row["id_canonical"],
    }
    for key, value in list(targets.items()):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing prompt target {key} for {row['concept_id']!r}")
        targets[key] = " " + normalize_nfc(value).strip()
    return {
        "concept_id": row["concept_id"],
        "split": row["split"],
        "m1_extension": bool(row["m1_extension"]),
        "format": "cloze",
        "condition": condition,
        "fewshot_set": fewshot_set,
        "prompt": prompt,
        **targets,
        "demo_concept_ids": list(demo_concept_ids),
    }


def render_prompt(
    demo_rows: list[dict[str, Any]],
    *,
    query: str,
    answer_label: str,
) -> str:
    """Render three cloze demonstrations and the target query in v1.2 format."""
    lines: list[str] = []
    for row in demo_rows:
        lines.extend((row["query"], f"{answer_label} {row['vi']}"))
    lines.extend((query, answer_label))
    return "\n".join(lines)


def nodiac_demo_rows(demo_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Strip Vietnamese diacritics from demo queries and fills for nodiac prompts."""
    return [
        {
            "query": strip_diacritics(item["query"], preserve_case=True),
            "vi": strip_diacritics(item["vi"], preserve_case=True),
        }
        for item in demo_rows
    ]


def verify_records(
    records: dict[str, list[dict[str, Any]]],
    *,
    rows_by_id: dict[str, dict[str, Any]],
    demo_concept_ids: list[str],
    answer_label: str,
    primary_set: int | None,
) -> None:
    """Assert the v1.2 cloze schema plus demo IDs and the task's prompt invariants."""
    expected_fields = {
        "concept_id", "split", "m1_extension", "format", "condition", "fewshot_set", "prompt",
        "target_vi", "target_en", "target_zh", "target_fr", "target_id", "demo_concept_ids",
    }
    for condition in ("diac", "nodiac"):
        condition_answer_label = answer_label if condition == "diac" else strip_diacritics(answer_label, preserve_case=True)
        for record in records.get(condition, []):
            if set(record) != expected_fields:
                raise AssertionError(f"Unexpected cloze {condition} schema: {sorted(record)!r}")
            if record["demo_concept_ids"] != demo_concept_ids:
                raise AssertionError(f"Cloze demo provenance changed on {record['concept_id']!r}")
            if record["condition"] != condition or record["format"] != "cloze" or record["fewshot_set"] != primary_set:
                raise AssertionError(f"Unexpected cloze prompt metadata on {record['concept_id']!r}")
            lines = record["prompt"].splitlines()
            if len(lines) != 8 or lines[-1] != condition_answer_label:
                raise AssertionError(f"Unexpected cloze prompt line layout on {record['concept_id']!r}")
            queries = lines[::2]
            if len(queries) != 4 or any(query.count("___") != 1 or "\n" in query or "\r" in query for query in queries):
                raise AssertionError(f"Cloze prompts need four one-blank single-line queries: {record['concept_id']!r}")
            if any(not lines[index].startswith(f"{condition_answer_label} ") for index in (1, 3, 5)):
                raise AssertionError(f"Cloze demonstrations lack answer lines on {record['concept_id']!r}")
            row = rows_by_id[record["concept_id"]]
            vi_target = row["vi_canonical"]
            if condition == "nodiac":
                vi_target = strip_diacritics(vi_target, preserve_case=True)
            expected_targets = {
                "target_vi": vi_target,
                "target_en": row["en_lemma"],
                "target_zh": row["zh_canonical"],
                "target_fr": row["fr_canonical"],
                "target_id": row["id_canonical"],
            }
            for field, target in expected_targets.items():
                if record[field] != " " + normalize_nfc(target).strip():
                    raise AssertionError(f"Cloze target {field} mismatches concept fields for {record['concept_id']!r}")
            if condition == "nodiac" and row.get("collapsed") is True:
                raise AssertionError(f"Collapsed concept leaked into nodiac cloze: {record['concept_id']!r}")


def report_funnel(
    counts: dict[str, dict[str, int]],
    *,
    stage_pos: dict[str, dict[str, Counter[str]]],
) -> dict[str, list[dict[str, Any]]]:
    """Format main and extension candidate funnel rows with POS counts."""
    result: dict[str, list[dict[str, Any]]] = {}
    for scope in ("main", "extension"):
        rows: list[dict[str, Any]] = []
        for stage in ("candidates", *(f"after_{rule}" for rule in RULES)):
            value = counts.get(scope, {}).get(stage, 0)
            pos_counts = dict(sorted(stage_pos.get(stage, {}).get(scope, Counter()).items()))
            rows.append({"stage": stage, "n": value, "n_by_pos": pos_counts})
        result[scope] = rows
    return result


def build_cloze_v13(
    *,
    all_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    fewshot_sets: dict[int, list[dict[str, Any]]],
    config: dict[str, Any],
    translator: Any,
    step07: Any,
    step08: Any,
    attest_matcher: Callable[[str, str], bool],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Build C1–C8 cloze selections, audit files, records, and a deterministic report."""
    import spacy

    settings = config["prompts"]["cloze_v13"]
    paths = settings["paths"]
    fewshot_pool_ids = {row["concept_id"] for group in fewshot_sets.values() for row in group}
    rows_by_id = {row["concept_id"]: row for row in all_rows}
    test_ids = {row["concept_id"] for row in test_rows}
    if test_ids.intersection(fewshot_pool_ids):
        raise AssertionError("Test cloze targets overlap the selected few-shot sets")
    direction_ids = {row["concept_id"] for row in all_rows if row.get("split") == "directions"}
    candidate_rows = [rows_by_id[concept_id] for concept_id in sorted(test_ids | fewshot_pool_ids | direction_ids)]
    directions_rows = [rows_by_id[concept_id] for concept_id in sorted(direction_ids)]
    wanted_keys = {vi_orth_key(row["vi_canonical"]) for row in candidate_rows}
    vi_dump = Path(config["etymology"]["paths"]["vietnamese_dump"])
    entries_by_key, headwords = stream_vi_entries_and_headwords(
        vi_dump, wanted_keys,
        progress_every=int(config["logging"]["progress_every"]), logger=logger,
    )
    example_candidates, empty_concepts = collect_example_candidates(
        candidate_rows, entries_by_key, contains_whole_word=attest_matcher,
    )
    logger.info("Cloze v1.3 raw example candidates=%d; concepts without example records=%d",
                len(example_candidates), sum(value == 0 for value in empty_concepts.values()))

    step07_config = {
        **config["nllb"],
        "progress_every": int(config["logging"]["progress_every"]),
    }
    step08_config = config["etymology"]
    wordnet_reader, wordnet_version = step08.load_local_wordnet(step08_config["paths"]["nltk_data"])
    spacy_model = config["concreteness"]["spacy_model"]
    nlp = spacy.load(spacy_model)
    logger.info("Cloze v1.3 semantic resources: spaCy=%s; WordNet=%s", spacy_model, wordnet_version)

    source_code = config["nllb"]["lang_codes"]["vi"]
    target_code = config["nllb"]["lang_codes"]["en"]
    legacy_point_recheck = recheck_point_sample(
        example_candidates,
        translator=translator,
        nlp=nlp,
        wordnet_reader=wordnet_reader,
        source_code=source_code,
        target_code=target_code,
        minimum_beam_matches=int(settings["semantic_min_beam_matches"]),
        expected_beam_count=int(step07_config["num_return"]),
    )
    before_calls = int(getattr(translator, "translation_calls", 0))
    before_hits = int(getattr(translator, "cache_hits", 0))
    c1_c7, funnel_counts, failures, per_concept_counts, nllb_stats, stage_pos, c3_context_candidates = filter_cloze_candidates_c1_to_c7(
        example_candidates,
        headwords=headwords,
        forbidden_characters=settings["forbidden_characters"],
        min_query_tokens=int(settings["min_query_tokens"]),
        min_occurrences=int(settings["min_canonical_occurrences"]),
        max_occurrences=int(settings["max_canonical_occurrences"]),
        minimum_beam_matches=int(settings["semantic_min_beam_matches"]),
        expected_beam_count=int(step07_config["num_return"]),
        spacy_batch_size=int(settings["spacy_batch_size"]),
        translator=translator,
        nlp=nlp,
        wordnet_reader=wordnet_reader,
        step08=step08,
        source_code=source_code,
        target_code=target_code,
    )
    demo_candidates = select_demo_concept_ids(
        candidate_rows=c1_c7,
        fewshot_sets=fewshot_sets,
        directions_rows=directions_rows,
        preferred_vi=list(settings["preferred_demo_vi"]),
        demo_count=int(settings["demo_count"]),
        preferred_min_tokens=int(settings["preferred_example_min_tokens"]),
        test_rows=test_rows,
        context_candidates=c3_context_candidates,
    )
    demo_ids = [candidate["concept_id"] for candidate in demo_candidates]
    final_concept_ids = test_ids | set(demo_ids)
    c8_input = [candidate for candidate in c1_c7 if candidate["concept_id"] in test_ids] + demo_candidates
    per_concept_counts = {
        concept_id: per_concept_counts.get(concept_id, {"candidates": 0})
        for concept_id in final_concept_ids
    }
    c8_kept, failures, per_concept_counts = apply_c8(
        c8_input,
        context_candidates=c3_context_candidates,
        per_concept_counts=per_concept_counts,
        prior_failures=failures,
    )
    funnel_counts["after_C8"] = _stage_counts(c8_kept)
    stage_pos["after_C8"] = defaultdict(Counter)
    for candidate in c8_kept:
        stage_pos["after_C8"][_scope_name(candidate)][candidate["pos"]] += 1
    if any(concept_id not in {candidate["concept_id"] for candidate in c8_kept} for concept_id in demo_ids):
        raise ValueError("A selected demonstration has no example that passes C1–C8")

    selected_by_id = choose_examples(
        c8_kept,
        preferred_min_tokens=int(settings["preferred_example_min_tokens"]),
    )
    demo_rows = [
        {
            **selected_by_id[concept_id],
            "vi": rows_by_id[concept_id]["vi_canonical"],
        }
        for concept_id in demo_ids
    ]
    collapsed_demos = [
        {"concept_id": concept_id, "vi": rows_by_id[concept_id]["vi_canonical"]}
        for concept_id in demo_ids
        if rows_by_id[concept_id].get("collapsed") is not False
    ]
    if collapsed_demos:
        raise ValueError(f"Selected cloze demonstration(s) are collapsed; stop per Step 5: {collapsed_demos!r}")

    test_selected = {
        concept_id: candidate
        for concept_id, candidate in selected_by_id.items()
        if concept_id in test_ids
    }
    test_cloze_rows = [row for row in test_rows if row["concept_id"] in test_selected]
    main_rows = [row for row in test_cloze_rows if row["m1_extension"] is False]
    extension_rows = [row for row in test_cloze_rows if row["m1_extension"] is True]
    test_cloze_nodiac_rows = []
    for row in test_cloze_rows:
        collapsed = row.get("collapsed")
        if type(collapsed) is not bool:
            raise ValueError(f"Cloze concept {row['concept_id']!r} lacks boolean collapsed metadata")
        if not collapsed:
            test_cloze_nodiac_rows.append(row)
    main_nodiac_rows = [row for row in test_cloze_nodiac_rows if row["m1_extension"] is False]
    extension_nodiac_rows = [row for row in test_cloze_nodiac_rows if row["m1_extension"] is True]

    main_by_stratum_total = Counter(row["stratum"] for row in test_rows if row["m1_extension"] is False)
    main_by_stratum_survived = Counter(row["stratum"] for row in main_rows)
    extension_by_stratum_total = Counter(row["stratum"] for row in test_rows if row["m1_extension"] is True)
    extension_by_stratum_survived = Counter(row["stratum"] for row in extension_rows)
    main_nodiac_by_stratum = Counter(row["stratum"] for row in main_nodiac_rows)
    extension_nodiac_by_stratum = Counter(row["stratum"] for row in extension_nodiac_rows)

    old_path = Path(config["prompts"]["paths"]["output_dir"]) / "cloze_diac_set1.jsonl"
    old_rows = list(iter_jsonl(old_path))
    if any(not isinstance(row, dict) or "concept_id" not in row or "prompt" not in row for row in old_rows):
        raise ValueError(f"Unexpected existing v1.2 cloze record schema in {old_path}")
    dropped = write_dropped_csv(
        paths["dropped_csv"], old_rows=old_rows,
        selected_by_id=test_selected, per_concept_counts=per_concept_counts,
        answer_label=config["prompts"]["cloze"]["answer_label"],
    )
    write_failure_jsonl(paths["candidate_failures"], failures)
    dropped_by_id = {item["concept_id"]: item["failed rule"] for item in dropped}
    legacy_sample_rechecks: list[dict[str, Any]] = []
    for sample_id, expected_vi, expected_en in LEGACY_SAMPLE_ITEMS:
        matches = [
            row for row in test_rows
            if normalize_nfc(row["vi_canonical"]).strip().casefold() == expected_vi.casefold()
            and normalize_nfc(row["en_lemma"]).strip().casefold() == expected_en.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one test concept for sample item {sample_id} {expected_vi!r}/{expected_en!r}; found {len(matches)}"
            )
        source_row = matches[0]
        concept_id = source_row["concept_id"]
        survivor = test_selected.get(concept_id)
        legacy_sample_rechecks.append({
            "sample_item": sample_id,
            "concept_id": concept_id,
            "vi": source_row["vi_canonical"],
            "en": source_row["en_lemma"],
            "failed_rule": (
                None if survivor is not None else
                dropped_by_id.get(concept_id) or first_empty_rule(per_concept_counts.get(concept_id, {"candidates": 0}))
            ),
            "survived": survivor is not None,
            "NLLB English": survivor["nllb_english"] if survivor is not None else None,
            "NLLB limit hits": survivor["nllb_limit_hits"] if survivor is not None else None,
            "C7 matched type": survivor["c7_match_type"] if survivor is not None else None,
            "C7 matched word": survivor["c7_match_word"] if survivor is not None else None,
        })

    funnel = {
        scope: [
            {
                "stage": stage,
                "n": funnel_counts.get(stage, {}).get(scope, 0),
                "n_by_pos": dict(sorted(stage_pos.get(stage, {}).get(scope, Counter()).items())),
            }
            for stage in ("candidates", *(f"after_{rule}" for rule in RULES))
        ]
        for scope in ("main", "extension")
    }
    demo_set_by_id = {
        row["concept_id"]: set_number
        for set_number, ids in fewshot_sets.items()
        for row in ids
        for _ in [0]
    }
    demo_sources = {
        concept_id: f"fewshot_set_{demo_set_by_id[concept_id]}" if concept_id in demo_set_by_id else "directions"
        for concept_id in demo_ids
    }
    sample_pool = sorted(test_selected.values(), key=lambda item: item["concept_id"])
    sample_n = min(int(settings["survivor_sample_n"]), len(sample_pool))
    rng = np.random.default_rng(int(config["seed"]))
    sample = [sample_pool[index] for index in rng.permutation(len(sample_pool)).tolist()[:sample_n]]
    sample_rows = [
        {
            "concept_id": candidate["concept_id"],
            "vi": candidate["vi"],
            "en": candidate["en"],
            "en_sense_gloss": candidate["en_sense_gloss"],
            "query": candidate["query"],
            "NLLB English": candidate["nllb_english"],
            "NLLB limit hits": candidate["nllb_limit_hits"],
            "C7 matched type": candidate["c7_match_type"],
            "C7 matched word": candidate["c7_match_word"],
            "C7 matching beam count": candidate["c7_beam_match_count"],
            "m1_extension": candidate["m1_extension"],
            "stratum": candidate["stratum"],
        }
        for candidate in sample
    ]
    surviving_test_items = [
        {
            "concept_id": candidate["concept_id"],
            "vi": candidate["vi"],
            "en": candidate["en"],
            "en_sense_gloss": candidate["en_sense_gloss"],
            "query": candidate["query"],
            "NLLB English": candidate["nllb_english"],
            "C7 matched type": candidate["c7_match_type"],
            "C7 matched word": candidate["c7_match_word"],
            "C7 matching beam count": candidate["c7_beam_match_count"],
            "m1_extension": candidate["m1_extension"],
            "stratum": candidate["stratum"],
        }
        for candidate in sorted(test_selected.values(), key=lambda item: item["concept_id"])
    ]
    coverage = {
        "main": {
            "n_survived": len(main_rows),
            "n_test": sum(main_by_stratum_total.values()),
            "by_stratum": {
                stratum: {
                    "n_survived": main_by_stratum_survived.get(stratum, 0),
                    "n_test": count,
                    "percent": 100.0 * main_by_stratum_survived.get(stratum, 0) / count if count else None,
                }
                for stratum, count in sorted(main_by_stratum_total.items())
            },
        },
        "extension": {
            "n_survived": len(extension_rows),
            "n_test": sum(extension_by_stratum_total.values()),
            "by_stratum": {
                stratum: {
                    "n_survived": extension_by_stratum_survived.get(stratum, 0),
                    "n_test": count,
                    "percent": 100.0 * extension_by_stratum_survived.get(stratum, 0) / count if count else None,
                }
                for stratum, count in sorted(extension_by_stratum_total.items())
            },
        },
        "main_sino_percent": 100.0 * main_by_stratum_survived.get("sino", 0) / main_by_stratum_total["sino"] if main_by_stratum_total["sino"] else None,
        "main_nonsino_percent": 100.0 * main_by_stratum_survived.get("nonsino", 0) / main_by_stratum_total["nonsino"] if main_by_stratum_total["nonsino"] else None,
    }
    for candidate in example_candidates:
        # Any concept without examples is represented in the C1 funnel as zero; concept drop rows retain C1 attribution.
        per_concept_counts.setdefault(candidate["concept_id"], {})

    report = {
        "schema_version": 1,
        "rules": list(RULES),
        "candidate_funnel": funnel,
        "final_counts": {
            "main": len(main_rows),
            "extension": len(extension_rows),
            "diac_records": len(test_cloze_rows),
            "nodiac_records": len(test_cloze_nodiac_rows),
            "main_nodiac_records": len(main_nodiac_rows),
            "extension_nodiac_records": len(extension_nodiac_rows),
            "by_stratum": {
                "diac": {
                    "main": dict(sorted(main_by_stratum_survived.items())),
                    "extension": dict(sorted(extension_by_stratum_survived.items())),
                },
                "nodiac": {
                    "main": dict(sorted(main_nodiac_by_stratum.items())),
                    "extension": dict(sorted(extension_nodiac_by_stratum.items())),
                },
            },
        },
        "coverage": coverage,
        "legacy_sample_rechecks": legacy_sample_rechecks,
        "legacy_point_c7_recheck": legacy_point_recheck,
        "demo_concept_ids": demo_ids,
        "demo_sources": demo_sources,
        "demo_count_required": int(settings["demo_count"]),
        "demo_count_available": len(demo_ids),
        "candidate_failure_records": len(failures),
        "dropped_old_concepts": len(dropped),
        "nllb": {
            **nllb_stats,
            "calls_delta": int(getattr(translator, "translation_calls", 0)) - before_calls,
            "cache_hits_delta": int(getattr(translator, "cache_hits", 0)) - before_hits,
            "model": step07_config["model"],
            "revision": step07_config["revision"],
            "num_beams": step07_config["num_beams"],
            "num_return": step07_config["num_return"],
            "max_new_tokens": int(getattr(translator, "max_new_tokens", step07_config["max_new_tokens"])),
        },
        "semantic_rule": (
            "the untruncated top beam must match, and at least "
            f"{int(settings['semantic_min_beam_matches'])} of "
            f"{int(step07_config['num_return'])} return beams must match; "
            "each match uses the target lemma, permitted two-word head lemma, "
            "or same-POS WordNet synset lemma"
        ),
        "semantic_min_beam_matches": int(settings["semantic_min_beam_matches"]),
        "semantic_expected_beam_count": int(step07_config["num_return"]),
        "survivor_sample_seed": int(config["seed"]),
        "survivor_sample": sample_rows,
        "surviving_test_items": surviving_test_items,
        "maximum_main_survivors": int(settings["maximum_main_survivors"]),
        "maximum_extension_survivors": int(settings["maximum_extension_survivors"]),
        "within_previous_upper_bounds": (
            len(main_rows) <= int(settings["maximum_main_survivors"])
            and len(extension_rows) <= int(settings["maximum_extension_survivors"])
        ),
    }

    audit_path = Path(paths["audit_report"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_temp = audit_path.with_name(f".{audit_path.name}.tmp")
    audit_temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_temp.replace(audit_path)

    for scope, stages in funnel.items():
        logger.info("Cloze v1.3 candidate funnel %s: %s", scope, json.dumps(stages, sort_keys=True))
    logger.info("Cloze v1.3 final coverage: %s", json.dumps(coverage, sort_keys=True))
    logger.info("Cloze v1.3 demonstrations: %s; sources=%s", demo_ids, demo_sources)
    logger.info("Cloze v1.3 candidate failures=%d; old concepts dropped=%d; NLLB stats=%s",
                len(failures), len(dropped), json.dumps(report["nllb"], sort_keys=True))

    insufficient_demos = len(demo_ids) < int(settings["demo_count"])
    above_upper_bound = (
        len(main_rows) > int(settings["maximum_main_survivors"])
        or len(extension_rows) > int(settings["maximum_extension_survivors"])
    )
    if insufficient_demos or above_upper_bound:
        return {
            "records": {"diac": [], "nodiac": []},
            "demo_concept_ids": demo_ids,
            "demo_cloze_rows": [],
            "test_cloze_rows": test_cloze_rows,
            "test_nodiac_rows": test_cloze_nodiac_rows,
            "masked_by_id": {row["concept_id"]: test_selected.get(row["concept_id"], {}).get("query") for row in test_rows},
            "report": report,
            "stage_pos": stage_pos,
            "c8_pos": stage_pos["after_C8"],
            "c8_input_by_scope": _stage_counts(c8_input),
            "funnel_counts": funnel_counts,
            "per_concept_counts": per_concept_counts,
            "failure_count": len(failures),
            "main_survivors": len(main_rows),
            "dropped_old_concepts": len(dropped),
            "old_query_count": len(old_rows),
            "demo_sources": demo_sources,
            "stop_reason": "insufficient_cloze_demonstrations" if insufficient_demos else "survivors_above_previous_upper_bound",
        }

    demo_concept_ids = list(demo_ids)
    demo_cloze_rows = [
        {"query": candidate["query"], "vi": rows_by_id[concept_id]["vi_canonical"]}
        for concept_id, candidate in zip(demo_ids, [selected_by_id[concept_id] for concept_id in demo_ids])
    ]
    diac_records: list[dict[str, Any]] = []
    nodiac_records: list[dict[str, Any]] = []
    answer_label = config["prompts"]["cloze"]["answer_label"]
    primary_set = None
    for row in ([] if insufficient_demos else test_cloze_rows):
        candidate = test_selected[row["concept_id"]]
        vi_text = row["vi_canonical"]
        prompt = render_prompt(demo_cloze_rows, query=candidate["query"], answer_label=answer_label)
        diac_records.append(build_record(
            row, prompt=prompt, condition="diac", fewshot_set=primary_set,
            vi_target=vi_text, demo_concept_ids=demo_concept_ids,
        ))
        if not row["collapsed"]:
            nodiac_demos = nodiac_demo_rows(demo_cloze_rows)
            nodiac_query = strip_diacritics(candidate["query"], preserve_case=True)
            nodiac_prompt = render_prompt(nodiac_demos, query=nodiac_query, answer_label=strip_diacritics(answer_label, preserve_case=True))
            nodiac_records.append(build_record(
                row, prompt=nodiac_prompt, condition="nodiac", fewshot_set=primary_set,
                vi_target=strip_diacritics(vi_text, preserve_case=True),
                demo_concept_ids=demo_concept_ids,
            ))

    if not insufficient_demos:
        verify_records(
            {"diac": diac_records, "nodiac": nodiac_records},
            rows_by_id=rows_by_id,
            demo_concept_ids=demo_concept_ids,
            answer_label=answer_label,
            primary_set=primary_set,
        )

    return {
        "records": {"diac": diac_records, "nodiac": nodiac_records},
        "demo_concept_ids": demo_concept_ids,
        "demo_cloze_rows": demo_cloze_rows,
        "test_cloze_rows": test_cloze_rows,
        "test_nodiac_rows": test_cloze_nodiac_rows,
        "masked_by_id": {row["concept_id"]: test_selected.get(row["concept_id"], {}).get("query") for row in test_rows},
        "report": report,
        "stage_pos": stage_pos,
        "c8_pos": stage_pos["after_C8"],
        "c8_input_by_scope": _stage_counts(c8_input),
        "funnel_counts": funnel_counts,
        "per_concept_counts": per_concept_counts,
        "failure_count": len(failures),
        "main_survivors": len(main_rows),
        "dropped_old_concepts": len(dropped),
        "old_query_count": len(old_rows),
        "demo_sources": demo_sources,
        "stop_reason": (
            "insufficient_cloze_demonstrations" if insufficient_demos
            else "survivors_above_previous_upper_bound"
            if len(main_rows) > int(settings["maximum_main_survivors"])
            or len(extension_rows) > int(settings["maximum_extension_survivors"])
            else None
        ),
    }


def record_candidate_dropflow(flow: Any, result: dict[str, Any]) -> None:
    """Append candidate-level C1–C8 dropflow counts for main and extension rows."""
    counts = result["funnel_counts"]
    stage_pos = result["stage_pos"]
    for scope in ("main", "extension"):
        previous_stage = "candidates"
        for rule in RULES:
            current_stage = f"after_{rule}"
            n_in = counts.get(previous_stage, {}).get(scope, 0)
            n_out = counts.get(current_stage, {}).get(scope, 0)
            by_pos = dict(sorted(stage_pos.get(current_stage, {}).get(scope, Counter()).items()))
            flow.record(
                step="12", stage=f"cloze_v13_{rule}_{scope}", unit="items",
                n_in=n_in, n_out=n_out, n_out_by_pos=by_pos,
            )
            previous_stage = current_stage
