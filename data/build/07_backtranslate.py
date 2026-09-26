"""Run cached NLLB round-trip filtering and finalize multilingual canonical forms."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from opencc import OpenCC
from wordfreq import zipf_frequency
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from common import DropflowLogger, load_config, normalize_nfc, normalize_strings, normalized_levenshtein, setup_logging, strip_diacritics, vi_orth_key
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, load_config, normalize_nfc, normalize_strings, normalized_levenshtein, setup_logging, strip_diacritics, vi_orth_key


STEP = "07_backtranslate"
SPLITS = ("fewshot_reservoir", "directions", "test")
REQUEST_ORDER = ("vi_to_en", "en_to_vi", "en_to_other")


def norm(value: str, *, language: str, pos: str, opencc: OpenCC | None = None) -> str:
    """Normalize a translation for lexical comparison using the configured language rules."""
    if not isinstance(value, str):
        raise TypeError(f"norm expects str, got {type(value).__name__}")
    result = normalize_nfc(value).casefold().strip()
    while result and (result[0].isspace() or unicodedata.category(result[0]).startswith("P")):
        result = result[1:].lstrip()
    while result and (result[-1].isspace() or unicodedata.category(result[-1]).startswith("P")):
        result = result[:-1].rstrip()
    if result.endswith("."):
        result = result[:-1].rstrip()
    if language == "fr":
        result = re.sub(r"^(?:le|la|les|un|une|des)\s+", "", result, count=1)
        result = re.sub(r"^l['’]\s*", "", result, count=1)
    elif language == "zh":
        converter = opencc or OpenCC("t2s")
        result = normalize_nfc(converter.convert(result))
        if pos == "adj" and result.endswith("的"):
            result = result[:-1].rstrip()
    return result


def pass_rule(canonical: str, alternatives: list[str], outputs: list[str], *, language: str, pos: str, opencc: OpenCC) -> tuple[bool, str]:
    """Return whether a back-translation matches the canonical form or an alternative."""
    output_norms = {norm(output, language=language, pos=pos, opencc=opencc) for output in outputs}
    canonical_norm = norm(canonical, language=language, pos=pos, opencc=opencc)
    if canonical_norm in output_norms:
        return True, "canonical"
    if any(norm(alt, language=language, pos=pos, opencc=opencc) in output_norms for alt in alternatives):
        return True, "alt"
    return False, "none"


def norm_vi(value: str) -> str:
    """NFC/casefold Vietnamese output, remove punctuation, and collapse whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"norm_vi expects str, got {type(value).__name__}")
    text = normalize_nfc(value).casefold()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return " ".join(text.split())


def contains(target: str, output: str, *, max_extra_syllables: int) -> bool:
    """Test whether target syllables form a bounded-length contiguous output span."""
    if type(max_extra_syllables) is not int or max_extra_syllables < 0:
        raise ValueError("max_extra_syllables must be a non-negative integer")
    target_tokens = vi_orth_key(norm_vi(target)).split()
    output_tokens = vi_orth_key(norm_vi(output)).split()
    if not target_tokens or len(output_tokens) > len(target_tokens) + max_extra_syllables:
        return False
    width = len(target_tokens)
    return any(output_tokens[index:index + width] == target_tokens for index in range(len(output_tokens) - width + 1))


def alt_only_match(canonical: str, alternatives: list[str], outputs: list[str], *, max_extra_syllables: int) -> tuple[str, str] | None:
    """Return the first matching alternative only when canonical has no match."""
    if any(contains(canonical, output, max_extra_syllables=max_extra_syllables) for output in outputs):
        return None
    for alternative in alternatives:
        for output in outputs:
            if contains(alternative, output, max_extra_syllables=max_extra_syllables):
                return alternative, output
    return None


def promote_vi_alternative(row: dict[str, Any], alternative: str) -> tuple[str, str]:
    """Promote an alternative and transfer its VI-entry metadata to the row."""
    old = row["vi_canonical"]
    groups = row.get("vi_variants")
    if not isinstance(groups, list):
        raise ValueError(f"Concept {row.get('concept_id')!r} lacks VI orthographic groups")
    key = vi_orth_key(alternative)
    matches = [group for group in groups if isinstance(group, dict) and group.get("orth_key") == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one VI orthographic group for promoted form {alternative!r}; found {len(matches)}")
    group = matches[0]
    display = group.get("display")
    senses = group.get("n_senses_vi")
    entries = group.get("n_vi_entries")
    if not isinstance(display, str) or not isinstance(senses, list) or type(entries) is not int or entries != len(senses):
        raise ValueError(f"Malformed VI orthographic group for promotion: {group!r}")
    row["vi_canonical"] = display
    row["vi_alts"] = [old, *(form for form in row["vi_alts"] if vi_orth_key(form) != key and vi_orth_key(form) != vi_orth_key(old))]
    row["n_senses_vi"] = sum(senses)
    row["n_vi_entries"] = entries
    row["n_sources"] = int(group["n_sources"])
    row["external_attested"] = bool(group["external_attested"])
    row["loan_templates_found"] = group["loan_templates_found"]
    row["canonical_promoted"] = True
    row["vi_promoted_from"] = old
    return old, display


def clean_second_hop(value: str, *, pos: str) -> str:
    """Remove terminal punctuation and leading English articles/verb marker."""
    text = normalize_nfc(value).strip()
    while text and unicodedata.category(text[-1]).startswith("P"):
        text = text[:-1].rstrip()
    text = re.sub(r"^(?:the|a|an)\s+", "", text, count=1, flags=re.IGNORECASE)
    if pos == "verb":
        text = re.sub(r"^to\s+", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def containment_match(targets: list[str], outputs: list[str], *, max_extra_syllables: int) -> tuple[str, str | None]:
    """Return canonical/alt/none and the first output containing any target."""
    if targets and any(contains(targets[0], output, max_extra_syllables=max_extra_syllables) for output in outputs):
        return "canonical", next(output for output in outputs if contains(targets[0], output, max_extra_syllables=max_extra_syllables))
    for target in targets[1:]:
        for output in outputs:
            if contains(target, output, max_extra_syllables=max_extra_syllables):
                return "alt", output
    return "none", None


def select_part_b(provisional: str, alternatives: list[str], outputs: list[str], *, language: str, pos: str, opencc: OpenCC) -> tuple[str, bool, bool, list[str]]:
    """Select the candidate with the earliest matching NLLB beam; ties retain provisional order."""
    candidates = [provisional, *alternatives]
    beam_rank: dict[str, int] = {}
    for rank, output in enumerate(outputs):
        key = norm(output, language=language, pos=pos, opencc=opencc)
        if key and key not in beam_rank:
            beam_rank[key] = rank
    matches = [(beam_rank[norm(candidate, language=language, pos=pos, opencc=opencc)], index, candidate)
               for index, candidate in enumerate(candidates)
               if norm(candidate, language=language, pos=pos, opencc=opencc) in beam_rank]
    if not matches:
        return provisional, False, False, list(alternatives)
    _, chosen_index, chosen = min(matches, key=lambda item: (item[0], item[1]))
    remaining = [candidate for index, candidate in enumerate(candidates) if index != chosen_index]
    return chosen, True, chosen != provisional, remaining


class TranslationProvider(Protocol):
    """Batched translator interface used by the pipeline and fixture tests."""

    def translate_many(self, requests: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], list[str]]: ...


class NLLBTranslator:
    """Pinned-revision NLLB model with per-input content-addressed translation caching."""

    def __init__(self, config: dict[str, Any], device: torch.device, logger: Any) -> None:
        self.model_id = config["model"]
        self.revision = config["revision"]
        self.device = device
        self.num_beams = int(config["num_beams"])
        self.num_return = int(config["num_return"])
        self.max_new_tokens = int(config["max_new_tokens"])
        self.batch_size = int(config["batch_size"])
        self.do_sample = config["do_sample"]
        self.progress_every = int(config["progress_every"])
        self.cache_dir = Path(config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, revision=self.revision)
        self.model.to(device)
        self.model.eval()
        self.logger = logger
        self.translation_calls = 0
        self.cache_hits = 0
        self.in_memory_reuses = 0
        self.generation_batches = 0
        if self.num_beams != self.num_return:
            raise ValueError("NLLB num_beams must equal num_return for this step")

    def _payload(self, source: str, target: str, text: str) -> dict[str, Any]:
        target_id = self.tokenizer.convert_tokens_to_ids(target)
        if target_id is None or target_id == self.tokenizer.unk_token_id:
            raise ValueError(f"NLLB tokenizer does not recognize target language code {target!r}")
        return {
            "model": self.model_id, "revision": self.revision, "source": source,
            "target": target, "text": normalize_nfc(text),
            "generation": {
                "num_beams": self.num_beams, "num_return_sequences": self.num_return,
                "max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample,
                "forced_bos_token_id": int(target_id),
            },
        }

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"

    def _generate_batch(self, source: str, target: str, texts: list[str]) -> list[list[str]]:
        self.tokenizer.src_lang = source
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = encoded.to(self.device)
        target_id = self.tokenizer.convert_tokens_to_ids(target)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, num_beams=self.num_beams, num_return_sequences=self.num_return,
                max_new_tokens=self.max_new_tokens, do_sample=self.do_sample,
                forced_bos_token_id=target_id,
            )
        decoded = [normalize_nfc(text) for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
        expected = len(texts) * self.num_return
        if len(decoded) != expected:
            raise RuntimeError(f"NLLB returned {len(decoded)} sequences for {len(texts)} inputs; expected {expected}")
        return [decoded[index:index + self.num_return] for index in range(0, expected, self.num_return)]

    def _write_cache(self, path: Path, payload: dict[str, Any], outputs: list[str]) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=".nllb.", suffix=".json", delete=False) as handle:
                temporary = handle.name
                json.dump({"payload": payload, "outputs": outputs}, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def translate_many(self, requests: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], list[str]]:
        """Resolve each (source, target, text) request from cache or batched generation."""
        payloads: dict[tuple[str, str, str], dict[str, Any]] = {}
        for source, target, text in requests:
            key = (source, target, normalize_nfc(text))
            payloads.setdefault(key, self._payload(source, target, text))
        self.in_memory_reuses += len(requests) - len(payloads)
        results: dict[tuple[str, str, str], list[str]] = {}
        misses: dict[tuple[str, str, str], tuple[dict[str, Any], Path]] = {}
        for key in sorted(payloads):
            payload = payloads[key]
            path = self._cache_path(payload)
            if path.exists():
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid NLLB cache file {path}: {exc}") from exc
                if cached.get("payload") != payload or not isinstance(cached.get("outputs"), list) or len(cached["outputs"]) != self.num_return or any(not isinstance(item, str) for item in cached["outputs"]):
                    raise ValueError(f"NLLB cache content does not match its key or expected output schema: {path}")
                results[key] = [normalize_nfc(item) for item in cached["outputs"]]
                self.cache_hits += 1
            else:
                misses[key] = (payload, path)
        grouped: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for key in misses:
            grouped[(key[0], key[1])].append(key)
        total_misses = len(misses)
        completed_misses = 0
        next_report = self.progress_every
        if total_misses:
            self.logger.info("NLLB cache lookup: %d unique requests, %d disk hits, %d new translations", len(payloads), self.cache_hits, total_misses)
        for source, target in sorted(grouped):
            keys = sorted(grouped[(source, target)])
            for offset in range(0, len(keys), self.batch_size):
                batch_keys = keys[offset:offset + self.batch_size]
                batch_outputs = self._generate_batch(source, target, [key[2] for key in batch_keys])
                self.generation_batches += 1
                self.translation_calls += len(batch_keys)
                for key, outputs in zip(batch_keys, batch_outputs, strict=True):
                    payload, path = misses[key]
                    self._write_cache(path, payload, outputs)
                    results[key] = outputs
                completed_misses += len(batch_keys)
                if completed_misses >= next_report or completed_misses == total_misses:
                    self.logger.info("NLLB translation progress: %d/%d input texts; generate batches=%d", completed_misses, total_misses, self.generation_batches)
                    while next_report <= completed_misses:
                        next_report += self.progress_every
        for source, target, text in requests:
            key = (source, target, normalize_nfc(text))
            if key in results:
                continue
            raise RuntimeError(f"NLLB translation result missing for request {key!r}")
        return results


def syllable_bucket(word: str, edges: list[int]) -> str:
    """Bucket a Vietnamese surface by whitespace-delimited syllable count."""
    if len(edges) != 2 or any(type(edge) is not int for edge in edges) or edges != sorted(set(edges)) or edges[0] < 1:
        raise ValueError(f"nllb report syllable bins must be two ascending positive integers, got {edges!r}")
    count = len(normalize_nfc(word).strip().split())
    if count <= edges[0]:
        return str(edges[0])
    if count <= edges[1]:
        return str(edges[1])
    return f"{edges[1] + 1}+"


def process_rows(rows: list[dict[str, Any]], *, translator: TranslationProvider, config: dict[str, Any], logger: Any) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Apply VI round-trip filter and finalize zh/fr/id candidates for passing concepts."""
    code = config["lang_codes"]
    converter = OpenCC("t2s")
    ordered_rows = sorted(rows, key=lambda row: row["concept_id"])
    for row in ordered_rows:
        for field in ("concept_id", "en_lemma", "pos", "split", "vi_canonical"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Step-06 row has missing or malformed {field}: {row!r}")
        if row["split"] not in SPLITS or not isinstance(row.get("vi_alts"), list):
            raise ValueError(f"Step-06 row has unexpected split or vi_alts shape: {row['concept_id']!r}")
        for language in ("zh", "fr", "id"):
            if not isinstance(row.get(f"{language}_canonical"), str) or not isinstance(row.get(f"{language}_alts"), list):
                raise ValueError(f"Step-06 row has missing {language} candidate fields: {row['concept_id']!r}")

    en_requests = [(code["vi"], code["en"], row["vi_canonical"]) for row in ordered_rows]
    en_results = translator.translate_many(en_requests)
    per_row_en: dict[str, list[str]] = {row["concept_id"]: en_results[(code["vi"], code["en"], row["vi_canonical"])] for row in ordered_rows}

    back_requests: list[tuple[str, str, str]] = []
    second_hop_texts: dict[str, list[str]] = {}
    for row in ordered_rows:
        variants = []
        for output in per_row_en[row["concept_id"]]:
            for variant in (output, clean_second_hop(output, pos=row["pos"])):
                if variant and variant not in variants:
                    variants.append(variant)
                    back_requests.append((code["en"], code["vi"], variant))
        second_hop_texts[row["concept_id"]] = variants
    back_results = translator.translate_many(back_requests)

    forward_requests = [(code["en"], code["vi"], row["en_lemma"]) for row in ordered_rows]
    forward_results = translator.translate_many(forward_requests)

    pass_by_id: dict[str, bool] = {}
    route_counts: Counter[str] = Counter()
    promotions: list[dict[str, str]] = []
    en_hits = 0
    max_extra = int(config["bt"]["max_extra_syllables"])
    for row in ordered_rows:
        en_outs = per_row_en[row["concept_id"]]
        original_outputs = [item for output in en_outs for item in back_results[(code["en"], code["vi"], output)]]
        vi_back = list(dict.fromkeys(item for text in second_hop_texts[row["concept_id"]] for item in back_results[(code["en"], code["vi"], text)]))
        fwd_outs = forward_results[(code["en"], code["vi"], row["en_lemma"])]
        targets = [row["vi_canonical"], *row["vi_alts"]]
        old_canonical = row["vi_canonical"]
        alternatives = list(row["vi_alts"])
        strict_pass, strict_match = pass_rule(old_canonical, alternatives, original_outputs, language="vi", pos=row["pos"], opencc=converter)
        rt_match, rt_output = containment_match(targets, vi_back, max_extra_syllables=max_extra)
        fwd_match, fwd_output = containment_match(targets, fwd_outs, max_extra_syllables=max_extra)
        route = "strict" if strict_pass else "rt_contain" if rt_match != "none" else "fwd_only" if fwd_match != "none" else "none"
        passed = route != "none"
        match = rt_match if rt_match != "none" else fwd_match
        matching_output = rt_output if rt_output is not None else fwd_output
        top1_pass = bool(original_outputs and norm(original_outputs[0], language="vi", pos=row["pos"], opencc=converter) == norm(old_canonical, language="vi", pos=row["pos"], opencc=converter))
        en_hit = norm(row["en_lemma"], language="en", pos=row["pos"], opencc=converter) in {norm(item, language="en", pos=row["pos"], opencc=converter) for item in en_outs}
        row.update({
            "en_outs": en_outs, "vi_back": vi_back, "fwd_outs": fwd_outs,
            "bt_pass": passed, "bt_pass_strict_v1": strict_pass, "rt_contain": rt_match != "none",
            "fwd_hit": fwd_match != "none", "bt_route": route, "bt_matching_output": matching_output,
            "bt_match": match, "rt_match": rt_match, "fwd_match": fwd_match,
            "bt_pass_top1": top1_pass, "en_hit": en_hit,
            "canonical_promoted": False, "vi_promoted_from": None,
        })
        canonical_hit = any(contains(old_canonical, output, max_extra_syllables=max_extra) for output in [*vi_back, *fwd_outs])
        promotion_match = alt_only_match(old_canonical, alternatives, vi_back, max_extra_syllables=max_extra)
        if promotion_match is None:
            promotion_match = alt_only_match(old_canonical, alternatives, fwd_outs, max_extra_syllables=max_extra)
        if passed and not canonical_hit and promotion_match is not None:
            old, new = promote_vi_alternative(row, promotion_match[0])
            promotions.append({"concept_id": row["concept_id"], "en_lemma": row["en_lemma"], "old": old, "new": new})
        pass_by_id[row["concept_id"]] = passed
        route_counts[route] += 1
        en_hits += int(en_hit)

    survivors = [row for row in ordered_rows if pass_by_id[row["concept_id"]] or row["split"] == "directions"]
    other_requests = [
        (code["en"], code[language], row["en_lemma"])
        for row in survivors for language in ("zh", "fr", "id")
    ]
    other_results = translator.translate_many(other_requests)
    for row in survivors:
        for language in ("zh", "fr", "id"):
            outputs = other_results[(code["en"], code[language], row["en_lemma"])]
            chosen, agrees, changed, alternatives = select_part_b(
                row[f"{language}_canonical"], row[f"{language}_alts"], outputs,
                language=language, pos=row["pos"], opencc=converter,
            )
            row[f"{language}_outs"] = outputs
            row[f"{language}_nllb_agree"] = agrees
            row[f"{language}_canonical_changed"] = changed
            row[f"{language}_canonical"] = chosen
            row[f"{language}_alts"] = alternatives

    for row in ordered_rows:
        if not pass_by_id[row["concept_id"]] and row["split"] != "directions":
            for language in ("zh", "fr", "id"):
                row[f"{language}_outs"] = None
                row[f"{language}_nllb_agree"] = None
                row[f"{language}_canonical_changed"] = None
    info = {"pass_by_id": pass_by_id, "route_counts": route_counts, "en_hits": en_hits, "survivors": survivors, "promotions": promotions}
    return ordered_rows, {split: [row for row in ordered_rows if row["split"] == split] for split in SPLITS}, info


def _postpromo_surface_norm(value: str, *, vietnamese: bool, config: dict[str, Any]) -> str:
    """Apply the exact configured step-06 surface normalization to a form."""
    result = strip_diacritics(normalize_nfc(value)).lower()
    for character in config["remove_chars"]:
        result = result.replace(character, "")
    if vietnamese:
        for source, target in sorted(config["vi_replace"].items(), key=lambda item: (-len(item[0]), item[0])):
            result = result.replace(source, target)
    return result


def _counts(rows: list[dict[str, Any]]) -> Counter[str]:
    """Count rows by part of speech for dropflow reporting."""
    return Counter(row["pos"] for row in rows)


def _postpromo_surface_distance(row: dict[str, Any], filters: dict[str, Any]) -> tuple[float, str, str, float]:
    """Recompute step-06 min/raw surface distances for a promoted VI canonical."""
    norm_config = filters["surface_norm"]
    compare = [("en", row["en_lemma"])]
    for language in ("fr", "id"):
        candidates = row.get(f"{language}_cands")
        if not isinstance(candidates, list):
            raise ValueError(f"Promoted concept {row['concept_id']!r} has malformed {language}_cands")
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("word"), str):
                raise ValueError(f"Promoted concept {row['concept_id']!r} has malformed {language} candidate")
            word = normalize_nfc(candidate["word"].strip())
            if not word:
                raise ValueError(f"Promoted concept {row['concept_id']!r} has an empty {language} candidate")
            key = word.casefold()
            if key not in seen:
                seen.add(key)
                compare.append((language, word))
    vi_norm = _postpromo_surface_norm(row["vi_canonical"], vietnamese=True, config=norm_config)
    choices = []
    for language_index, language in enumerate(("en", "fr", "id")):
        for item_language, form in compare:
            if item_language != language:
                continue
            candidate_norm = _postpromo_surface_norm(form, vietnamese=False, config=norm_config)
            distance = normalized_levenshtein(vi_norm, candidate_norm)
            raw_distance = normalized_levenshtein(
                strip_diacritics(row["vi_canonical"]).lower(), strip_diacritics(form).lower(),
            )
            choices.append((distance, language_index, language, form, raw_distance))
    if not choices:
        raise ValueError(f"Promoted concept {row['concept_id']!r} has no surface-comparison candidates")
    distance, _, language, form, raw_distance = min(choices)
    return distance, language, form, raw_distance


def _postpromo_deduplicate(rows: list[dict[str, Any]], field: str, zipf_cache: dict[str, float]) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Apply step-06 attestation ranking to one split after canonical promotion."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row["vi_canonical"] if field == "vi_canonical" else row["en_lemma"]
        key = vi_orth_key(value) if field == "vi_canonical" else normalize_nfc(value).casefold()
        groups[key].append(row)

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        lemma = normalize_nfc(row["en_lemma"])
        frequency = zipf_cache.setdefault(lemma, float(zipf_frequency(lemma, "en")))
        return (-int(row["n_sources"]), -int(bool(row["external_attested"])), int(row["n_senses_vi"]), -frequency, row["concept_id"])

    kept: list[dict[str, Any]] = []
    actions: list[tuple[str, str, str]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=rank)
        kept.append(ordered[0])
        actions.extend((field, ordered[0]["concept_id"], row["concept_id"]) for row in ordered[1:])
    return sorted(kept, key=lambda row: row["concept_id"]), actions


def post_promotion_filters(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, tuple[int, Counter[str], int, Counter[str]]]], list[tuple[str, str, str, str]]]:
    """Reapply 3/4/4b/4c only to promoted forms, then deduplicate each whole split."""
    filters = config["filters"]
    threshold = float(filters["surface_edit_threshold"])
    loan = filters["loanword"]
    drop_codes = {str(code).casefold() for code in loan["drop_source_langs"]}
    prefixes = [str(prefix).casefold() for prefix in loan["drop_source_lang_prefixes"]]
    templates = set(loan["loan_templates"])
    zipf_cache: dict[str, float] = {}
    stage_names = ("filter3_polysemy", "filter4_surface", "filter4b_loanword", "filter4c_hyphen", "dedup_vi_form", "dedup_en_lemma")
    flow: dict[str, dict[str, tuple[int, Counter[str], int, Counter[str]]]] = {}
    final: dict[str, list[dict[str, Any]]] = {}
    dedup_actions: list[tuple[str, str, str, str]] = []

    for split in SPLITS:
        current = [row for row in rows_by_split[split] if split == "directions" or row["bt_pass"]]
        flow[split] = {}
        for stage in stage_names[:4]:
            before = current
            kept = []
            for row in before:
                reject = False
                if row["canonical_promoted"]:
                    if stage == "filter3_polysemy":
                        cutoff = row.get("polysemy_cutoff")
                        if not isinstance(cutoff, (int, float)):
                            raise ValueError(f"Promoted concept {row['concept_id']!r} lacks its step-06 polysemy cutoff")
                        reject = int(row["n_senses_vi"]) > cutoff
                    elif stage == "filter4_surface":
                        distance, closest_lang, closest_form, raw_distance = _postpromo_surface_distance(row, filters)
                        row.update({"min_surface_dist": distance, "closest_lang": closest_lang, "closest_form": closest_form, "raw_surface_dist": raw_distance})
                        reject = distance < threshold
                    elif stage == "filter4b_loanword":
                        for item in row.get("loan_templates_found", []):
                            source = str(item.get("source_lang", "")).casefold()
                            template = item.get("template")
                            if template in templates and (source in drop_codes or any(source.startswith(prefix) for prefix in prefixes)):
                                reject = True
                                break
                    elif stage == "filter4c_hyphen":
                        reject = "-" in row["vi_canonical"]
                if reject:
                    row["post_promotion_drop_reason"] = stage
                else:
                    kept.append(row)
            current = kept
            flow[split][stage] = (len(before), _counts(before), len(current), _counts(current))

        before_vi = current
        after_vi, vi_actions = _postpromo_deduplicate(before_vi, "vi_canonical", zipf_cache)
        for _, kept_id, dropped_id in vi_actions:
            next(row for row in before_vi if row["concept_id"] == dropped_id)["post_promotion_drop_reason"] = "dedup_vi_form"
        flow[split]["dedup_vi_form"] = (len(before_vi), _counts(before_vi), len(after_vi), _counts(after_vi))
        before_en = after_vi
        after_en, en_actions = _postpromo_deduplicate(before_en, "en_lemma", zipf_cache)
        for _, kept_id, dropped_id in en_actions:
            next(row for row in before_en if row["concept_id"] == dropped_id)["post_promotion_drop_reason"] = "dedup_en_lemma"
        flow[split]["dedup_en_lemma"] = (len(before_en), _counts(before_en), len(after_en), _counts(after_en))
        final[split] = after_en
        dedup_actions.extend((split, *action) for action in (*vi_actions, *en_actions))

    included_ids = {row["concept_id"] for split in SPLITS for row in final[split]}
    seen: dict[str, dict[str, str]] = {"concept_id": {}, "vi": {}, "en": {}}
    for split in SPLITS:
        for row in final[split]:
            terms = {
                "concept_id": row["concept_id"],
                "vi": vi_orth_key(row["vi_canonical"]),
                "en": normalize_nfc(row["en_lemma"]).casefold(),
            }
            for field, value in terms.items():
                previous = seen[field].setdefault(value, split)
                if previous != split:
                    raise AssertionError(f"Post-promotion {field} collision across splits: {value!r} in {previous!r}/{split!r}")
    for split_rows in rows_by_split.values():
        for row in split_rows:
            row.setdefault("post_promotion_drop_reason", None)
            row["post_promotion_survives"] = row["concept_id"] in included_ids
    return final, flow, dedup_actions


def choose_device() -> torch.device:
    """Select CUDA, then Apple MPS, then CPU as required by the project."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _without_step_dropflow(path: Path, step: str) -> None:
    if not path.exists():
        return
    kept = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed dropflow record at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("step"), str):
            raise ValueError(f"Unexpected dropflow record at {path}:{line_number}")
        if record["step"] != step:
            kept.append(line if line.endswith("\n") else line + "\n")
    path.write_text("".join(kept), encoding="utf-8", newline="\n")


def _output_schema(schema: pa.Schema) -> pa.Schema:
    additions = [
        pa.field("en_outs", pa.list_(pa.string())), pa.field("vi_back", pa.list_(pa.string())),
        pa.field("bt_pass", pa.bool_()), pa.field("bt_match", pa.string()),
        pa.field("bt_pass_strict_v1", pa.bool_()), pa.field("rt_contain", pa.bool_()),
        pa.field("fwd_hit", pa.bool_()), pa.field("bt_route", pa.string()),
        pa.field("bt_matching_output", pa.string()), pa.field("fwd_outs", pa.list_(pa.string())),
        pa.field("rt_match", pa.string()), pa.field("fwd_match", pa.string()),
        pa.field("canonical_promoted", pa.bool_()), pa.field("vi_promoted_from", pa.string()),
        pa.field("post_promotion_survives", pa.bool_()), pa.field("post_promotion_drop_reason", pa.string()),
        pa.field("bt_pass_top1", pa.bool_()), pa.field("en_hit", pa.bool_()),
    ]
    for language in ("zh", "fr", "id"):
        additions.extend([
            pa.field(f"{language}_outs", pa.list_(pa.string())),
            pa.field(f"{language}_nllb_agree", pa.bool_()),
            pa.field(f"{language}_canonical_changed", pa.bool_()),
        ])
    existing = set(schema.names)
    return pa.schema([*schema, *(field for field in additions if field.name not in existing)])


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path, row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as handle:
            temporary = handle.name
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd", version="2.6", row_group_size=row_group_size, write_statistics=True)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def run(config_path: str) -> dict[str, Any]:
    """Run step 07 end-to-end, including the checkpoint filter and B canonicalization."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging("07_backtranslate", config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    try:
        nllb = config["nllb"]
        settings = config["backtranslate"]
        for key in ("model", "revision", "num_beams", "num_return", "max_new_tokens", "batch_size", "cache_dir", "lang_codes", "do_sample"):
            if key not in nllb:
                raise ValueError(f"Missing config key nllb.{key}")
        if not isinstance(nllb["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", nllb["revision"]):
            raise ValueError(f"nllb.revision must be a pinned 40-character commit hash, got {nllb['revision']!r}")
        if nllb["num_beams"] != nllb["num_return"] or int(nllb["num_return"]) != 5:
            raise ValueError("Step 07 requires config num_beams=num_return=5")
        if int(nllb["max_new_tokens"]) < 1 or int(nllb["batch_size"]) < 1 or nllb["do_sample"] is not False:
            raise ValueError("Invalid NLLB generation config")
        nllb = {**nllb, "progress_every": int(config["logging"]["progress_every"]), "bt": settings["bt"]}
        expected_codes = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "fr": "fra_Latn", "id": "ind_Latn"}
        if nllb["lang_codes"] != expected_codes:
            raise ValueError(f"Unexpected nllb.lang_codes: {nllb['lang_codes']!r}")
        report = settings["report"]
        sample_n = int(report["failure_sample_n"])
        new_pass_sample_n = int(report["new_pass_sample_n"])
        still_fail_sample_n = int(report["still_fail_sample_n"])
        change_n = int(report["change_examples_n"])
        if min(sample_n, new_pass_sample_n, still_fail_sample_n, change_n) < 0:
            raise ValueError("NLLB report sample sizes must be non-negative")
        bt_config = settings.get("bt", {})
        if type(bt_config.get("max_extra_syllables")) is not int or bt_config["max_extra_syllables"] < 0:
            raise ValueError("backtranslate.bt.max_extra_syllables must be a non-negative integer")
        input_path = Path(settings["paths"]["input"])
        table = pq.read_table(input_path)
        required = {"concept_id", "en_lemma", "pos", "split", "vi_canonical", "vi_alts", "vi_variants", "polysemy_cutoff", "n_senses_vi", "n_sources", "external_attested", "loan_templates_found", "fr_cands", "id_cands", "zh_canonical", "zh_alts", "fr_canonical", "fr_alts", "id_canonical", "id_alts"}
        missing = sorted(required - set(table.column_names))
        if missing:
            raise ValueError(f"Step-06 parquet is missing required columns: {missing}")
        rows = [normalize_strings(row) for row in table.to_pylist()]
        if not rows:
            raise ValueError("Step-06 parquet has no concepts")
        if len({row["concept_id"] for row in rows}) != len(rows):
            raise ValueError("Step-06 parquet contains duplicate concept_id values")
        if {row["split"] for row in rows} != set(SPLITS):
            raise ValueError(f"Step-06 split values differ from expected {SPLITS}: {sorted({row['split'] for row in rows})}")
        edges = report["syllable_bins"]
        rng = np.random.default_rng(config["seed"])
        original_canonicals = {
            row["concept_id"]: {language: row[f"{language}_canonical"] for language in ("zh", "fr", "id")}
            for row in rows
        }
        device = choose_device()
        logger.info("Device: %s", device)
        logger.info("Pinned NLLB: %s@%s", nllb["model"], nllb["revision"])
        translator = NLLBTranslator(nllb, device, logger)
        processed, rows_by_split, info = process_rows(rows, translator=translator, config=nllb, logger=logger)
        final_by_split, post_flow, post_dedup_actions = post_promotion_filters(rows_by_split, config=config)

        dropflow_path = Path(config["paths"]["dropflow"])
        _without_step_dropflow(dropflow_path, "07")
        flow = DropflowLogger(dropflow_path)
        for split in SPLITS:
            before = [row for row in processed if row["split"] == split]
            after = before if split == "directions" else [row for row in before if row["bt_pass"]]
            by_pos = Counter(row["pos"] for row in after)
            flow.record(step="07", stage=f"filter2_backtranslation_{split}", unit="concepts", n_in=len(before), n_out=len(after), n_out_by_pos=dict(sorted(by_pos.items())))
            logger.info("filter2_backtranslation_%s | n_in=%d | n_out=%d | n_out_by_pos=%s", split, len(before), len(after), json.dumps(dict(sorted(by_pos.items())), sort_keys=True))
            for stage in ("filter3_polysemy", "filter4_surface", "filter4b_loanword", "filter4c_hyphen", "dedup_vi_form", "dedup_en_lemma"):
                n_in, _, n_out, out_pos = post_flow[split][stage]
                flow.record(step="07", stage=f"{stage}_{split}_post_promotion", unit="concepts", n_in=n_in, n_out=n_out, n_out_by_pos=dict(sorted(out_pos.items())))
                logger.info("%s_%s_post_promotion | n_in=%d | n_out=%d | n_out_by_pos=%s", stage, split, n_in, n_out, json.dumps(dict(sorted(out_pos.items())), sort_keys=True))

        test_rows = rows_by_split["test"]
        logger.info("TEST filter-2 rates by POS x syllable bucket (denominator = concepts):")
        metrics: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in test_rows:
            bucket = syllable_bucket(row["vi_canonical"], edges)
            metrics[(row["pos"], bucket)].append(row)
        for (pos, bucket), group in sorted(metrics.items()):
            n = len(group)
            rates = {key: sum(bool(row[field]) for row in group) / n for key, field in (("strict_v1", "bt_pass_strict_v1"), ("rt_contain", "rt_contain"), ("fwd_hit", "fwd_hit"), ("bt_pass", "bt_pass"))}
            logger.info("%s | %s | n=%d | strict_v1=%.6f | rt_contain=%.6f | fwd_hit=%.6f | bt_pass=%.6f", pos, bucket, n, rates["strict_v1"], rates["rt_contain"], rates["fwd_hit"], rates["bt_pass"])

        logger.info("bt_route distribution: %s", json.dumps(dict(sorted(info["route_counts"].items())), sort_keys=True))
        logger.info("Canonical promotions: %d", len(info["promotions"]))
        logger.info("Canonical promotion examples (old -> new | en):")
        for item in sorted(info["promotions"], key=lambda value: (value["concept_id"], value["old"], value["new"]))[:int(report["promotion_examples_n"])]:
            logger.info("%s -> %s | %s", item["old"], item["new"], item["en_lemma"])
        logger.info("Post-promotion dedup removals: vi_form=%d | en_lemma=%d", sum(item[1] == "vi_canonical" for item in post_dedup_actions), sum(item[1] == "en_lemma" for item in post_dedup_actions))
        total = len(processed)
        logger.info("en_hit rate: %d/%d (%.6f)", info["en_hits"], total, info["en_hits"] / total if total else 0.0)
        newly_passed = [row for row in test_rows if row["bt_pass"] and not row["bt_pass_strict_v1"]]
        if newly_passed:
            chosen = [newly_passed[index] for index in sorted(rng.choice(len(newly_passed), size=min(new_pass_sample_n, len(newly_passed)), replace=False).tolist())]
        else:
            chosen = []
        logger.info("TEST new passers sample=%d of %d (vi | en_lemma | route | matching output):", len(chosen), len(newly_passed))
        for row in chosen:
            logger.info("%s | %s | %s | %s", row["vi_canonical"], row["en_lemma"], row["bt_route"], row["bt_matching_output"])
        failures = [row for row in test_rows if not row["bt_pass"]]
        if failures:
            selected = [failures[index] for index in sorted(rng.choice(len(failures), size=min(still_fail_sample_n, len(failures)), replace=False).tolist())]
        else:
            selected = []
        logger.info("TEST still-fail sample=%d of %d (vi | en_lemma | top 3 fwd | top 3 vi_back):", len(selected), len(failures))
        for row in selected:
            logger.info("%s | %s | %s | %s", row["vi_canonical"], row["en_lemma"], json.dumps(row["fwd_outs"][:3], ensure_ascii=False), json.dumps(row["vi_back"][:3], ensure_ascii=False))

        changed_by_lang: dict[str, list[dict[str, Any]]] = {}
        for language in ("zh", "fr", "id"):
            candidates = info["survivors"]
            agrees = sum(bool(row[f"{language}_nllb_agree"]) for row in candidates)
            changed = sum(bool(row[f"{language}_canonical_changed"]) for row in candidates)
            denom = len(candidates)
            logger.info("Part B %s | agree=%d/%d (%.6f) | canonical_changed=%d/%d (%.6f)", language, agrees, denom, agrees / denom if denom else 0.0, changed, denom, changed / denom if denom else 0.0)
            changed_by_lang[language] = [row for row in candidates if row[f"{language}_canonical_changed"]]
        prioritize = {str(item).casefold() for item in report["prioritize_en_lemmas"]}
        all_changes = [(language, row) for language in ("zh", "fr", "id") for row in changed_by_lang[language]]
        priority = sorted((item for item in all_changes if item[1]["en_lemma"].casefold() in prioritize), key=lambda item: (item[1]["en_lemma"].casefold(), item[0], item[1]["concept_id"]))
        priority_first: list[tuple[str, dict[str, Any]]] = []
        for lemma in report["prioritize_en_lemmas"]:
            matches = [item for item in priority if item[1]["en_lemma"].casefold() == str(lemma).casefold()]
            if matches:
                priority_first.append(matches[0])
        priority_first.extend(item for item in priority if item not in priority_first)
        remaining = [item for item in all_changes if item not in priority]
        if remaining:
            sample_indices = sorted(rng.choice(len(remaining), size=min(max(0, change_n - len(priority_first)), len(remaining)), replace=False).tolist())
            selected_changes = priority_first[:change_n] + [remaining[index] for index in sample_indices]
        else:
            selected_changes = priority_first[:change_n]
        absent_priority = [str(lemma) for lemma in report["prioritize_en_lemmas"] if not any(row["en_lemma"].casefold() == str(lemma).casefold() for _, row in all_changes)]
        logger.info("Prioritized ordinal lemmas with no canonical change: %s", json.dumps(absent_priority, ensure_ascii=False))
        logger.info("Part B canonical changes (en | lang | old -> new):")
        for language, row in selected_changes:
            old = original_canonicals[row["concept_id"]][language]
            logger.info("%s | %s | %s -> %s", row["en_lemma"], language, old, row[f"{language}_canonical"])

        final_test_strata = Counter((row["pos"], syllable_bucket(row["vi_canonical"], edges)) for row in final_by_split["test"])
        logger.info("Final TEST size after filter 2 and post-promotion rechecks by POS x syllable bucket:")
        for (pos, bucket), count in sorted(final_test_strata.items()):
            logger.info("%s | %s | %d", pos, bucket, count)
        logger.info("Final TEST concepts: %d", len(final_by_split["test"]))

        final_rows = sorted(
            (row for split in SPLITS for row in final_by_split[split]),
            key=lambda row: row["concept_id"],
        )
        if any(row.get("post_promotion_survives") is not True for row in final_rows):
            raise AssertionError("Step-07 output selection includes a concept that did not survive post-promotion checks")
        output_path = Path(settings["paths"]["output"])
        _write_parquet(final_rows, _output_schema(table.schema), output_path, int(config["split"]["parquet_row_group_size"]))
        logger.info("Output: %s (%d surviving concepts across all splits)", output_path, len(final_rows))
        logger.info("Translation calls=%d; generation batches=%d; cache hits=%d; in-memory duplicate requests=%d", translator.translation_calls, translator.generation_batches, translator.cache_hits, translator.in_memory_reuses)
        runtime = time.monotonic() - started
        logger.info("Runtime seconds: %.3f", runtime)
        return {"rows": final_rows, "final_by_split": final_by_split, "info": info, "translation_calls": translator.translation_calls, "cache_hits": translator.cache_hits, "runtime_seconds": runtime}
    except Exception:
        logger.exception("Step 07 failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
