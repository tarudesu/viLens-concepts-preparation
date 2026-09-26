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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from common import DropflowLogger, load_config, normalize_nfc, normalize_strings, setup_logging
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, load_config, normalize_nfc, normalize_strings, setup_logging


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
    for row in ordered_rows:
        back_requests.extend((code["en"], code["vi"], output) for output in per_row_en[row["concept_id"]])
    back_results = translator.translate_many(back_requests)

    pass_by_id: dict[str, bool] = {}
    bt_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    en_hits = 0
    for row in ordered_rows:
        en_outs = per_row_en[row["concept_id"]]
        back_by_en = [back_results[(code["en"], code["vi"], output)] for output in en_outs]
        vi_back = [item for group in back_by_en for item in group]
        passed, match = pass_rule(row["vi_canonical"], row["vi_alts"], vi_back, language="vi", pos=row["pos"], opencc=converter)
        top1_pass = bool(back_by_en and back_by_en[0] and norm(back_by_en[0][0], language="vi", pos=row["pos"], opencc=converter) == norm(row["vi_canonical"], language="vi", pos=row["pos"], opencc=converter))
        en_hit = norm(row["en_lemma"], language="en", pos=row["pos"], opencc=converter) in {norm(item, language="en", pos=row["pos"], opencc=converter) for item in en_outs}
        row.update({"en_outs": en_outs, "vi_back": vi_back, "bt_pass": passed, "bt_match": match, "bt_pass_top1": top1_pass, "en_hit": en_hit})
        pass_by_id[row["concept_id"]] = passed
        bt_counts[row["pos"]] += int(passed)
        match_counts[match] += 1
        en_hits += int(en_hit)

    survivors = [row for row in ordered_rows if pass_by_id[row["concept_id"]]]
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
        if not pass_by_id[row["concept_id"]]:
            for language in ("zh", "fr", "id"):
                row[f"{language}_outs"] = None
                row[f"{language}_nllb_agree"] = None
                row[f"{language}_canonical_changed"] = None
    info = {"pass_by_id": pass_by_id, "match_counts": match_counts, "en_hits": en_hits, "survivors": survivors}
    return ordered_rows, {split: [row for row in ordered_rows if row["split"] == split] for split in SPLITS}, info


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
        nllb = {**nllb, "progress_every": int(config["logging"]["progress_every"])}
        expected_codes = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "fr": "fra_Latn", "id": "ind_Latn"}
        if nllb["lang_codes"] != expected_codes:
            raise ValueError(f"Unexpected nllb.lang_codes: {nllb['lang_codes']!r}")
        report = settings["report"]
        sample_n = int(report["failure_sample_n"])
        change_n = int(report["change_examples_n"])
        if sample_n < 0 or change_n < 0:
            raise ValueError("NLLB report sample sizes must be non-negative")
        input_path = Path(settings["paths"]["input"])
        table = pq.read_table(input_path)
        required = {"concept_id", "en_lemma", "pos", "split", "vi_canonical", "vi_alts", "zh_canonical", "zh_alts", "fr_canonical", "fr_alts", "id_canonical", "id_alts"}
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

        dropflow_path = Path(config["paths"]["dropflow"])
        _without_step_dropflow(dropflow_path, "07")
        flow = DropflowLogger(dropflow_path)
        for split in SPLITS:
            before = [row for row in processed if row["split"] == split]
            after = [row for row in before if row["bt_pass"]]
            by_pos = Counter(row["pos"] for row in after)
            flow.record(step="07", stage=f"filter2_backtranslation_{split}", unit="concepts", n_in=len(before), n_out=len(after), n_out_by_pos=dict(sorted(by_pos.items())))
            logger.info("filter2_backtranslation_%s | n_in=%d | n_out=%d | n_out_by_pos=%s", split, len(before), len(after), json.dumps(dict(sorted(by_pos.items())), sort_keys=True))

        test_rows = rows_by_split["test"]
        logger.info("TEST bt_pass rates by POS:")
        by_pos: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        metrics: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in test_rows:
            bucket = syllable_bucket(row["vi_canonical"], edges)
            by_pos[row["pos"]].append(row)
            by_bucket[bucket].append(row)
            metrics[(row["pos"], bucket)].append(row)
        for pos, group in sorted(by_pos.items()):
            n = len(group)
            passed = sum(bool(row["bt_pass"]) for row in group)
            top1 = sum(bool(row["bt_pass_top1"]) for row in group)
            logger.info("%s | n=%d | bt_pass=%d (%.6f) | bt_pass_top1=%d (%.6f)", pos, n, passed, passed / n, top1, top1 / n)
        logger.info("TEST bt_pass rates by syllable bucket:")
        for bucket, group in sorted(by_bucket.items()):
            n = len(group)
            passed = sum(bool(row["bt_pass"]) for row in group)
            top1 = sum(bool(row["bt_pass_top1"]) for row in group)
            logger.info("%s | n=%d | bt_pass=%d (%.6f) | bt_pass_top1=%d (%.6f)", bucket, n, passed, passed / n, top1, top1 / n)
        logger.info("TEST bt_pass rates by POS x syllable bucket:")
        for (pos, bucket), group in sorted(metrics.items()):
            n = len(group)
            passed = sum(bool(row["bt_pass"]) for row in group)
            top1 = sum(bool(row["bt_pass_top1"]) for row in group)
            logger.info("%s | %s | n=%d | bt_pass=%d (%.6f) | bt_pass_top1=%d (%.6f)", pos, bucket, n, passed, passed / n, top1, top1 / n)

        total = len(processed)
        logger.info("bt_match distribution: %s", json.dumps(dict(sorted(info["match_counts"].items())), sort_keys=True))
        logger.info("en_hit rate: %d/%d (%.6f)", info["en_hits"], total, info["en_hits"] / total if total else 0.0)
        failures = [row for row in test_rows if not row["bt_pass"]]
        if failures:
            selected = [failures[index] for index in sorted(rng.choice(len(failures), size=min(sample_n, len(failures)), replace=False).tolist())]
        else:
            selected = []
        logger.info("TEST backtranslation failures sampled=%d of %d (vi | en_outs | top 5 vi_back):", len(selected), len(failures))
        for row in selected:
            logger.info("%s | %s | %s", row["vi_canonical"], json.dumps(row["en_outs"], ensure_ascii=False), json.dumps(row["vi_back"][:5], ensure_ascii=False))

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

        output_path = Path(settings["paths"]["output"])
        _write_parquet(processed, _output_schema(table.schema), output_path, int(config["split"]["parquet_row_group_size"]))
        logger.info("Output: %s (%d concepts)", output_path, len(processed))
        logger.info("Translation calls=%d; generation batches=%d; cache hits=%d; in-memory duplicate requests=%d", translator.translation_calls, translator.generation_batches, translator.cache_hits, translator.in_memory_reuses)
        runtime = time.monotonic() - started
        logger.info("Runtime seconds: %.3f", runtime)
        return {"rows": processed, "info": info, "translation_calls": translator.translation_calls, "cache_hits": translator.cache_hits, "runtime_seconds": runtime}
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
