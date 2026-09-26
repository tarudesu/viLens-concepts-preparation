"""Attest Vietnamese-English candidate pairs against MUSE and Wikidata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

try:  # Direct script execution and package-based tests.
    from common import DropflowLogger, iter_jsonl, load_config, normalize_nfc, setup_logging
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, iter_jsonl, load_config, normalize_nfc, setup_logging


STEP = "04_attest"
VALID_DROP_UNITS = {"entries", "groups", "concepts", "items"}
LEGACY_UNITS = {
    ("03_pool", "raw_groups_pos_filtered"): "entries",
    ("03_pool", "five_way"): "groups",
    ("03_pool", "filter6_missing_vi_entry"): "concepts",
}


def normalize_muse_term(value: str) -> str:
    """Normalize a MUSE term as NFC, trimmed, lowercase text."""
    if not isinstance(value, str):
        raise TypeError(f"MUSE term must be a string, got {type(value).__name__}")
    return normalize_nfc(value.strip()).lower()


def parse_muse_line(line: str, orientation: str) -> tuple[str, str]:
    """Parse one tab-separated MUSE row and return normalized (vi, en)."""
    if orientation not in {"vi-en", "en-vi"}:
        raise ValueError(f"Unexpected MUSE orientation: {orientation!r}")
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 2 or not all(field.strip() for field in fields):
        raise ValueError(f"Malformed {orientation} MUSE row: {line!r}")
    first, second = (normalize_muse_term(field) for field in fields)
    return (first, second) if orientation == "vi-en" else (second, first)


def parse_wikidata_response(payload: dict[str, Any]) -> set[tuple[str, str]]:
    """Parse WDQS JSON bindings into NFC/casefolded (English, Vietnamese) pairs."""
    try:
        bindings = payload["results"]["bindings"]
    except (KeyError, TypeError) as exc:
        raise ValueError("WDQS response lacks results.bindings") from exc
    if not isinstance(bindings, list):
        raise ValueError("WDQS results.bindings must be a list")
    pairs: set[tuple[str, str]] = set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            raise ValueError(f"WDQS binding {index} is not an object")
        try:
            english = binding["enLemma"]["value"]
            vietnamese = binding["viLabel"]["value"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"WDQS binding {index} lacks enLemma/viLabel values") from exc
        if not isinstance(english, str) or not isinstance(vietnamese, str):
            raise ValueError(f"WDQS binding {index} values must be strings")
        pairs.add((normalize_nfc(english).casefold(), normalize_nfc(vietnamese).casefold()))
    return pairs


def contains_whole_word(needle: str, text: str) -> bool:
    """Casefolded whole-word/whole-phrase containment (not substring matching)."""
    if not needle or not text:
        return False
    folded_needle = normalize_nfc(needle).casefold()
    folded_text = normalize_nfc(text).casefold()
    pattern = re.compile(r"(?<!\w)" + re.escape(folded_needle) + r"(?!\w)", re.UNICODE)
    return pattern.search(folded_text) is not None


def muse_contains(pairs: set[tuple[str, str]], vi_word: str, en_lemma: str) -> bool:
    """Return whether the normalized Vietnamese-English pair is in MUSE."""
    return (normalize_muse_term(vi_word), normalize_muse_term(en_lemma)) in pairs


def _fold(value: str) -> str:
    return normalize_nfc(value).casefold()


def syllable_bucket(word: str, edges: list[int]) -> str:
    """Bucket Vietnamese orthographic syllables using configured upper edges."""
    count = len(normalize_nfc(word).strip().split())
    if count <= edges[0]:
        return str(edges[0])
    if count <= edges[1]:
        return str(edges[1])
    return f"{edges[1] + 1}+"


def exact_english_qids(payload: dict[str, Any], lemma: str) -> list[str]:
    """Select Action API search hits with exact English label/alias matches."""
    hits = payload.get("search")
    if not isinstance(hits, list):
        raise ValueError("wbsearchentities response lacks a search list")
    wanted = _fold(lemma)
    qids: set[str] = set()
    for index, hit in enumerate(hits):
        if not isinstance(hit, dict):
            raise ValueError(f"wbsearchentities hit {index} is not an object")
        label = hit.get("label", "")
        aliases = hit.get("aliases", [])
        if not isinstance(label, str) or not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError(f"wbsearchentities hit {index} has malformed label/aliases")
        if any(_fold(value) == wanted for value in [label, *aliases]):
            qid = hit.get("id")
            if not isinstance(qid, str) or not re.fullmatch(r"Q[0-9]+", qid):
                raise ValueError(f"Exact wbsearchentities hit {index} lacks a valid QID")
            qids.add(qid)
    return sorted(qids)


def vietnamese_entity_terms(payload: dict[str, Any], qids: Iterable[str]) -> dict[str, set[str]]:
    """Extract NFC/casefolded Vietnamese labels and aliases for requested QIDs."""
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise ValueError("wbgetentities response lacks an entities object")
    result: dict[str, set[str]] = {}
    for qid in qids:
        entity = entities.get(qid)
        if not isinstance(entity, dict):
            raise ValueError(f"wbgetentities response lacks requested entity {qid}")
        labels = entity.get("labels", {})
        aliases = entity.get("aliases", {})
        if not isinstance(labels, dict) or not isinstance(aliases, dict):
            raise ValueError(f"wbgetentities entity {qid} has malformed labels/aliases")
        terms: set[str] = set()
        label = labels.get("vi")
        if label is not None:
            if not isinstance(label, dict) or not isinstance(label.get("value"), str):
                raise ValueError(f"wbgetentities entity {qid} has malformed vi label")
            terms.add(_fold(label["value"]))
        values = aliases.get("vi", [])
        if not isinstance(values, list):
            raise ValueError(f"wbgetentities entity {qid} has malformed vi aliases")
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("value"), str):
                raise ValueError(f"wbgetentities entity {qid} has malformed vi alias")
            terms.add(_fold(value["value"]))
        result[qid] = terms
    return result


def _retryable_api_error(error: Any) -> bool:
    """Identify transient Action API error codes eligible for backoff."""
    return isinstance(error, dict) and error.get("code") in {
        "maxlag", "ratelimited", "cirrussearch-too-busy-error",
    }


def build_sparql_query(candidate_pairs: Iterable[tuple[str, str]]) -> str:
    """Build exact label/alias lookups constrained to the observed candidate pairs."""
    unique = sorted({(normalize_nfc(lemma), normalize_nfc(candidate)) for lemma, candidate in candidate_pairs})
    if not unique:
        raise ValueError("Cannot build a WDQS query for an empty candidate-pair batch")
    values = " ".join(
        "(" + json.dumps(lemma, ensure_ascii=False) + "@en " + json.dumps(candidate, ensure_ascii=False) + "@vi)"
        for lemma, candidate in unique
    )
    return (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
        "SELECT DISTINCT ?enLemma ?viLabel WHERE {\n"
        f"  VALUES (?enLemma ?viCandidate) {{ {values} }}\n"
        "  { ?item rdfs:label ?enValue . } UNION { ?item skos:altLabel ?enValue . }\n"
        "  FILTER(LANG(?enValue) = \"en\")\n"
        "  FILTER(LCASE(STR(?enValue)) = LCASE(STR(?enLemma)))\n"
        "  VALUES ?viProperty { rdfs:label skos:altLabel }\n"
        "  ?item ?viProperty ?viLabel .\n"
        "  FILTER(LANG(?viLabel) = \"vi\")\n"
        "  FILTER(LCASE(STR(?viLabel)) = LCASE(STR(?viCandidate)))\n"
        "}"
    )


def backfill_dropflow_units(path: str | Path) -> int:
    """Backfill known legacy dropflow records and reject unknown unitless rows."""
    dropflow_path = Path(path)
    if not dropflow_path.exists():
        return 0
    records: list[dict[str, Any]] = []
    changed = 0
    with dropflow_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed dropflow line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Dropflow line {line_number} must be an object")
            if "unit" not in record:
                unit = LEGACY_UNITS.get((record.get("step"), record.get("stage")))
                if unit is None:
                    raise ValueError(f"Cannot infer unit for legacy dropflow record at line {line_number}: {record!r}")
                record["unit"] = unit
                changed += 1
            if record["unit"] not in VALID_DROP_UNITS:
                raise ValueError(f"Invalid dropflow unit at line {line_number}: {record['unit']!r}")
            records.append(record)
    if changed:
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=dropflow_path.parent, delete=False) as target:
                temp_name = target.name
                for record in records:
                    target.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            os.replace(temp_name, dropflow_path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
    return changed


def load_muse_pairs(path: str | Path, orientation: str) -> set[tuple[str, str]]:
    """Stream a MUSE dictionary into normalized (vi, en) pairs."""
    pairs: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                pairs.add(parse_muse_line(line, orientation))
            except ValueError as exc:
                raise ValueError(f"{path} line {line_number}: {exc}") from exc
    return pairs


def load_vietnamese_glosses(path: str | Path) -> dict[tuple[str, str], list[str]]:
    """Stream Vietnamese entries and index glosses by casefolded word and POS."""
    index: dict[tuple[str, str], list[str]] = {}
    for entry_number, entry in enumerate(iter_jsonl(path), start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("word"), str) or not isinstance(entry.get("pos"), str):
            raise ValueError(f"Vietnamese dump entry {entry_number} lacks string word/pos")
        senses = entry.get("senses", [])
        if not isinstance(senses, list) or any(not isinstance(sense, dict) for sense in senses):
            raise ValueError(f"Vietnamese dump entry {entry_number} has malformed senses")
        key = (entry["word"].casefold(), entry["pos"].casefold())
        glosses = index.setdefault(key, [])
        for sense in senses:
            values = sense.get("glosses", [])
            if values is None:
                continue
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"Vietnamese dump entry {entry_number} has malformed senses[].glosses")
            glosses.extend(values)
    return index


def _cache_record(path: Path, query: str, cache_envelope: dict[str, Any]) -> dict[str, str]:
    body = path.read_bytes()
    sha = hashlib.sha256(body).hexdigest()
    return {
        "source": "wikidata",
        "file": str(path),
        "URL": "https://query.wikidata.org/sparql",
        "retrieved UTC": cache_envelope["retrieved_utc"],
        "bytes": str(len(body)),
        "SHA-256": sha,
        "version/commit/extraction date": f"WDQS query date {cache_envelope['query_date']}; query SHA-1 {hashlib.sha1(query.encode('utf-8')).hexdigest()}",
        "license": "CC0 1.0 (Wikidata structured data)",
    }


def update_sources_table(path: str | Path, records: Iterable[dict[str, str]]) -> None:
    """Merge Wikidata cache provenance rows into the existing SOURCES table."""
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    header = "| source | file | URL | retrieved UTC | bytes | SHA-256 | version/commit/extraction date | license |"
    try:
        header_index = lines.index(header)
    except ValueError as exc:
        raise ValueError(f"SOURCES.md has no expected provenance table header: {source_path}") from exc
    existing: dict[str, dict[str, str]] = {}
    tail_index = len(lines)
    for index in range(header_index + 2, len(lines)):
        line = lines[index]
        if not line.startswith("|"):
            tail_index = index
            break
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 8:
            raise ValueError(f"Malformed SOURCES.md row at line {index + 1}")
        existing[columns[1]] = dict(zip(("source", "file", "URL", "retrieved UTC", "bytes", "SHA-256", "version/commit/extraction date", "license"), columns))
    for record in records:
        existing[record["file"]] = record
    field_names = ("source", "file", "URL", "retrieved UTC", "bytes", "SHA-256", "version/commit/extraction date", "license")
    rendered = ["| " + " | ".join(field_names) + " |", "|---|---|---|---|---|---|---|---|"]
    for record in sorted(existing.values(), key=lambda item: (item["source"], item["file"])):
        rendered.append("| " + " | ".join(record[field].replace("|", "\\|") for field in field_names) + " |")
    new_text = "\n".join([*lines[:header_index], *rendered, *lines[tail_index:]]) + "\n"
    if new_text != text:
        source_path.write_text(new_text, encoding="utf-8", newline="\n")


class WikidataClient:
    """Rate-limited WDQS client with query-hash response caching."""

    def __init__(self, config: dict[str, Any], cache_dir: Path, logger: Any) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.logger = logger
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.last_request: float | None = None
        self.requests = 0
        self.cache_hits = 0
        self.failures = 0
        self.retries = 0
        self.cache_records: dict[str, tuple[str, dict[str, Any]]] = {}

    def _pace(self) -> None:
        interval = float(self.config["min_interval_seconds"])
        if self.last_request is not None:
            remaining = interval - (time.monotonic() - self.last_request)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request = time.monotonic()

    def fetch_pairs(self, candidate_pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
        query = build_sparql_query(candidate_pairs)
        query_sha1 = hashlib.sha1(query.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{query_sha1}.json"
        if cache_path.exists():
            envelope = json.loads(cache_path.read_text(encoding="utf-8"))
            if envelope.get("query") != query or not isinstance(envelope.get("response"), dict):
                raise ValueError(f"WDQS cache does not match its query hash: {cache_path}")
            self.cache_hits += 1
            self.cache_records[str(cache_path)] = (query, envelope)
            return parse_wikidata_response(envelope["response"])

        url = self.config["endpoint"]
        max_attempts = int(self.config["max_retries"])
        response_json = None
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            self._pace()
            self.requests += 1
            self.logger.info("WDQS request batch_sha1=%s attempt=%d/%d pairs=%d",
                             query_sha1, attempt + 1, max_attempts, len(candidate_pairs))
            try:
                response = self.session.get(
                    url,
                    params={"query": query, "format": "json"},
                    headers={"Accept": "application/sparql-results+json", "User-Agent": self.config["user_agent"]},
                    timeout=float(self.config["timeout_seconds"]),
                )
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    last_error = RuntimeError(f"WDQS HTTP {response.status_code}")
                    if attempt + 1 < max_attempts:
                        self.retries += 1
                        time.sleep(float(self.config["retry_backoff_seconds"]) * (2 ** attempt))
                        continue
                    break
                if not 200 <= response.status_code < 300:
                    last_error = RuntimeError(f"WDQS HTTP {response.status_code}")
                    break
                response_json = response.json()
                if not isinstance(response_json, dict):
                    raise ValueError("WDQS response root must be an object")
                parse_wikidata_response(response_json)  # Fail closed before caching.
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if isinstance(exc, ValueError) and not isinstance(exc, requests.RequestException):
                    break
                if attempt + 1 < max_attempts:
                    self.retries += 1
                    time.sleep(float(self.config["retry_backoff_seconds"]) * (2 ** attempt))
        if response_json is None:
            self.failures += 1
            raise RuntimeError(f"WDQS query batch failed after {max_attempts} attempts: {last_error}")

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        envelope = {"query": query, "query_date": now[:10], "retrieved_utc": now, "response": response_json}
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=self.cache_dir, delete=False) as temp:
                temp_name = temp.name
                temp.write(encoded)
            os.replace(temp_name, cache_path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        self.cache_records[str(cache_path)] = (query, envelope)
        return parse_wikidata_response(response_json)


class ActionAPIClient:
    """Rate-limited, cached MediaWiki Action API client."""

    def __init__(self, config: dict[str, Any], cache_dir: Path, logger: Any) -> None:
        self.config, self.cache_dir, self.logger = config, cache_dir, logger
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.last_request: float | None = None
        self.requests = self.cache_hits = self.failures = self.retries = 0

    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        canonical = json.dumps({"url": self.config["api_endpoint"], "params": params},
                               ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"api_{digest}.json"
        if cache_path.exists():
            try:
                envelope = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid Action API cache {cache_path}: {exc}") from exc
            if envelope.get("request") != canonical or not isinstance(envelope.get("response"), dict):
                raise ValueError(f"Action API cache does not match request hash: {cache_path}")
            self.cache_hits += 1
            return envelope["response"]

        max_attempts = int(self.config["max_retries"])
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            interval = float(self.config["min_interval_seconds"])
            if self.last_request is not None:
                remaining = interval - (time.monotonic() - self.last_request)
                if remaining > 0:
                    time.sleep(remaining)
            self.last_request = time.monotonic()
            self.requests += 1
            self.logger.info("Action API request sha1=%s action=%s attempt=%d/%d",
                             digest, params.get("action"), attempt + 1, max_attempts)
            try:
                response = self.session.get(
                    self.config["api_endpoint"], params=params,
                    headers={"User-Agent": self.config["user_agent"]},
                    timeout=float(self.config["timeout_seconds"]),
                )
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    last_error = RuntimeError(f"Action API HTTP {response.status_code}")
                    if attempt + 1 < max_attempts:
                        self.retries += 1
                        time.sleep(float(self.config["retry_backoff_seconds"]) * (2 ** attempt))
                        continue
                    break
                if not 200 <= response.status_code < 300:
                    last_error = RuntimeError(f"Action API HTTP {response.status_code}")
                    break
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Action API response root must be an object")
                if "error" in payload:
                    last_error = RuntimeError(f"Action API error: {payload['error']!r}")
                    if _retryable_api_error(payload["error"]) and attempt + 1 < max_attempts:
                        self.retries += 1
                        time.sleep(float(self.config["retry_backoff_seconds"]) * (2 ** attempt))
                        continue
                    break
                now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                envelope = {"request": canonical, "retrieved_utc": now, "query_date": now[:10], "response": payload}
                encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                temp_name = None
                try:
                    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=self.cache_dir, delete=False) as temp:
                        temp_name = temp.name
                        temp.write(encoded)
                    os.replace(temp_name, cache_path)
                finally:
                    if temp_name and os.path.exists(temp_name):
                        os.unlink(temp_name)
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if isinstance(exc, ValueError):
                    break
                if attempt + 1 < max_attempts:
                    self.retries += 1
                    time.sleep(float(self.config["retry_backoff_seconds"]) * (2 ** attempt))
        self.failures += 1
        raise RuntimeError(f"Action API request failed after {max_attempts} attempts ({params.get('action')}): {last_error}")

    def fetch_pairs(
        self, lemmas: list[str], candidate_pairs: set[tuple[str, str]], progress_every: int
    ) -> set[tuple[str, str]]:
        """Search exact English terms, fetch matching entities, return attested pairs."""
        lemma_qids: dict[str, list[str]] = {}
        for index, lemma in enumerate(lemmas, start=1):
            payload = self._get({"action": "wbsearchentities", "search": lemma, "language": "en",
                                 "type": "item", "limit": str(self.config["api_search_limit"]), "format": "json"})
            lemma_qids[lemma] = exact_english_qids(payload, lemma)
            if index % progress_every == 0 or index == len(lemmas):
                self.logger.info("Action API English searches: %d/%d; physical calls=%d cache_hits=%d",
                                 index, len(lemmas), self.requests, self.cache_hits)
        qid_to_lemmas: dict[str, set[str]] = defaultdict(set)
        for lemma, qids in lemma_qids.items():
            for qid in qids:
                qid_to_lemmas[qid].add(lemma)
        qids = sorted(qid_to_lemmas)
        entity_batch_size = int(self.config["entity_batch_size"])
        if not 1 <= entity_batch_size <= 50:
            raise ValueError(f"wikidata.entity_batch_size must be in [1, 50], got {entity_batch_size}")
        qid_terms: dict[str, set[str]] = {}
        last_logged_batch = 0
        for offset in range(0, len(qids), entity_batch_size):
            batch = qids[offset:offset + entity_batch_size]
            payload = self._get({"action": "wbgetentities", "ids": "|".join(batch), "props": "labels|aliases",
                                 "languages": "en|vi", "format": "json"})
            qid_terms.update(vietnamese_entity_terms(payload, batch))
            processed = offset + len(batch)
            if processed // progress_every > last_logged_batch or processed == len(qids):
                last_logged_batch = processed // progress_every
                self.logger.info("Action API entity QIDs: %d/%d; physical calls=%d cache_hits=%d",
                                 processed, len(qids), self.requests, self.cache_hits)
        return {
            (_fold(lemma), vi_term)
            for qid, terms in qid_terms.items()
            for lemma in qid_to_lemmas[qid]
            for vi_term in terms
        }


def _update_dropflow(path: Path, *, n_before: int, n_after: int, by_pos: dict[str, int]) -> None:
    records: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed dropflow line {line_number}: {exc}") from exc
                if record.get("step") != STEP:
                    records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as temp:
            temp_name = temp.name
            for record in records:
                temp.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    DropflowLogger(path).record(
        step=STEP, stage="filter1_source_agreement", unit="concepts",
        n_in=n_before, n_out=n_after, n_out_by_pos=by_pos,
    )


def _source_schema(pool_schema: pa.Schema) -> pa.Schema:
    fields = list(pool_schema.field("vi_cands").type.value_type)
    existing = {field.name for field in fields}
    flags = [
        pa.field("in_wiktextract", pa.bool_()),
        pa.field("in_muse", pa.bool_()),
        pa.field("in_wikidata", pa.bool_()),
        pa.field("in_vi_gloss", pa.bool_()),
        pa.field("external_attested", pa.bool_()),
        pa.field("n_sources", pa.int8()),
    ]
    fields.extend(field for field in flags if field.name not in existing)
    return pool_schema.set(
        pool_schema.get_field_index("vi_cands"),
        pa.field("vi_cands", pa.list_(pa.struct(fields)), nullable=pool_schema.field("vi_cands").nullable),
    )


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as temp:
            temp_name = temp.name
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temp_name, compression="zstd", version="2.6", row_group_size=65536, write_statistics=True)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def run(config_path: str) -> dict[str, Any]:
    """Run attestation, report coverage, and write the filtered parquet."""
    started = time.monotonic()
    config = load_config(config_path)
    paths = config["paths"]
    attest_paths = config["attest"]["paths"]
    settings = config["wikidata"]
    logger = setup_logging(STEP, paths["logs"], level=config.get("logging", {}).get("level", "INFO"))
    client: WikidataClient | ActionAPIClient | None = None
    try:
        batch_size = int(settings["batch_size"])
        if not 1 <= batch_size <= 200:
            raise ValueError(f"wikidata.batch_size must be in [1, 200], got {batch_size}")
        attempts = int(settings["max_retries"])
        if attempts < 1:
            raise ValueError("wikidata.max_retries must be at least one")
        minimum_sources = int(config["filters"]["min_sources"])
        progress_every = int(config["logging"]["progress_every"])
        if progress_every < 1:
            raise ValueError("logging.progress_every must be at least one")
        source_flags = {
            "wiktextract_table": "in_wiktextract",
            "vi_gloss": "in_vi_gloss",
            "muse": "in_muse",
            "wikidata": "in_wikidata",
        }
        attestation_sources = config["filters"]["attestation_sources"]
        if not isinstance(attestation_sources, list) or not attestation_sources or any(name not in source_flags for name in attestation_sources):
            raise ValueError(f"filters.attestation_sources must contain known non-empty source names: {sorted(source_flags)}")
        if not isinstance(settings.get("enabled"), bool):
            raise ValueError("wikidata.enabled must be explicitly true or false")
        backend = settings.get("backend", "api")
        if backend not in {"api", "sparql"}:
            raise ValueError(f"wikidata.backend must be 'api' or 'sparql', got {backend!r}")
        bin_edges = config["attest"]["syllable_bins"]
        if not isinstance(bin_edges, list) or len(bin_edges) != 2 or any(type(x) is not int for x in bin_edges) or bin_edges != sorted(set(bin_edges)) or bin_edges[0] < 1:
            raise ValueError(f"attest.syllable_bins must be two ascending positive integers, got {bin_edges!r}")
        changed_units = backfill_dropflow_units(paths["dropflow"])
        logger.info("Backfilled legacy dropflow units: %d", changed_units)

        pool_path = Path(attest_paths["pool"])
        pool_table = pq.read_table(pool_path)
        pool_rows = pool_table.to_pylist()
        for required in ("concept_id", "en_lemma", "pos", "vi_cands"):
            if required not in pool_table.column_names:
                raise ValueError(f"Pool parquet is missing required column {required!r}")
        if not pool_rows:
            raise ValueError("Pool parquet has no concepts")
        candidate_pairs: set[tuple[str, str]] = set()
        for row in pool_rows:
            if not isinstance(row["vi_cands"], list):
                raise ValueError(f"Pool row has malformed vi_cands: {row['concept_id']}")
            candidate_pairs.update(
                (normalize_nfc(row["en_lemma"]), normalize_nfc(candidate["word"]))
                for candidate in row["vi_cands"]
            )
        lemmas = sorted({lemma for lemma, _ in candidate_pairs})
        muse_pairs = load_muse_pairs(attest_paths["muse_vi_en"], "vi-en")
        muse_pairs.update(load_muse_pairs(attest_paths["muse_en_vi"], "en-vi"))
        muse_vi_forms = {vi for vi, _ in muse_pairs}
        muse_vi_multiword = sum(" " in vi for vi in muse_vi_forms)
        logger.info("MUSE unique VI-side forms containing a space: %d/%d (%.6f)",
                    muse_vi_multiword, len(muse_vi_forms), muse_vi_multiword / len(muse_vi_forms) if muse_vi_forms else 0.0)
        gloss_index = load_vietnamese_glosses(attest_paths["vietnamese_dump"])
        wikidata_pairs: set[tuple[str, str]] = set()
        probe_started = time.perf_counter()
        probe_status: int | str
        probe_query = "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?l WHERE { wd:Q42 rdfs:label ?l } LIMIT 1"
        probe_cache_dir = Path(attest_paths["wikidata_cache"])
        probe_cache_dir.mkdir(parents=True, exist_ok=True)
        probe_path = probe_cache_dir / f"sparql_probe_{hashlib.sha1(probe_query.encode('utf-8')).hexdigest()}.json"
        try:
            if probe_path.exists():
                cached_probe = json.loads(probe_path.read_text(encoding="utf-8"))
                if cached_probe.get("query") != probe_query or "status_code" not in cached_probe:
                    raise ValueError(f"Invalid WDQS connectivity cache: {probe_path}")
                probe_status = cached_probe["status_code"]
                probe_latency = float(cached_probe["latency_seconds"])
                logger.info("WDQS trivial connectivity result loaded from cache")
            else:
                probe = requests.get(settings["endpoint"], params={"query": probe_query, "format": "json"},
                                     headers={"Accept": "application/sparql-results+json", "User-Agent": settings["user_agent"]},
                                     timeout=float(settings["connectivity_timeout_seconds"]))
                probe_status = probe.status_code
                probe_latency = time.perf_counter() - probe_started
                probe_path.write_text(json.dumps({"query": probe_query, "status_code": probe.status_code,
                                                  "latency_seconds": probe_latency,
                                                  "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                                  "response_text": probe.text}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        except requests.RequestException as exc:
            probe_status = f"ERROR: {type(exc).__name__}: {exc}"
            probe_latency = time.perf_counter() - probe_started
        except ValueError:
            raise
        logger.info("WDQS trivial connectivity query: status=%s latency_seconds=%.3f", probe_status, probe_latency)

        if settings["enabled"]:
            if backend == "api":
                if not 1 <= int(settings["api_search_limit"]) <= 10:
                    raise ValueError("wikidata.api_search_limit must be in [1, 10]")
                client = ActionAPIClient(settings, Path(attest_paths["wikidata_cache"]), logger)
                wikidata_pairs = client.fetch_pairs(lemmas, candidate_pairs, progress_every)
                logger.info("Action API exact VI-attested lemma-candidate pairs: %d", len(wikidata_pairs))
                if client.requests:
                    date = datetime.now(timezone.utc).date().isoformat()
                    sources = Path(paths["sources"])
                    note = f"Wikidata Action API query date: {date} UTC; cached responses in data/raw/wikidata_cache/ (per-request SHA-1 keys)."
                    current = sources.read_text(encoding="utf-8")
                    if note not in current:
                        sources.write_text(current.rstrip() + "\n\n" + note + "\n", encoding="utf-8", newline="\n")
            else:
                client = WikidataClient(settings, Path(attest_paths["wikidata_cache"]), logger)
                n_batches = (len(lemmas) + batch_size - 1) // batch_size
                for batch_number, offset in enumerate(range(0, len(lemmas), batch_size), start=1):
                    batch = lemmas[offset:offset + batch_size]
                    batch_set = set(batch)
                    pairs_batch = sorted(pair for pair in candidate_pairs if pair[0] in batch_set)
                    client.fetch_pairs(pairs_batch)
                    logger.info("WDQS batch %d/%d: en_lemmas=%d candidate_pairs=%d",
                                batch_number, n_batches, len(batch), len(pairs_batch))
                for cache_path, (query, _envelope) in sorted(client.cache_records.items()):
                    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))["response"]
                    wikidata_pairs.update(parse_wikidata_response(payload))
                update_sources_table(paths["sources"], [
                    _cache_record(Path(cache_path), query, envelope)
                    for cache_path, (query, envelope) in sorted(client.cache_records.items())
                ])
        else:
            logger.info("Wikidata stage disabled by wikidata.enabled=false; all in_wikidata flags are false")

        before_by_pos: Counter[str] = Counter()
        after_by_pos: Counter[str] = Counter()
        metric_counts: dict[str, Counter[str]] = {"in_muse": Counter(), "in_wikidata": Counter(), "in_vi_gloss": Counter(), "external_attested": Counter()}
        accepted_rows: list[dict[str, Any]] = []
        hypothetical_concepts: set[str] = set()
        rejected_gloss_pairs: list[dict[str, Any]] = []
        all_accepted_pairs: list[dict[str, Any]] = []
        rejected_pairs: list[dict[str, Any]] = []
        all_syllables: Counter[str] = Counter()
        accepted_syllables: Counter[str] = Counter()

        for row in pool_rows:
            en_lemma = normalize_nfc(row["en_lemma"])
            pos = normalize_nfc(row["pos"])
            enriched_candidates = []
            for candidate in row["vi_cands"]:
                vi_word = normalize_nfc(candidate["word"])
                in_muse = muse_contains(muse_pairs, vi_word, en_lemma)
                in_wikidata = (_fold(en_lemma), _fold(vi_word)) in wikidata_pairs
                positions = candidate.get("vi_pos")
                if not isinstance(positions, list) or any(not isinstance(value, str) for value in positions):
                    raise ValueError(f"Pool candidate has malformed vi_pos: {candidate!r}")
                glosses = [
                    gloss
                    for vi_pos in positions
                    for gloss in gloss_index.get((vi_word.casefold(), vi_pos.casefold()), [])
                ]
                in_vi_gloss = any(contains_whole_word(en_lemma, gloss) for gloss in glosses)
                external_attested = in_muse or in_wikidata
                flags = {"in_wiktextract": True, "in_muse": in_muse, "in_wikidata": in_wikidata,
                         "in_vi_gloss": in_vi_gloss, "external_attested": external_attested}
                n_sources = sum(int(flags[source_flags[name]]) for name in attestation_sources)
                enriched = {
                    **candidate,
                    **flags,
                    "n_sources": n_sources,
                }
                enriched_candidates.append(enriched)
                before_by_pos[pos] += 1
                bucket = syllable_bucket(vi_word, bin_edges)
                all_syllables[bucket] += 1
                pair = {"concept_id": row["concept_id"], "en_lemma": en_lemma, "pos": pos,
                        "vi_candidate": vi_word, **flags, "n_sources": n_sources}
                if n_sources >= minimum_sources:
                    after_by_pos[pos] += 1
                    metric_counts["in_muse"][pos] += int(in_muse)
                    metric_counts["in_wikidata"][pos] += int(in_wikidata)
                    metric_counts["in_vi_gloss"][pos] += int(in_vi_gloss)
                    metric_counts["external_attested"][pos] += int(external_attested)
                    accepted_syllables[bucket] += 1
                    all_accepted_pairs.append(pair)
                    enriched_candidates[-1] = enriched
                else:
                    rejected_pairs.append(pair)

            survivors = [candidate for candidate in enriched_candidates if candidate["n_sources"] >= minimum_sources]
            if survivors:
                output_row = {**row, "vi_cands": survivors}
                accepted_rows.append(output_row)

        schema = _source_schema(pool_table.schema)
        _write_parquet(accepted_rows, schema, Path(attest_paths["output"]))
        _update_dropflow(
            Path(paths["dropflow"]),
            n_before=len(pool_rows),
            n_after=len(accepted_rows),
            by_pos=dict(sorted(Counter(row["pos"] for row in accepted_rows).items())),
        )

        logger.info("By-POS attestation metrics (candidate pairs; shares among accepted pairs):")
        for pos in sorted(before_by_pos):
            before = before_by_pos[pos]
            after = after_by_pos[pos]
            shares = {name: metric_counts[name][pos] / after if after else 0.0 for name in metric_counts}
            logger.info("%s | n_before=%d | n_after=%d | in_vi_gloss=%.6f | in_muse=%.6f | in_wikidata=%.6f | external_attested=%.6f",
                        pos, before, after, shares["in_vi_gloss"], shares["in_muse"], shares["in_wikidata"], shares["external_attested"])
        logger.info("MUSE pool-pair coverage by syllable bucket and POS (unique en/vi/POS triples):")
        unique_muse_pairs: dict[tuple[str, str, str], bool] = {}
        for row in pool_rows:
            for candidate in row["vi_cands"]:
                vi = normalize_nfc(candidate["word"])
                en = normalize_nfc(row["en_lemma"])
                pos = normalize_nfc(row["pos"])
                unique_muse_pairs[(en, vi, pos)] = muse_contains(muse_pairs, vi, en)
        muse_diagnostic: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for (en, vi, pos), matched in unique_muse_pairs.items():
            muse_diagnostic[(pos, syllable_bucket(vi, bin_edges))].append(matched)
        for pos, bucket in sorted(muse_diagnostic):
            vals = muse_diagnostic[(pos, bucket)]
            logger.info("%s | syllables=%s | in_muse=%d/%d (%.6f)", pos, bucket, sum(vals), len(vals), sum(vals) / len(vals))
        logger.info("Vietnamese candidate syllable counts (candidate occurrences):")
        for bucket in [str(edge) for edge in bin_edges] + [f"{bin_edges[-1] + 1}+"]:
            logger.info("syllables=%s | all_pool=%d | accepted=%d", bucket, all_syllables[bucket], accepted_syllables[bucket])
        logger.info("Rejected pairs (%d available; first %d deterministic):", len(rejected_pairs), int(config["attest"]["report_sample_n"]))
        report_sample_n = int(config["attest"]["report_sample_n"])
        for pair in sorted(rejected_pairs, key=lambda item: (item["en_lemma"], item["pos"], item["vi_candidate"], item["concept_id"]))[:report_sample_n]:
            logger.info("%s", json.dumps(pair, ensure_ascii=False, sort_keys=True))
        rng = np.random.default_rng(config["seed"])
        sample_size = min(report_sample_n, len(all_accepted_pairs))
        if sample_size:
            chosen = rng.choice(len(all_accepted_pairs), size=sample_size, replace=False).tolist()
            selected = sorted((all_accepted_pairs[index] for index in chosen), key=lambda item: (item["en_lemma"], item["pos"], item["vi_candidate"], item["concept_id"]))
        else:
            selected = []
        logger.info("Seeded accepted-pair sample (%d):", len(selected))
        for pair in selected:
            logger.info("%s", json.dumps(pair, ensure_ascii=False, sort_keys=True))
        logger.info("Action API calls=%d; cache hits=%d; failures=%d; retries=%d",
                    client.requests if isinstance(client, ActionAPIClient) else 0,
                    client.cache_hits if isinstance(client, ActionAPIClient) else 0,
                    client.failures if isinstance(client, ActionAPIClient) else 0,
                    client.retries if isinstance(client, ActionAPIClient) else 0)
        logger.info("Output: %s (%d concepts)", attest_paths["output"], len(accepted_rows))
        elapsed = time.monotonic() - started
        logger.info("Runtime seconds: %.3f", elapsed)
        return {"rows": len(accepted_rows), "runtime_seconds": elapsed,
                "action_api_calls": client.requests if isinstance(client, ActionAPIClient) else 0,
                "cache_hits": client.cache_hits if client else 0, "failures": client.failures if client else 0,
                "wdqs_status": probe_status, "wdqs_latency_seconds": probe_latency}
    except Exception as exc:
        if client is not None:
            logger.error("Wikidata requests=%d; cache hits=%d; failures=%d; retries=%d",
                         client.requests, client.cache_hits, client.failures, client.retries)
        logger.error("Attestation stopped: %s", exc)
        logger.error("Runtime seconds: %.3f", time.monotonic() - started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
