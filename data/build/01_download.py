"""Download the versioned source resources used by the viLens data pipeline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import html.parser
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import urljoin
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from common import DataConfigError, load_config, normalize_nfc, setup_logging


TABLE_HEADER = ("source", "file", "URL", "retrieved UTC", "bytes", "SHA-256", "version/commit/extraction date", "license")
SOURCE_NAMES = (
    "wiktextract_en", "wiktextract_vi", "muse", "unihan", "cedict",
    "brysbaert", "flores", "nllb", "lid", "wordnet", "gated",
)


class SourceError(RuntimeError):
    """A source failed validation or could not be retrieved safely."""


class FrozenSourceError(SourceError):
    """An existing frozen file conflicts with its provenance record."""


@dataclass(frozen=True)
class SourceRecord:
    """Provenance for one local source file."""

    source: str
    file: str
    url: str
    retrieved_utc: str
    bytes: str
    sha256: str
    version: str
    license: str

    def as_row(self) -> tuple[str, ...]:
        return (
            self.source, self.file, self.url, self.retrieved_utc,
            self.bytes, self.sha256, self.version, self.license,
        )


def sha256_file(path: str | Path, chunk_size: int) -> str:
    """Hash a file in bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def skip_if_hash_matches(path: str | Path, record: SourceRecord | None, chunk_size: int) -> bool:
    """Skip only a present file whose digest agrees with its provenance row."""

    candidate = Path(path)
    if not candidate.exists():
        return False
    if not candidate.is_file():
        raise FrozenSourceError(f"Expected a file but found another path type: {candidate}")
    if record is None:
        raise FrozenSourceError(f"File already exists without a SOURCES.md hash; refusing overwrite: {candidate}")
    actual = sha256_file(candidate, chunk_size)
    if actual != record.sha256 or candidate.stat().st_size != int(record.bytes):
        raise FrozenSourceError(
            f"Frozen source hash mismatch for {candidate}: SOURCES.md={record.sha256}, actual={actual}; stopping."
        )
    return True


def render_sources(records: Iterable[SourceRecord]) -> str:
    """Render provenance records as a deterministic Markdown table."""

    ordered = sorted(records, key=lambda row: (row.source, row.file))
    lines = ["# Source provenance", "", "| " + " | ".join(TABLE_HEADER) + " |", "|" + "|".join("---" for _ in TABLE_HEADER) + "|"]
    for record in ordered:
        cells = [value.replace("|", "\\|").replace("\n", " ") for value in record.as_row()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_sources(text: str) -> list[SourceRecord]:
    """Read the provenance Markdown table, rejecting malformed columns/rows."""

    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("| ")), None)
    if header_index is None:
        if text.strip() in ("", "# Source provenance"):
            return []
        raise ValueError("SOURCES.md is missing its Markdown table")

    def cells(line: str) -> list[str]:
        if not line.startswith("|") or not line.endswith("|"):
            raise ValueError(f"Malformed SOURCES.md table row: {line}")
        parts = re.split(r"(?<!\\)\|", line[1:-1])
        return [normalize_nfc(part.strip().replace("\\|", "|")) for part in parts]

    header = cells(lines[header_index])
    if tuple(header) != TABLE_HEADER:
        raise ValueError(f"Unexpected SOURCES.md columns: {header!r}")
    result: list[SourceRecord] = []
    for line in lines[header_index + 2:]:
        if not line.strip():
            continue
        if not line.startswith("|"):
            break
        values = cells(line)
        if len(values) != len(TABLE_HEADER):
            raise ValueError(f"Malformed SOURCES.md row: {line}")
        result.append(SourceRecord(*values))
    return result


def write_sources(path: str | Path, records: Iterable[SourceRecord]) -> None:
    """Rewrite the provenance table while preserving any trailing source notes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_sources(records)
    if target.exists():
        old_lines = target.read_text(encoding="utf-8").splitlines()
        header_index = next((i for i, line in enumerate(old_lines) if line.startswith("| ")), None)
        if header_index is not None:
            cursor = header_index + 2
            while cursor < len(old_lines) and (not old_lines[cursor].strip() or old_lines[cursor].startswith("|")):
                cursor += 1
            notes = "\n".join(old_lines[cursor:]).strip("\n")
            if notes:
                text = text.rstrip("\n") + "\n\n" + notes + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=target.parent, prefix=".sources-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def utc_now() -> str:
    """Return an ISO-8601 UTC retrieval timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def available_bytes(path: str | Path) -> int:
    """Return free bytes on the filesystem containing path."""

    return shutil.disk_usage(Path(path)).free


def source_record(source: str, path: Path, repo_root: Path, url: str, retrieved: str, version: str, license_text: str, chunk_size: int) -> SourceRecord:
    """Build a provenance row from one verified local file."""

    return SourceRecord(source, path.resolve().relative_to(repo_root.resolve()).as_posix(), url, retrieved, str(path.stat().st_size), sha256_file(path, chunk_size), version, license_text)


def read_records(path: Path) -> dict[str, SourceRecord]:
    """Index the existing table by repository-relative file path."""

    if not path.exists():
        return {}
    records = parse_sources(path.read_text(encoding="utf-8"))
    indexed = {record.file: record for record in records}
    if len(indexed) != len(records):
        raise SourceError(f"SOURCES.md contains duplicate file paths: {path}")
    return indexed


def session_for(config: dict[str, Any]) -> requests.Session:
    """Create a polite, retrying HTTP session with a descriptive user-agent."""

    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({"User-Agent": config["downloads"]["user_agent"]})
    retry = Retry(total=config["downloads"]["retries"], backoff_factor=config["downloads"]["retry_backoff_seconds"], status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(("GET", "HEAD")))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(
    session: requests.Session, url: str, target: Path, source: str,
    version: str, license_text: str, config: dict[str, Any], root: Path,
    records: dict[str, SourceRecord], sources_path: Path, logger,
) -> SourceRecord:
    """Stream one HTTP resource to a fresh path, hash it, then record provenance."""

    rel = target.resolve().relative_to(root.resolve()).as_posix()
    existing = records.get(rel)
    if skip_if_hash_matches(target, existing, config["downloads"]["chunk_size_bytes"]):
        logger.info("Verified and skipped %s", rel)
        return existing  # type: ignore[return-value]
    if existing is not None:
        raise FrozenSourceError(f"SOURCES.md has a row for missing frozen file {rel}; refusing replacement.")

    target.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, stream=True, timeout=config["downloads"]["timeout_seconds"])
    response.raise_for_status()
    digest = hashlib.sha256()
    size = 0
    retrieved = utc_now()
    with response, tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=".download-", delete=False) as handle:
        temporary = Path(handle.name)
        total = int(response.headers["Content-Length"]) if response.headers.get("Content-Length", "").isdigit() else None
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=target.name, leave=False, file=sys.stdout)
        try:
            for chunk in response.iter_content(chunk_size=config["downloads"]["chunk_size_bytes"]):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    progress.update(len(chunk))
        finally:
            progress.close()
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if not size:
            raise SourceError(f"Server returned an empty file for {url}")
        if target.exists():
            raise FrozenSourceError(f"Refusing to overwrite existing untracked file: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    record = SourceRecord(source, rel, response.url, retrieved, str(size), digest.hexdigest(), version, license_text)
    records[rel] = record
    write_sources(sources_path, records.values())
    logger.info("Downloaded %s (%s bytes, sha256 %s)", rel, size, record.sha256)
    return record


class LinkParser(html.parser.HTMLParser):
    """Collect links from source pages without assuming generated filenames."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def links_from_page(session: requests.Session, url: str, timeout: int) -> list[tuple[str, str]]:
    """Fetch a source page and return its literal link targets and labels."""

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    parser = LinkParser()
    parser.feed(response.content.decode(response.encoding or "utf-8", errors="strict"))
    return [(urljoin(response.url, href), label) for href, label in parser.links if href]


def discover_wiktextract(session: requests.Session, page_url: str, expected_name: str, source_page: str, timeout: int) -> tuple[str, str, str]:
    """Follow a language page's current download link and read its extraction date."""

    page = session.get(page_url, timeout=timeout)
    page.raise_for_status()
    parser = LinkParser()
    parser.feed(page.content.decode(page.encoding or "utf-8", errors="strict"))
    page_links = [(urljoin(page.url, href), label) for href, label in parser.links if href]
    candidates = [(url, label) for url, label in page_links if expected_name in url and url.endswith(".jsonl")]
    if len(candidates) != 1:
        raise SourceError(f"Expected one current {expected_name} JSONL download on {page_url}; found {len(candidates)}")

    page_text = normalize_nfc(html.parser.unescape(page.content.decode(page.encoding or "utf-8", errors="strict")))
    match = re.search(r"\bextracted\s+on\s+(\d{4}-\d{2}-\d{2})", page_text, flags=re.IGNORECASE)
    if not match:
        raise SourceError(f"Could not find the extraction date on language page {page_url}")
    extraction_date = match.group(1)

    source_response = session.get(source_page, timeout=timeout)
    source_response.raise_for_status()
    source_html = normalize_nfc(html.parser.unescape(source_response.content.decode(source_response.encoding or "utf-8", errors="strict")))
    raw_name = "English-language edition of Wiktionary" if "English" in page_url else "Vietnamese"
    if raw_name not in source_html:
        raise SourceError(f"The current Wiktextract page no longer identifies {raw_name}")
    dump_match = re.search(r"current version was extracted from the .*? dated\s+(\d{4}-\d{2}-\d{2})", source_html, flags=re.IGNORECASE)
    if not dump_match:
        raise SourceError(f"Could not find the current Wiktionary dump date on {source_page}")
    return candidates[0][0], f"extraction {extraction_date}; enwiktionary dump {dump_match.group(1)}", page.url


def license_from_text(text: str, patterns: tuple[str, ...], description: str) -> str:
    """Return an explicitly stated license string or fail instead of guessing."""

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_nfc(match.group(0).strip())
    raise SourceError(f"Could not verify the stated license for {description}")


def run_wiktextract(source: str, config, session, root, records, sources_path, logger) -> None:
    frozen = [record for record in records.values() if record.source == source]
    if frozen:
        if len(frozen) != 1:
            raise SourceError(f"Expected one frozen Wiktextract file for {source}; found {len(frozen)}")
        target = root.parent.parent / frozen[0].file
        if not skip_if_hash_matches(target, frozen[0], config["downloads"]["chunk_size_bytes"]):
            raise FrozenSourceError(f"SOURCES.md has a row for missing frozen file {frozen[0].file}")
        logger.info("Verified and skipped %s without re-querying kaikki.org", frozen[0].file)
        return
    entry = config["downloads"]["wiktextract"]
    page_key, expected, name = ("english_page", "kaikki.org-dictionary-English", "English") if source == "wiktextract_en" else ("vietnamese_page", "kaikki.org-dictionary-Vietnamese", "Vietnamese")
    url, version, page_url = discover_wiktextract(session, entry[page_key], expected, entry["source_page"], config["downloads"]["timeout_seconds"])
    wiktionary_url = "https://en.wiktionary.org/wiki/Wiktionary:Text_of_Creative_Commons_Attribution-ShareAlike_4.0_International_License"
    license_text = "CC BY-SA 4.0 (Wiktionary content; see Wiktionary license page)"
    filename = Path(url).name
    download_file(session, url, root / source / filename, source, f"{version}; source page {page_url}; license details: {wiktionary_url}", license_text, config, root.parent.parent, records, sources_path, logger)


def run_muse(config, session, root, records, sources_path, logger) -> None:
    settings = config["downloads"]["muse"]
    page = session.get(settings["page"], timeout=config["downloads"]["timeout_seconds"])
    page.raise_for_status()
    for filename in settings["files"]:
        url = urljoin(settings["base_url"], filename)
        license_text = "Attribution-NonCommercial 4.0 International (https://github.com/facebookresearch/MUSE/blob/main/LICENSE)"
        record = download_file(session, url, root / "muse" / filename, "muse", "MUSE ground-truth bilingual dictionary; version not stated at download endpoint", license_text, config, root.parent.parent, records, sources_path, logger)
        if record.license != license_text:
            records[record.file] = SourceRecord(record.source, record.file, record.url, record.retrieved_utc, record.bytes, record.sha256, record.version, license_text)
            write_sources(sources_path, records.values())


def run_unihan(config, session, root, records, sources_path, logger) -> None:
    settings = config["downloads"]["unihan"]
    target = root / "unihan" / "Unihan.zip"
    relative_zip = target.resolve().relative_to(root.parent.parent.resolve()).as_posix()
    old_zip = records.get(relative_zip)
    if target.exists() and old_zip is not None and skip_if_hash_matches(target, old_zip, config["downloads"]["chunk_size_bytes"]):
        row = old_zip
        version = row.version
        logger.info("Verified and skipped Unihan.zip without re-querying unicode.org")
    else:
        response = session.get(settings["version_url"], timeout=config["downloads"]["timeout_seconds"])
        response.raise_for_status()
        readme = normalize_nfc(response.content.decode(response.encoding or "utf-8", errors="strict"))
        version_match = re.search(r"\bversion\s+([0-9]+\.[0-9]+\.[0-9]+)", readme, flags=re.IGNORECASE)
        if not version_match:
            raise SourceError("Unicode latest ReadMe.txt did not identify a Unicode version")
        version = f"Unicode {version_match.group(1)}"
        row = download_file(session, settings["url"], target, "unihan", version, "Unicode® License and Terms of Use", config, root.parent.parent, records, sources_path, logger)

    with zipfile.ZipFile(target) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        output_root = root / "unihan" / "extracted"
        planned: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in members:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise SourceError(f"Unsafe path in Unihan archive: {info.filename}")
            output = output_root.joinpath(*member.parts)
            rel = output.resolve().relative_to(root.parent.parent.resolve()).as_posix()
            if output.exists():
                if skip_if_hash_matches(output, records.get(rel), config["downloads"]["chunk_size_bytes"]):
                    continue
            if rel in records:
                raise FrozenSourceError(f"SOURCES.md has a row for missing extracted file {rel}; stopping.")
            planned.append((info, output))
        for info, output in planned:
            with archive.open(info) as source_handle:
                output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("wb", dir=output.parent, prefix=".unihan-", delete=False) as handle:
                    temp = Path(handle.name)
                    digest = hashlib.sha256()
                    size = 0
                    for chunk in iter(lambda: source_handle.read(config["downloads"]["chunk_size_bytes"]), b""):
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            try:
                if output.exists():
                    raise FrozenSourceError(f"Refusing to overwrite extracted Unicode file {output}")
                os.replace(temp, output)
            finally:
                temp.unlink(missing_ok=True)
            rel = output.resolve().relative_to(root.parent.parent.resolve()).as_posix()
            records[rel] = SourceRecord("unihan", rel, settings["url"] + "#" + info.filename, row.retrieved_utc, str(size), digest.hexdigest(), version, "Unicode® License and Terms of Use")
            write_sources(sources_path, records.values())


def run_cedict(config, session, root, records, sources_path, logger) -> None:
    settings = config["downloads"]["cedict"]
    target = root / "cedict" / Path(settings["url"]).name
    rel = target.resolve().relative_to(root.parent.parent.resolve()).as_posix()
    record = download_file(session, settings["url"], target, "cedict", "CC-CEDICT release date from the file header", "Creative Commons Attribution-ShareAlike 4.0 International License", config, root.parent.parent, records, sources_path, logger)
    release_date = None
    with gzip.open(target, "rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith("#! date="):
                release_date = line.removeprefix("#! date=").strip()
                break
            if not line.startswith("#"):
                break
    if not release_date:
        raise SourceError(f"CC-CEDICT header has no #! date= release line: {target}")
    version = f"CC-CEDICT release date {release_date} (header #! date={release_date})"
    if record.version != version:
        records[rel] = SourceRecord(record.source, record.file, record.url, record.retrieved_utc, record.bytes, record.sha256, version, record.license)
        write_sources(sources_path, records.values())


def run_brysbaert(config, session, root, records, sources_path, logger) -> None:
    settings = config["downloads"]["brysbaert"]
    directory = root / "brysbaert"
    directory.mkdir(parents=True, exist_ok=True)
    local_files = sorted(path for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))
    if local_files:
        for path in local_files:
            rel = path.resolve().relative_to(root.parent.parent.resolve()).as_posix()
            old = records.get(rel)
            if skip_if_hash_matches(path, old, config["downloads"]["chunk_size_bytes"]):
                continue
            if old is not None:
                raise FrozenSourceError(f"Manually supplied Brysbaert file conflicts with SOURCES.md: {rel}")
            record = source_record("brysbaert", path, root.parent.parent, "manual download; see " + settings["manual_url"], "unknown (file supplied before retrieval was recorded)", "Brysbaert, Warriner & Kuperman (2014)", "License/terms not stated in local file; check publisher terms before redistribution", config["downloads"]["chunk_size_bytes"])
            records[rel] = record
        write_sources(sources_path, records.values())
        logger.info("Hashed %s manually supplied Brysbaert file(s)", len(local_files))
        return

    response = session.head(settings["url"], allow_redirects=True, timeout=config["downloads"]["timeout_seconds"])
    if response.ok and "csv" in response.headers.get("Content-Type", "").lower():
        download_file(session, response.url, directory / "Concreteness_ratings_Brysbaert_et_al_BRM.csv", "brysbaert", "Brysbaert, Warriner & Kuperman (2014)", "No license stated at file URL; check publisher terms", config, root.parent.parent, records, sources_path, logger)
        return
    logger.warning("MANUAL ACTION: download the Brysbaert et al. (2014) concreteness ratings from %s and place the original file in %s; direct CSV URL returned HTTP %s (%s). No scraping was attempted.", settings["manual_url"], directory, response.status_code, response.headers.get("Content-Type", "no content type"))
    for path in sorted(item for item in directory.iterdir() if item.is_file() and not item.name.startswith(".")):
        rel = path.resolve().relative_to(root.parent.parent.resolve()).as_posix()
        old = records.get(rel)
        if skip_if_hash_matches(path, old, config["downloads"]["chunk_size_bytes"]):
            continue
        if old is not None:
            raise FrozenSourceError(f"Brysbaert file conflicts with SOURCES.md: {rel}")
        records[rel] = source_record("brysbaert", path, root.parent.parent, "manual download; see " + settings["manual_url"], "unknown", "Brysbaert, Warriner & Kuperman (2014)", "License/terms must be verified before redistribution", config["downloads"]["chunk_size_bytes"])
    write_sources(sources_path, records.values())


def hub_token() -> str | None:
    """Read the caller-provided Hugging Face token without printing it."""

    token = os.environ.get("HF_TOKEN")
    return token.strip() if token and token.strip() else None


def pin_config_revision(config_path: Path, key_path: tuple[str, ...], revision: str) -> None:
    """Replace one explicit TODO revision while preserving other YAML comments."""

    content = config_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    stack: list[tuple[int, str]] = []
    found = False
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        match = re.match(r"\s*([\w-]+):\s*(.*?)\s*(#.*)?(?:\r?\n)?$", line)
        if not match:
            continue
        key, value, comment = match.groups()
        path = tuple(item[1] for item in stack) + (key,)
        if path == key_path:
            if value.strip(" '") not in ("TODO", ""):
                if value.strip(" '") == revision:
                    return
                raise DataConfigError(f"Refusing to change already pinned {'/'.join(key_path)} revision {value!r}")
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"{' ' * indent}{key}: {revision}{('  ' + comment) if comment else ''}{newline}"
            found = True
            break
        stack.append((indent, key))
    if not found:
        raise DataConfigError(f"Could not find config key {'/'.join(key_path)}")
    config_path.write_text("".join(lines), encoding="utf-8", newline="")


def hub_client():
    """Construct the Hugging Face metadata client lazily."""

    from huggingface_hub import HfApi

    return HfApi()


def repo_revision(api, repo_id: str, repo_type: str, token: str | None) -> str:
    """Resolve the current repository commit through metadata only."""

    info = api.dataset_info(repo_id, token=token) if repo_type == "dataset" else api.model_info(repo_id, token=token)
    sha = info.sha
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SourceError(f"Hugging Face did not return a full commit hash for {repo_id}")
    return sha


def run_flores(config, session, root, records, sources_path, logger, config_path: Path) -> None:
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError
    from huggingface_hub.errors import HfHubHTTPError

    token = hub_token()
    if token is None:
        logger.warning("FLORES+ needs an HF_TOKEN. Accept the dataset terms on %s, then set HF_TOKEN; continuing with other sources.", config["downloads"]["flores"]["page"])
        return
    settings = config["downloads"]["flores"]
    try:
        revision = settings["revision"]
        if revision == "TODO":
            revision = repo_revision(hub_client(), settings["repo"], "dataset", token)
            pin_config_revision(config_path, ("downloads", "flores", "revision"), revision)
        elif not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise DataConfigError(f"downloads.flores.revision must be a full commit hash or TODO; got {revision!r}")
        for language in settings["languages"]:
            for split in settings["splits"]:
                output = root / "flores" / f"{split}_{language}.parquet"
                relative = output.resolve().relative_to(root.parent.parent.resolve()).as_posix()
                old = records.get(relative)
                if skip_if_hash_matches(output, old, config["downloads"]["chunk_size_bytes"]):
                    continue
                if old is not None:
                    raise FrozenSourceError(f"SOURCES.md has a row for missing frozen file {relative}; refusing replacement.")
                dataset = load_dataset(settings["repo"], language, split=split, revision=revision, token=token)
                table = dataset.data.table
                if "sentence" not in table.column_names:
                    raise SourceError(f"Unexpected FLORES+ fields for {split}/{language}: {table.column_names}")
                for column_index, field in enumerate(table.schema):
                    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                        normalized = pa.array((normalize_nfc(value) if value is not None else None for value in table.column(field.name).to_pylist()), type=field.type)
                        table = table.set_column(column_index, field, normalized)
                output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("wb", dir=output.parent, prefix=".flores-", delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    pq.write_table(table, temporary, compression="zstd", version="2.6")
                    if output.exists():
                        raise FrozenSourceError(f"Refusing to overwrite existing file {output}")
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
                record = source_record("flores", output, root.parent.parent, f"https://huggingface.co/datasets/{settings['repo']}/tree/{revision}", utc_now(), revision, "CC BY-SA 4.0", config["downloads"]["chunk_size_bytes"])
                records[relative] = record
                write_sources(sources_path, records.values())
    except DatasetNotFoundError:
        logger.warning("FLORES+ gated access denied. Accept/request access on %s, then retry with an HF_TOKEN authorized for this dataset; continuing with other sources.", settings["page"])
        print(f"FLORES+ access denied. On {settings['page']}, accept/request access to the gated dataset, then retry with an authorized HF_TOKEN.")
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status not in (401, 403):
            raise
        logger.warning("FLORES+ gated access denied (HTTP %s). Log into Hugging Face, accept the FLORES+ terms at %s, and use an HF_TOKEN with access; continuing with other sources.", status, settings["page"])


def choose_device(torch_module) -> str:
    """Select CUDA, then Apple MPS, then CPU as required by AGENTS.md."""

    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_nllb(config, session, root, records, sources_path, logger, config_path: Path) -> None:
    import torch
    from huggingface_hub.errors import HfHubHTTPError
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    settings = config["nllb"]
    token = hub_token()
    revision = settings["revision"]
    if revision == "TODO":
        revision = repo_revision(hub_client(), settings["model"], "model", token)
        pin_config_revision(config_path, ("nllb", "revision"), revision)
    device = choose_device(torch)
    logger.info("Loading NLLB commit %s on %s", revision, device)
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=revision, src_lang=settings["test_source_lang"], token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings["model"], revision=revision, token=token).to(device)
    encoded = tokenizer(settings["test_input"], return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(settings["test_target_lang"]), num_beams=settings["num_beams"], num_return_sequences=settings["num_return"])
    translations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    if not translations:
        raise SourceError("NLLB returned no translations for the required device check")
    print(f"NLLB device: {device}")
    print(f'NLLB translation ({settings["test_input"]}): {translations[0]}')
    logger.info("NLLB top translation on %s: %s", device, translations[0])


def run_lid(config, session, root, records, sources_path, logger, config_path: Path) -> None:
    import fasttext
    from huggingface_hub import hf_hub_download

    settings = config["downloads"]["lid"]
    token = hub_token()
    revision = settings["glotlid_revision"]
    if revision == "TODO":
        revision = repo_revision(hub_client(), settings["glotlid_repo"], "model", token)
        pin_config_revision(config_path, ("downloads", "lid", "glotlid_revision"), revision)

    glot_record = next((row for row in records.values() if row.source == "lid" and row.file.endswith("/glotlid/model.bin")), None)
    glot_path = root / "lid" / "glotlid" / "model.bin"
    if skip_if_hash_matches(glot_path, glot_record, config["downloads"]["chunk_size_bytes"]):
        glot_model_path = str(glot_path)
    else:
        if glot_record is not None:
            raise FrozenSourceError(f"SOURCES.md has a row for missing frozen file {glot_record.file}")
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(settings["glotlid_repo"], "model.bin", revision=revision, token=token)
        glot_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=glot_path.parent, prefix=".glotlid-", delete=False) as handle:
            temp = Path(handle.name)
            with Path(cached).open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=config["downloads"]["chunk_size_bytes"])
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if glot_path.exists():
                raise FrozenSourceError(f"Refusing to overwrite existing GlotLID model: {glot_path}")
            os.replace(temp, glot_path)
        finally:
            temp.unlink(missing_ok=True)
        glot_model_path = str(glot_path)
        row = source_record("lid", glot_path, root.parent.parent, f"https://huggingface.co/{settings['glotlid_repo']}/resolve/{revision}/model.bin", utc_now(), revision, "Apache License, Version 2.0 plus notices", config["downloads"]["chunk_size_bytes"])
        records[row.file] = row
        write_sources(sources_path, records.values())
    ft_path = root / "lid" / "lid.176.bin"
    ft_row = next((row for row in records.values() if row.source == "lid" and row.file.endswith("/lid.176.bin")), None)
    if not skip_if_hash_matches(ft_path, ft_row, config["downloads"]["chunk_size_bytes"]):
        if ft_row is not None:
            raise FrozenSourceError(f"SOURCES.md has a row for missing frozen file {ft_row.file}")
        download_file(session, settings["fasttext_url"], ft_path, "lid", "fastText lid.176 language identification model", "Creative Commons Attribution-Share-Alike License 3.0", config, root.parent.parent, records, sources_path, logger)

    glot = fasttext.load_model(glot_model_path)
    lid176 = fasttext.load_model(str(ft_path))
    sample = settings["test_text"]
    glot_label = glot.predict(sample, k=settings["top_k"])[0][0]
    lid_label = lid176.predict(sample, k=settings["top_k"])[0][0]
    print(f"GlotLID top-1: {glot_label}")
    print(f"fastText lid.176 top-1: {lid_label}")
    logger.info("GlotLID top-1: %s; fastText lid.176 top-1: %s", glot_label, lid_label)


def wordnet_version_string(nltk_version: str, wordnet_version: str) -> str:
    """Format the package and corpus versions recorded for the WordNet source."""

    if not nltk_version.strip() or not wordnet_version.strip():
        raise ValueError("NLTK and WordNet versions must be non-empty")
    return f"NLTK {nltk_version}; WordNet {wordnet_version}"


def run_wordnet(config, root, records, sources_path, logger) -> None:
    """Download and provenance-record NLTK's WordNet archive under data/raw."""

    import nltk
    from nltk.corpus import wordnet
    from nltk.downloader import Downloader

    settings = config["downloads"]["wordnet"]
    corpus_id = settings["corpus"]
    if corpus_id != "wordnet":
        raise DataConfigError(f"downloads.wordnet.corpus must be 'wordnet'; got {corpus_id!r}")
    data_dir = root / settings["directory"]
    archive = data_dir / "corpora" / "wordnet.zip"
    repo_root = root.parent.parent
    relative = archive.resolve().relative_to(repo_root.resolve()).as_posix()
    existing = records.get(relative)
    package_url: str
    if skip_if_hash_matches(archive, existing, config["downloads"]["chunk_size_bytes"]):
        logger.info("Verified and skipped frozen WordNet archive %s", relative)
        retrieved = existing.retrieved_utc  # type: ignore[union-attr]
        package_url = existing.url  # type: ignore[union-attr]
    else:
        if existing is not None:
            raise FrozenSourceError(f"SOURCES.md has a row for missing frozen WordNet archive {relative}")
        if archive.exists():
            raise FrozenSourceError(f"WordNet archive exists without a SOURCES.md hash; refusing overwrite: {archive}")
        data_dir.mkdir(parents=True, exist_ok=True)
        package = Downloader().info(corpus_id)
        if not package.url:
            raise SourceError("NLTK package metadata did not provide a WordNet download URL")
        package_url = package.url
        retrieved = utc_now()
        if not nltk.download(corpus_id, download_dir=str(data_dir), quiet=True, raise_on_error=True):
            raise SourceError("NLTK downloader returned failure for the WordNet corpus")
        if not archive.is_file():
            raise SourceError(f"NLTK reported WordNet installed but archive is missing: {archive}")

    nltk.data.path = [str(data_dir.resolve())]
    try:
        wordnet.ensure_loaded()
        wordnet_version = wordnet.get_version()
        # Confirm the installed corpus can answer a normal lexical query.
        wordnet.synsets("entity")
    except LookupError as exc:
        raise SourceError(f"Downloaded WordNet cannot be loaded from {data_dir}: {exc}") from exc
    version = wordnet_version_string(nltk.__version__, wordnet_version)
    if existing is not None:
        if existing.version != version:
            raise FrozenSourceError(
                f"Frozen WordNet version mismatch for {relative}: SOURCES.md={existing.version!r}, installed={version!r}"
            )
        logger.info("WordNet already recorded: %s", version)
        return

    record = source_record(
        "wordnet", archive, repo_root, package_url,
        retrieved, version, "WordNet License", config["downloads"]["chunk_size_bytes"],
    )
    records[record.file] = record
    write_sources(sources_path, records.values())
    logger.info("Downloaded and verified WordNet archive %s (%s)", record.file, version)


def run_gated(config, logger) -> None:
    """Check Gemma and Llama access using repository metadata only."""

    from huggingface_hub.errors import HfHubHTTPError

    token = hub_token()
    api = hub_client()
    for name in ("gemma", "llama"):
        repo = config["models"][name]["hf_id"]
        try:
            info = api.model_info(repo, token=token, files_metadata=False)
            print(f"Gated access {name}: ACCESS (metadata only; commit {info.sha})")
            logger.info("Gated access %s: ACCESS (metadata only)", name)
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403, 404):
                print(f"Gated access {name}: NO ACCESS (HTTP {status})")
                logger.info("Gated access %s: NO ACCESS (HTTP %s)", name, status)
            else:
                raise


def run_step(args: argparse.Namespace) -> int:
    root = Path(args.config).resolve().parents[1]
    config = load_config(args.config)
    paths = config["paths"]
    raw = Path(paths["raw"])
    if not raw.is_absolute():
        raw = root / raw
    log_dir = Path(paths["logs"])
    if not log_dir.is_absolute():
        log_dir = root / log_dir
    logger = setup_logging(Path(__file__).stem, log_dir, level=config["logging"]["level"])
    free = available_bytes(raw)
    threshold = config["downloads"]["min_free_bytes"]
    print(f"Free disk space before download: {free} bytes ({free / 1_000_000_000:.2f} GB)")
    if free < threshold:
        logger.error("STOP: free disk space below configured preflight minimum.")
        print(f"STOP: only {free} bytes free on the data/raw volume; at least {threshold} bytes required. Nothing downloaded.")
        return 2

    source_path = Path(paths["sources"])
    if not source_path.is_absolute():
        source_path = root / source_path
    raw.mkdir(parents=True, exist_ok=True)
    records = read_records(source_path)
    source_root = raw
    session = session_for(config)
    config_path = Path(args.config).resolve()
    actions = {
        "wiktextract_en": lambda: run_wiktextract("wiktextract_en", config, session, source_root, records, source_path, logger),
        "wiktextract_vi": lambda: run_wiktextract("wiktextract_vi", config, session, source_root, records, source_path, logger),
        "muse": lambda: run_muse(config, session, source_root, records, source_path, logger),
        "unihan": lambda: run_unihan(config, session, source_root, records, source_path, logger),
        "cedict": lambda: run_cedict(config, session, source_root, records, source_path, logger),
        "brysbaert": lambda: run_brysbaert(config, session, source_root, records, source_path, logger),
        "flores": lambda: run_flores(config, session, source_root, records, source_path, logger, config_path),
        "nllb": lambda: run_nllb(config, session, source_root, records, source_path, logger, config_path),
        "lid": lambda: run_lid(config, session, source_root, records, source_path, logger, config_path),
        "wordnet": lambda: run_wordnet(config, source_root, records, source_path, logger),
        "gated": lambda: run_gated(config, logger),
    }
    selected = [args.only] if args.only else list(SOURCE_NAMES)
    failures: list[str] = []
    for source in selected:
        try:
            logger.info("Starting source %s", source)
            actions[source]()
        except FrozenSourceError as exc:
            logger.error("STOP: %s", exc)
            raise
        except Exception as exc:
            message = f"{source}: {type(exc).__name__}: {exc}"
            failures.append(message)
            logger.error("Source failed; continuing with other sources: %s", message)
        finally:
            free = available_bytes(raw)
            print(f"Free disk space after {source}: {free} bytes ({free / 1_000_000_000:.2f} GB)")
    for item in failures:
        print(f"SOURCE FAILED: {item}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", choices=SOURCE_NAMES, help="run one named source")
    return run_step(parser.parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FrozenSourceError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
