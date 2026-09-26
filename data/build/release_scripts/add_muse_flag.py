"""Recompute the omitted MUSE attestation flag from a user's own dictionaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import unicodedata
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(path: Path) -> list[dict[str, str]]:
    """Read the provenance table without project-specific dependencies."""
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| source |"):
            continue
        fields = [part.strip().replace("\\|", "|") for part in line.strip().strip("|").split("|")]
        if len(fields) == 8:
            rows.append(dict(zip(("source", "file", "URL", "retrieved UTC", "bytes", "SHA-256", "version", "license"), fields)))
    return rows


def verify_source(path: Path, sources_path: Path) -> None:
    """Warn if the source file digest differs from its provenance record."""
    candidates = [row for row in source_rows(sources_path)
                  if row["source"] == "muse" and Path(row["file"]).name == path.name]
    if len(candidates) != 1:
        print(f"WARNING: no unique MUSE provenance row for {path.name}; continuing.", file=sys.stderr)
        return
    actual = sha256_file(path)
    if actual != candidates[0]["SHA-256"]:
        print(f"WARNING: SHA-256 mismatch for {path.name}; expected {candidates[0]['SHA-256']}, got {actual}.", file=sys.stderr)


def normalize_term(value: str) -> str:
    """Apply the pipeline's NFC, trim, lowercase MUSE normalization."""
    return unicodedata.normalize("NFC", value.strip()).lower()


def read_pairs(path: Path, orientation: str) -> set[tuple[str, str]]:
    """Read tab-separated pairs into normalized (Vietnamese, English) keys."""
    if orientation not in {"vi-en", "en-vi"}:
        raise ValueError(f"Unsupported MUSE orientation: {orientation}")
    result: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or not all(field.strip() for field in fields):
                raise ValueError(f"Malformed MUSE line {line_no} in {path}")
            first, second = (normalize_term(field) for field in fields)
            result.add((first, second) if orientation == "vi-en" else (second, first))
    return result


def read_concepts(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    """Read the release TSV while preserving its pipeline comment line."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        comment = handle.readline()
        if not comment.startswith("# Pipeline commit: "):
            raise ValueError(f"Missing pipeline commit comment in {path}")
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing TSV header in {path}")
        if "in_muse" in reader.fieldnames:
            raise ValueError("Input already contains in_muse; refusing to add it twice")
        required = {"vi", "en", "in_vi_gloss"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Input is missing required columns: {sorted(required - set(reader.fieldnames))}")
        return comment, list(reader.fieldnames), list(reader)


def write_with_flag(
    comment: str, columns: list[str], rows: list[dict[str, str]], pairs: set[tuple[str, str]], output: Path,
) -> None:
    """Insert in_muse after in_vi_gloss and write a deterministic TSV."""
    position = columns.index("in_vi_gloss") + 1
    output_columns = columns[:position] + ["in_muse"] + columns[position:]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(comment)
        writer = csv.DictWriter(handle, fieldnames=output_columns, delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            key = (normalize_term(row["vi"]), normalize_term(row["en"]))
            row["in_muse"] = "true" if key in pairs else "false"
            writer.writerow(row)


def main() -> int:
    """Run the standalone flag restoration command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--vi-en", type=Path, required=True)
    parser.add_argument("--en-vi", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_source(args.vi_en, args.sources)
    verify_source(args.en_vi, args.sources)
    pairs = read_pairs(args.vi_en, "vi-en") | read_pairs(args.en_vi, "en-vi")
    comment, columns, rows = read_concepts(args.concepts)
    write_with_flag(comment, columns, rows, pairs, args.output)
    print(f"Wrote in_muse for {len(rows)} concepts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
