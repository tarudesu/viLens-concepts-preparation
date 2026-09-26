"""Rebuild the FLORES+ dev directions file from locally downloaded Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


LANGUAGE_CODES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "id": "ind_Latn",
    "vi": "vie_Latn",
    "zh": "cmn_Hans",
}
EXPECTED_SENTENCES = 997
VALIDATE_IDS = frozenset("""
104 113 114 122 123 128 130 134 135 136 137 140 141 145 156 168 176 184 185 186 187 188 192 197 20 213 219 22 225 227 231 237 24 241 242 244 247 251 267 27 283 284 298 30 306 309 321 322 325 339 347 362 370 374 379 390 391 394 405 407 409 41 414 415 416 423 424 429 438 439 442 452 456 457 46 462 466 472 475 498 500 51 510 512 515 516 526 533 54 542 547 550 556 557 561 562 566 567 572 590 591 60 602 617 618 62 623 636 646 649 653 654 658 668 671 673 677 679 68 680 682 684 687 689 698 708 709 718 720 723 729 730 747 750 762 768 770 771 776 778 780 786 787 789 794 797 802 808 81 816 818 839 841 852 857 858 862 865 871 872 875 876 883 886 887 889 892 894 896 898 899 9 90 906 908 909 910 914 915 921 928 929 933 934 936 944 949 950 954 957 958 96 962 968 970 981 989 99 990
""".split())


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
    """Warn if a FLORES Parquet file differs from its recorded digest."""
    candidates = [row for row in source_rows(sources_path)
                  if row["source"] == "flores" and Path(row["file"]).name == path.name]
    if len(candidates) != 1:
        print(f"WARNING: no unique FLORES provenance row for {path.name}; continuing.", file=sys.stderr)
        return
    actual = sha256_file(path)
    if actual != candidates[0]["SHA-256"]:
        print(f"WARNING: SHA-256 mismatch for {path.name}; expected {candidates[0]['SHA-256']}, got {actual}.", file=sys.stderr)


def sentence_id(value: Any) -> str:
    """Normalize one non-empty sentence identifier to NFC text."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"Invalid FLORES sentence id: {value!r}")
    result = unicodedata.normalize("NFC", str(value)).strip()
    if not result:
        raise ValueError("Empty FLORES sentence id")
    return result


def read_language(path: Path, sources_path: Path) -> dict[str, str]:
    """Read an id/sentence table, checking schema, duplicates, and source hash."""
    verify_source(path, sources_path)
    frame = pd.read_parquet(path, columns=["id", "sentence"])
    if list(frame.columns) != ["id", "sentence"]:
        raise ValueError(f"Unexpected FLORES columns in {path}: {list(frame.columns)}")
    result: dict[str, str] = {}
    for row_number, (raw_id, raw_sentence) in enumerate(frame.itertuples(index=False, name=None), start=1):
        key = sentence_id(raw_id)
        if key in result:
            raise ValueError(f"Duplicate FLORES sentence id {key!r} in {path}")
        if not isinstance(raw_sentence, str) or not raw_sentence.strip():
            raise ValueError(f"Empty FLORES sentence at row {row_number} in {path}")
        result[key] = unicodedata.normalize("NFC", raw_sentence)
    return result


def build_records(flores_dir: Path, sources_path: Path) -> list[dict[str, str]]:
    """Join the five languages and restore the frozen seeded fit/validate split."""
    datasets: dict[str, dict[str, str]] = {}
    for language, code in LANGUAGE_CODES.items():
        path = flores_dir / f"dev_{code}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FLORES file: {path}")
        datasets[language] = read_language(path, sources_path)
    reference = datasets["en"]
    reference_ids = set(reference)
    if len(reference_ids) != EXPECTED_SENTENCES:
        raise ValueError(f"Expected {EXPECTED_SENTENCES} FLORES dev IDs, found {len(reference_ids)}")
    if not VALIDATE_IDS.issubset(reference_ids):
        raise ValueError("FLORES IDs do not match the frozen validate split manifest")
    for language, dataset in datasets.items():
        if set(dataset) != reference_ids:
            raise ValueError(f"FLORES sentence IDs are not aligned for language {language}")
    records: list[dict[str, str]] = []
    for language in sorted(datasets):
        for key in sorted(reference_ids):
            records.append({
                "lang": language,
                "sentence_id": key,
                "split": "validate" if key in VALIDATE_IDS else "fit",
                "sentence": datasets[language][key],
            })
    return records


def main() -> int:
    """Run the standalone FLORES directions rebuild command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flores-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(args.flores_dir, args.sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Wrote {len(records)} FLORES direction records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
