"""Build the private-ready, provenance-preserving viLens public-release package."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

try:
    from common import load_config, setup_logging
except ModuleNotFoundError:
    from data.build.common import load_config, setup_logging


STEP = "15_release_package"
LANGUAGES = ("vi", "en", "zh", "fr", "id", "ru")
READOUT_LANGUAGES = ("vi", "en", "zh", "fr", "id")
MODELS = ("gemma", "qwen", "llama")

FIELD_DESCRIPTIONS = {
    "concept_id": "Stable 12-character identifier for this concept.",
    "split": "Leakage-safe partition: test, directions, or fewshot_reservoir.",
    "vi": "Canonical Vietnamese form.",
    "en": "Canonical English lemma used as the alignment pivot.",
    "zh": "Canonical simplified Mandarin form.",
    "fr": "Canonical French form.",
    "id": "Canonical Indonesian form.",
    "ru": "Canonical Russian prompt-source form; may be empty if unavailable.",
    "vi_alts": "Other surviving Vietnamese forms for the concept, stored as a JSON list.",
    "vi_nodiac": "Vietnamese canonical form with diacritics removed (đ→d).",
    "collapsed": "Whether the stripped Vietnamese form collides with another real form.",
    "pos": "Part of speech: noun, verb, or adjective.",
    "en_sense_gloss": "English gloss associated with the aligned sense.",
    "stratum": "Frozen etymological stratum: sino, nonsino, ambiguous, or other_loan.",
    "sigA": "Signal A: Wiktionary/Wiktextract etymological classification.",
    "sigB": "Primary Signal B: external Han-spelling verification classification.",
    "sigB_strict": "Signal B requiring the strict reading, dictionary, and meaning checks.",
    "sigB_relaxed": "Signal B requiring a verified Vietnamese reading and CEDICT headword.",
    "sino_via": "Recorded route/type supporting a Sino classification, when available.",
    "native_strict": "Whether the concept meets the strict native-origin evidence rule.",
    "reading_source": "Reading source used for a verified Han string, when applicable.",
    "n_sources": "Number of configured independent attestation sources for the selected Vietnamese form.",
    "in_vi_gloss": "Whether the Vietnamese entry contains the English lemma in an English gloss.",
    "in_wikidata": "Whether the Vietnamese–English pair is attested by Wikidata labels or aliases.",
    "external_attested": "Whether the pair is attested by MUSE or Wikidata.",
    "bt_route": "Back-translation pass route: strict, rt_contain, fwd_only, or none.",
    "bt_pass_strict_v1": "Whether the original strict exact-match round-trip rule passed.",
    "zh_nllb_agree": "Whether the provisional Mandarin form agrees with NLLB beam output.",
    "fr_nllb_agree": "Whether the provisional French form agrees with NLLB beam output.",
    "id_nllb_agree": "Whether the provisional Indonesian form agrees with NLLB beam output.",
    "ru_nllb_agree": "Whether the selected Russian form agrees with NLLB beam output.",
    "min_surface_dist": "Minimum normalized Vietnamese-to-en/fr/id surface edit distance.",
    "n_senses_vi": "Vietnamese sense count summed over homograph entries for the canonical form.",
    "n_syllables": "Whitespace-separated syllable count of the Vietnamese canonical form.",
    "concreteness_match": "Brysbaert matching route (exact/head/none); concreteness values are omitted.",
    "single_token_vi_gemma": "Whether the Vietnamese form is one token under the Gemma tokenizer.",
    "single_token_vi_qwen": "Whether the Vietnamese form is one token under the Qwen tokenizer.",
    "single_token_vi_llama": "Whether the Vietnamese form is one token under the Llama tokenizer.",
    "m1_eligible_gemma": "Gemma M1 eligibility flag for main-test or M1-extension concepts.",
    "m1_eligible_qwen": "Qwen M1 eligibility flag for main-test or M1-extension concepts.",
    "m1_eligible_llama": "Llama M1 eligibility flag for main-test or M1-extension concepts.",
    "m1_extension": "True only for concepts in the separate M1-only extension set.",
    "cloze_available": "Whether a qualifying Vietnamese cloze example is available.",
    "fewshot_set": "Fixed few-shot set assignment (1–6), or empty when not selected.",
}


def _path(root: Path, value: str) -> Path:
    """Resolve one configured path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_tsv(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    """Read concepts.tsv, preserving its comment and stable source row order."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        comment = handle.readline()
        if not comment.startswith("# Pipeline commit: "):
            raise ValueError(f"Missing pipeline commit header in {path}")
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing TSV header in {path}")
        rows = list(reader)
    if not rows or "concept_id" not in reader.fieldnames:
        raise ValueError(f"No concept rows or no concept_id in {path}")
    if len({row["concept_id"] for row in rows}) != len(rows):
        raise ValueError("Input concepts.tsv has duplicate concept_id values")
    return comment, list(reader.fieldnames), rows


def _cell(value: Any) -> str:
    """Serialize existing TSV cell strings without changing their contents."""
    return "" if value is None else str(value)


def _write_concepts(
    input_path: Path, output_path: Path, drop_columns: list[str], include_concreteness: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    """Remove only configured sensitive/restricted columns and write a stable TSV."""
    comment, columns, rows = _read_tsv(input_path)
    to_drop = list(dict.fromkeys(drop_columns))
    if not include_concreteness:
        to_drop.append("concreteness")
    missing = sorted(set(to_drop) - set(columns))
    if missing:
        raise ValueError(f"Configured release columns are absent from concepts.tsv: {missing}")
    released_columns = [column for column in columns if column not in set(to_drop)]
    if "in_muse" in released_columns:
        raise AssertionError("MUSE attestation flag must not be redistributed")
    if not include_concreteness and "concreteness" in released_columns:
        raise AssertionError("Concreteness values must be omitted by configuration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(comment)
        writer = csv.DictWriter(handle, fieldnames=released_columns, delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row.get(column)) for column in released_columns})
    return released_columns, rows


def _field_description(column: str) -> str:
    """Describe each released field, including model/language-indexed columns."""
    if column in FIELD_DESCRIPTIONS:
        return FIELD_DESCRIPTIONS[column]
    if column.startswith("zipf_"):
        language = column.removeprefix("zipf_")
        return f"wordfreq Zipf frequency for the canonical {language} form."
    if column.startswith("n_chars_"):
        language = column.removeprefix("n_chars_")
        return f"Unicode codepoint count, including spaces, in the canonical {language} form."
    if column.startswith("ntok_"):
        parts = column.removeprefix("ntok_").rsplit("_", 1)
        if len(parts) == 2:
            model, language = parts
            return f"{model} tokenizer count for {language}, measured in context as len(tok('x: '+form))−len(tok('x: '))."
    raise ValueError(f"No dataset-card description is defined for released column {column!r}")


def _field_table(columns: list[str]) -> str:
    """Render a table documenting every released TSV column."""
    lines = ["| Column | Meaning |", "|---|---|"]
    for column in columns:
        lines.append(f"| `{column}` | {_field_description(column)} |")
    return "\n".join(lines)


def _share(rows: list[dict[str, str]], column: str) -> str:
    """Format a true-flag share for a selected set of concepts."""
    count = sum(row.get(column, "").casefold() == "true" for row in rows)
    return f"{count:,}/{len(rows):,} ({100 * count / len(rows):.2f}%)" if rows else "0/0 (NA)"


def _agreement_metrics(path: Path) -> dict[str, str]:
    """Extract the primary κ estimate, confidence interval, and H3 status."""
    text = path.read_text(encoding="utf-8")
    primary = next((line for line in text.splitlines() if line.startswith("| primary |")), None)
    if primary is None:
        raise ValueError(f"No primary kappa row in {path}")
    fields = [part.strip() for part in primary.strip().strip("|").split("|")]
    if len(fields) != 4:
        raise ValueError(f"Malformed primary kappa row in {path}: {primary}")
    status_match = re.search(r"H3 status by configured rule: \*\*(exploratory|confirmatory)\*\*", text)
    if not status_match:
        raise ValueError(f"No configured H3 status in {path}")
    reading_match = re.search(r"Primary-verified reading_source = wiktionary_char for (\d+)/(\d+) \(([0-9.]+)\)", text)
    if not reading_match:
        raise ValueError(f"No primary reading-source share in {path}")
    return {
        "kappa_n": fields[1], "kappa": fields[2], "kappa_ci": fields[3],
        "h3_status": status_match.group(1),
        "wiktionary_readings": f"{reading_match.group(1)}/{reading_match.group(2)} ({float(reading_match.group(3)) * 100:.2f}%)",
    }


def _card_blocks(
    rows: list[dict[str, str]], columns: list[str], agreement_path: Path, amendment_path: Path,
) -> dict[str, str]:
    """Render all dynamic dataset-card sections from frozen repository outputs."""
    split_counts = Counter(row["split"] for row in rows)
    test = [row for row in rows if row["split"] == "test"]
    main_test = [row for row in test if row.get("m1_extension", "false").casefold() != "true"]
    extension = [row for row in test if row.get("m1_extension", "false").casefold() == "true"]
    if len(rows) != 1837 or len(main_test) != 1535 or len(extension) != 123 or len(split_counts) != 3:
        raise ValueError(
            "Unexpected frozen package sizes: "
            f"total={len(rows)}, main_test={len(main_test)}, extension={len(extension)}, splits={dict(split_counts)}"
        )
    agreement = _agreement_metrics(agreement_path)
    all_strata = Counter(row["stratum"] for row in main_test)
    noun_n = sum(row["pos"] == "noun" for row in main_test)
    one_sense_main = sum(row["n_senses_vi"] == "1" for row in main_test)
    if one_sense_main != len(main_test):
        raise ValueError(f"Expected one-sense main-test rows; found {one_sense_main}/{len(main_test)}")
    extension_two_sense = sum(row["n_senses_vi"] == "2" for row in extension)
    if extension_two_sense != len(extension):
        raise ValueError(f"Expected two-sense M1 extension rows; found {extension_two_sense}/{len(extension)}")

    source_flags = ("in_vi_gloss", "in_muse", "in_wikidata", "external_attested")
    attestation_lines = ["| Source flag | Main test coverage |", "|---|---|"]
    for flag in source_flags:
        attestation_lines.append(f"| `{flag}` | {_share(main_test, flag)} |")
    routes = Counter(row["bt_route"] for row in main_test)
    route_lines = ["| Back-translation route | Main test concepts |", "|---|---:|"]
    for route in ("strict", "rt_contain", "fwd_only", "none"):
        route_lines.append(f"| `{route}` | {routes.get(route, 0):,}/{len(main_test):,} ({100 * routes.get(route, 0) / len(main_test):.2f}%) |")
    concreteness = Counter(row["concreteness_match"] for row in rows)
    concreteness_lines = ["| Match route | Concepts |", "|---|---:|"]
    for match in ("exact", "head", "none"):
        concreteness_lines.append(f"| `{match}` | {concreteness.get(match, 0):,}/{len(rows):,} ({100 * concreteness.get(match, 0) / len(rows):.2f}%) |")

    split_lines = ["| Split | Rows | Notes |", "|---|---:|---|"]
    split_lines.append(f"| test | {split_counts['test']:,} | Main test {len(main_test):,} plus M1-only extension {len(extension):,}; `m1_extension` distinguishes the extension. |")
    split_lines.append(f"| directions | {split_counts['directions']:,} | Prompt-matched direction concepts. |")
    split_lines.append(f"| fewshot_reservoir | {split_counts['fewshot_reservoir']:,} | Reservoir concepts; selected prompt demonstrations are identified by `fewshot_set`. |")

    amendments = amendment_path.read_text(encoding="utf-8")
    imbalance = "d = +0.89 (Vietnamese frequency), +0.83 (Chinese frequency), −0.83 (concreteness), −0.62 (Vietnamese tokens)."
    if imbalance not in " ".join(amendments.split()):
        # The source line wraps across Markdown lines; compare a whitespace-normalized form.
        compact = " ".join(amendments.split())
        if imbalance not in compact:
            raise ValueError("The preregistered covariate-imbalance statement is missing from PROTOCOL_AMENDMENTS.md")

    return {
        "SUMMARY": (
            f"This release contains **{len(rows):,} aligned concepts** with canonical forms in the five readout languages "
            "(Vietnamese, English, Mandarin Chinese, French, Indonesian) and Russian as the translation-prompt source. "
            f"The frozen partitions contain {len(main_test):,} main-test concepts, {len(extension):,} M1-extension concepts, "
            f"{split_counts['directions']:,} directions concepts, and {split_counts['fewshot_reservoir']:,} few-shot-reservoir concepts."
        ),
        "CONSTRUCTION": (
            "The English Wiktionary/Wiktextract translation tables provide the alignment pivot; Vietnamese Wiktextract entries "
            "supply etymology and gloss cross-checks. Additional attestation uses MUSE and Wikidata; Mandarin strings are checked "
            "against Unihan readings and CC-CEDICT, with WordNet used for single-syllable meaning checks. Frequency comes from "
            "wordfreq, while Brysbaert values are withheld in this package. FLORES+ is used for tokenizer fertility and a separate "
            "gated direction dataset; its sentence text is not included here.\n\n"
            "Selection keeps noun/verb/adjective concepts with English lemmas of at most two words and requires two configured "
            "attestation sources. The pipeline applies the frozen back-translation, median-polysemy, proper-noun, surface-distance, "
            "loanword, and hyphenated-transliteration rules, then assigns the frozen sino/nonsino/ambiguous strata. The v1.2 "
            "amendments and analysis plan are included in [`PROTOCOL_AMENDMENTS.md`](PROTOCOL_AMENDMENTS.md) and [`PREREG.md`](PREREG.md). "
            "The restoration scripts are bundled in [`scripts/`](scripts/); the frozen source-repository tag is `prereg-v1.2`."
        ),
        "FIELDS": _field_table(columns),
        "SPLITS": (
            "\n".join(split_lines) + "\n\n"
            "Partitions are disjoint by concept ID, English lemma, and Vietnamese spelling-equivalence key. The test partition "
            "contains the main test set and the separately flagged M1-only extension; the extension is never part of the main test set."
        ),
        "QUALITY": (
            f"Primary Signal A/B Cohen's κ = **{agreement['kappa']}** (95% CI {agreement['kappa_ci']}; n={agreement['kappa_n']}); "
            f"under the preregistered rule, H3 is **{agreement['h3_status']}**. {agreement['wiktionary_readings']} of verified Han strings "
            "used Wiktionary character entries for readings, so Signal B is partly dependent on Wiktionary.\n\n"
            "Attestation coverage among the main-test concepts:\n\n" + "\n".join(attestation_lines) + "\n\n"
            "MUSE was used only to compute an attestation flag; the flag column and dictionary pairs are not distributed. "
            "Back-translation routes in the main test set:\n\n" + "\n".join(route_lines) + "\n\n"
            "Concreteness match-route counts across all package concepts (scores omitted):\n\n" + "\n".join(concreteness_lines)
        ),
        "LIMITATIONS": (
            f"The main test set is {100 * noun_n / len(main_test):.1f}% nouns, and its concepts have one Vietnamese sense each; "
            "the separate M1 extension has exactly two senses per concept. Etymology classification is automatic (primary κ = "
            f"{float(agreement['kappa']):.3f}); Signal B partly reuses Wiktionary character-entry readings. Vietnamese Zipf frequency "
            "is not directly comparable with Zipf scores for other languages. The preregistered pre-model imbalance was "
            f"{imbalance}"
        ),
        "NOT_INCLUDED": (
            "- `in_muse` is omitted because MUSE is CC BY-NC; restore the flag with [`scripts/add_muse_flag.py`](scripts/add_muse_flag.py) "
            "and the user's own MUSE dictionary files.\n"
            "- Brysbaert concreteness values are omitted until redistribution terms are confirmed; restore them from the user's own "
            "workbook with [`scripts/add_concreteness.py`](scripts/add_concreteness.py). The `concreteness_match` route is retained.\n"
            "- `directions_flores.jsonl` and all FLORES sentence text are omitted because FLORES+ is gated; rebuild from the user's "
            "own local FLORES+ files with [`scripts/build_flores_directions.py`](scripts/build_flores_directions.py)."
        ),
        "LICENSING": (
            "This package is offered under **CC BY-SA 4.0**. Attribution: Wiktionary/Wiktextract (kaikki.org) and wordfreq; "
            "Wikidata structured data is CC0; Unihan is distributed under the Unicode License; CC-CEDICT is CC BY-SA. "
            "NLLB-200 was used only for validation flags and model weights are not redistributed. MUSE pairs, Brysbaert values, "
            "and FLORES sentence text are not included. See [`LICENSE`](LICENSE) and [`SOURCES.md`](SOURCES.md) for the legal-code "
            "reference and provenance."
        ),
    }


def _release_sources_text(source_path: Path) -> str:
    """Copy source provenance while avoiding a recursive post-upload Release section."""
    text = source_path.read_text(encoding="utf-8")
    marker = "\n## Release\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    return text


def build_package(config_path: str | Path) -> dict[str, Any]:
    """Build a deterministic release directory from the frozen release inputs."""
    started = time.monotonic()
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = load_config(config_path)
    release_config = config["release"]
    settings = release_config["package"]
    output = _path(root, settings["output_dir"])
    input_concepts = _path(root, settings["input_concepts"])
    agreement_path = _path(root, settings["agreement"])
    amendments_path = _path(root, settings["amendments"])
    prereg_path = _path(root, settings["prereg"])
    source_path = _path(root, settings["sources"])
    template_path = _path(root, settings["card_template"])
    script_source = _path(root, settings["scripts_source"])
    prompts_dir = _path(root, settings["prompts_dir"])
    required = [input_concepts, agreement_path, amendments_path, prereg_path, source_path, template_path, script_source, prompts_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Release package inputs are missing: " + ", ".join(missing))
    if release_config.get("private") is not True:
        raise ValueError("Release package is required to remain private")
    drop_columns = release_config.get("drop_columns")
    if not isinstance(drop_columns, list) or any(not isinstance(value, str) for value in drop_columns):
        raise ValueError("release.drop_columns must be a list of column names")
    include_concreteness = release_config.get("include_concreteness_values") is True

    comment, source_columns, source_rows_data = _read_tsv(input_concepts)
    del comment
    prompt_files = sorted(path for path in prompts_dir.glob("*.jsonl") if path.name != "directions_flores.jsonl")
    if not prompt_files:
        raise ValueError(f"No non-FLORES prompt JSONL files found in {prompts_dir}")
    scripts = sorted(path for path in script_source.glob("*.py") if path.is_file())
    expected_scripts = {"add_muse_flag.py", "add_concreteness.py", "build_flores_directions.py"}
    if {path.name for path in scripts} != expected_scripts:
        raise ValueError(f"Release script source directory must contain exactly {sorted(expected_scripts)}")
    expected_files = {
        "README.md", "LICENSE", "concepts.tsv", "agreement.md", "PROTOCOL_AMENDMENTS.md", "PREREG.md", "SOURCES.md",
        *(f"prompts/{path.name}" for path in prompt_files), *(f"scripts/{path.name}" for path in scripts),
    }
    if output.exists():
        existing = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
        unknown = sorted(existing - expected_files)
        if unknown:
            raise ValueError(f"Unrecognized files already exist under {output}: {unknown}")
    output.mkdir(parents=True, exist_ok=True)
    concepts_path = output / "concepts.tsv"
    columns, rows = _write_concepts(input_concepts, concepts_path, drop_columns, include_concreteness)

    shutil.copyfile(agreement_path, output / "agreement.md")
    shutil.copyfile(amendments_path, output / "PROTOCOL_AMENDMENTS.md")
    shutil.copyfile(prereg_path, output / "PREREG.md")
    (output / "SOURCES.md").write_text(_release_sources_text(source_path), encoding="utf-8", newline="\n")
    (output / "prompts").mkdir(parents=True, exist_ok=True)
    for prompt in prompt_files:
        shutil.copyfile(prompt, output / "prompts" / prompt.name)
    (output / "scripts").mkdir(parents=True, exist_ok=True)
    for script in scripts:
        shutil.copyfile(script, output / "scripts" / script.name)
    license_text = (
        "viLens Concepts is offered under the Creative Commons Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0).\n\n"
        "Full legal code: https://creativecommons.org/licenses/by-sa/4.0/legalcode\n"
        "License summary: https://creativecommons.org/licenses/by-sa/4.0/\n\n"
        "Attribution: Wiktionary/Wiktextract (kaikki.org) and wordfreq. Additional source provenance and licenses are listed in SOURCES.md.\n"
    )
    (output / "LICENSE").write_text(license_text, encoding="utf-8", newline="\n")
    template = template_path.read_text(encoding="utf-8")
    blocks = _card_blocks(source_rows_data, columns, agreement_path, amendments_path)
    card = template
    for key, value in blocks.items():
        card = card.replace("{{" + key + "}}", value)
    unresolved = re.findall(r"\{\{[A-Z_]+\}\}", card)
    if unresolved:
        raise ValueError(f"Dataset card has unresolved placeholders: {unresolved}")
    (output / "README.md").write_text(card, encoding="utf-8", newline="\n")

    actual_files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(f"Unexpected release file set: missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}")
    runtime = time.monotonic() - started
    return {
        "output": str(output), "rows": len(rows), "columns": columns,
        "dropped_columns": [column for column in source_columns if column not in columns],
        "files": sorted(actual_files), "runtime_seconds": runtime,
    }


def main() -> int:
    """Build the private-ready package and print its non-sensitive summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    try:
        result = build_package(args.config)
    except Exception as exc:
        logger.error("Release package build stopped: %s: %s", type(exc).__name__, exc)
        raise SystemExit(1) from exc
    logger.info("Output: %s (%d files; %d concepts; %d columns)", result["output"], len(result["files"]), result["rows"], len(result["columns"]))
    logger.info("Dropped columns: %s", result["dropped_columns"])
    logger.info("Runtime seconds: %.3f", result["runtime_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
