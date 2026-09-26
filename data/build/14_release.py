"""Assemble the final concept TSV, audit report, licenses, and QA sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

try:
    from common import load_config, normalize_nfc, normalize_strings, require_resolved_concreteness, setup_logging, vi_orth_key
except ModuleNotFoundError:
    from data.build.common import load_config, normalize_nfc, normalize_strings, require_resolved_concreteness, setup_logging, vi_orth_key


STEP = "14_release"
SPLIT_ORDER = ("fewshot_reservoir", "directions", "test")
STRATUM_ORDER = ("sino", "nonsino", "ambiguous", "other_loan")
LANGUAGES = ("vi", "en", "zh", "fr", "id", "ru")
ATTESTATION_FLAGS = ("in_vi_gloss", "in_muse", "in_wikidata", "external_attested")


def missing_input_message(path: Path) -> str:
    """Return the stable diagnostic for absent step-12 release data."""
    return f"Step 14 requires step-12 enriched input: missing {path}"


def selected_vi_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Find the attestation record matching the row's chosen VI spelling."""
    candidates = row.get("vi_cands")
    if not isinstance(candidates, list):
        raise ValueError(f"Release row {row.get('concept_id')!r} lacks vi_cands for attestation flags")
    canonical_key = vi_orth_key(row["vi_canonical"])
    matches = [item for item in candidates if isinstance(item, dict) and isinstance(item.get("word"), str)
               and vi_orth_key(item["word"]) == canonical_key]
    if not matches:
        raise ValueError(f"No selected VI candidate matches the canonical form for {row.get('concept_id')!r}")
    for item in matches:
        for flag in ATTESTATION_FLAGS:
            if type(item.get(flag)) is not bool:
                raise ValueError(f"Selected VI candidate for {row.get('concept_id')!r} lacks boolean {flag}")
    return {flag: any(item.get(flag) is True for item in matches)
            for flag in (*ATTESTATION_FLAGS, "in_wiktextract")}


def derive_reading_source(row: dict[str, Any]) -> str:
    """Summarize reading sources used by primary-verified Han strings."""
    verifications = row.get("han_verifications")
    if not isinstance(verifications, list):
        raise ValueError(f"Release row {row.get('concept_id')!r} lacks Signal-B verification records")
    sources = sorted({
        item["reading_source"] for item in verifications
        if isinstance(item, dict) and item.get("primary_pass") is True
        and isinstance(item.get("reading_source"), str) and item["reading_source"]
    })
    return "+".join(sources)


def release_columns(languages: list[str], models: list[str]) -> list[str]:
    """Return the fixed public column order, expanded for configured languages/models."""
    if tuple(languages) != ("vi", "en", "zh", "fr", "id"):
        raise ValueError(f"Release language order must be vi/en/zh/fr/id, got {languages!r}")
    if not models or len(models) != len(set(models)):
        raise ValueError(f"Release model list must be non-empty and unique, got {models!r}")
    columns = [
        "concept_id", "split", "vi", "en", "zh", "fr", "id", "ru", "vi_alts", "vi_nodiac",
        "collapsed", "pos", "en_sense_gloss", "stratum", "sigA", "sigB", "sigB_strict",
        "sigB_relaxed", "sino_via", "native_strict", "reading_source", "n_sources", "in_vi_gloss",
        "in_muse", "in_wikidata", "external_attested", "bt_route", "bt_pass_strict_v1",
        "zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree", "ru_nllb_agree", "min_surface_dist",
        "n_senses_vi",
    ]
    columns.extend(f"zipf_{language}" for language in languages)
    columns.append("n_syllables")
    columns.extend(f"n_chars_{language}" for language in languages)
    columns.extend(["concreteness", "concreteness_match"])
    for model in models:
        columns.extend(f"ntok_{model}_{language}" for language in languages[:5])
    columns.extend(f"single_token_vi_{model}" for model in models)
    columns.extend(f"m1_eligible_{model}" for model in models)
    columns.extend(["m1_extension", "cloze_available", "fewshot_set"])
    return columns


def required_release_columns(languages: list[str], models: list[str]) -> set[str]:
    """Return all columns that must exist in the enriched parquet before export."""
    columns = set(release_columns(languages, models))
    columns.discard("cloze_available")
    columns.discard("fewshot_set")
    columns.discard("reading_source")
    columns.discard("in_vi_gloss")
    columns.discard("in_muse")
    columns.discard("in_wikidata")
    columns.update({
        "vi_cands", "vi_canonical", "en_lemma", "zh_canonical", "fr_canonical", "id_canonical",
        "ru_canonical", "sense_gloss", "signal_a", "signal_b", "signal_b_strict", "signal_b_relaxed",
        "han_verifications",
    })
    columns.difference_update({"vi", "en", "zh", "fr", "id", "ru", "en_sense_gloss", "sigA", "sigB",
                               "sigB_strict", "sigB_relaxed"})
    return columns


def project_release_row(
    row: dict[str, Any], *, columns: list[str], fewshot_sets: dict[str, int],
    cloze_ids: set[str],
) -> dict[str, Any]:
    """Project one enriched pipeline row onto the explicitly declared TSV schema."""
    candidate = selected_vi_candidate(row)
    if candidate["external_attested"] != row["external_attested"]:
        raise ValueError(f"Selected VI candidate external_attested disagrees with row for {row.get('concept_id')!r}")
    canonical = {"vi": row.get("vi_canonical"), "en": row.get("en_lemma"),
                 "zh": row.get("zh_canonical"), "fr": row.get("fr_canonical"),
                 "id": row.get("id_canonical"), "ru": row.get("ru_canonical")}
    required_strings = [language for language, value in canonical.items()
                        if language != "ru" and (not isinstance(value, str) or not value.strip())]
    if required_strings:
        raise ValueError(f"Release row {row.get('concept_id')!r} lacks canonical form(s): {required_strings!r}")
    if canonical["ru"] is not None and (not isinstance(canonical["ru"], str) or not canonical["ru"].strip()):
        raise ValueError(f"Release row {row.get('concept_id')!r} has malformed Russian canonical form")
    if type(row.get("external_attested")) is not bool:
        raise ValueError(f"Release row {row.get('concept_id')!r} lacks boolean external_attested")
    if type(row.get("m1_extension")) is not bool or type(row.get("collapsed")) is not bool:
        raise ValueError(f"Release row {row.get('concept_id')!r} has malformed collapse/extension flags")
    result: dict[str, Any] = {
        "concept_id": row.get("concept_id"), "split": row.get("split"), **canonical,
        "vi_alts": row.get("vi_alts"), "vi_nodiac": row.get("vi_nodiac"),
        "collapsed": row.get("collapsed"), "pos": row.get("pos"),
        "en_sense_gloss": row.get("sense_gloss"), "stratum": row.get("stratum"),
        "sigA": row.get("signal_a"), "sigB": row.get("signal_b"),
        "sigB_strict": row.get("signal_b_strict"), "sigB_relaxed": row.get("signal_b_relaxed"),
        "sino_via": row.get("sino_via"), "native_strict": row.get("native_strict"),
        "reading_source": derive_reading_source(row), "n_sources": row.get("n_sources"),
        "in_vi_gloss": candidate.get("in_vi_gloss"), "in_muse": candidate.get("in_muse"),
        "in_wikidata": candidate.get("in_wikidata"), "external_attested": row.get("external_attested"),
        "bt_route": row.get("bt_route"), "bt_pass_strict_v1": row.get("bt_pass_strict_v1"),
        "zh_nllb_agree": row.get("zh_nllb_agree"), "fr_nllb_agree": row.get("fr_nllb_agree"),
        "id_nllb_agree": row.get("id_nllb_agree"), "ru_nllb_agree": row.get("ru_nllb_agree"),
        "min_surface_dist": row.get("min_surface_dist"), "n_senses_vi": row.get("n_senses_vi"),
        "n_syllables": row.get("n_syllables"), "concreteness": row.get("concreteness"),
        "concreteness_match": row.get("concreteness_match"),
        "m1_extension": row.get("m1_extension"), "cloze_available": row["concept_id"] in cloze_ids,
        "fewshot_set": fewshot_sets.get(row["concept_id"]),
    }
    for language in ("vi", "en", "zh", "fr", "id"):
        result[f"zipf_{language}"] = row.get(f"zipf_{language}")
        result[f"n_chars_{language}"] = row.get(f"n_chars_{language}")
    for model in (column.removeprefix("single_token_vi_") for column in columns if column.startswith("single_token_vi_")):
        result[f"single_token_vi_{model}"] = row.get(f"single_token_vi_{model}")
        result[f"m1_eligible_{model}"] = row.get(f"m1_eligible_{model}")
        for language in ("vi", "en", "zh", "fr", "id"):
            result[f"ntok_{model}_{language}"] = row.get(f"ntok_{model}_{language}")
    absent = [name for name in columns if name not in result]
    if absent:
        raise AssertionError(f"Release projection did not populate declared fields: {absent!r}")
    return {name: result.get(name) for name in columns}


def _cell(value: Any) -> str:
    """Serialize TSV values without lossy formatting or Python repr syntax."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return normalize_nfc(str(value))


def render_concepts_tsv(rows: list[dict[str, Any]], columns: list[str], commit_hash: str) -> str:
    """Render the final UTF-8 TSV with a commit-identifying comment line."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit_hash):
        raise ValueError(f"Pipeline commit hash must be 40 lowercase hex characters, got {commit_hash!r}")
    sorted_rows = sorted(rows, key=lambda row: normalize_nfc(row["concept_id"]))
    if len({row["concept_id"] for row in sorted_rows}) != len(sorted_rows):
        raise ValueError("Release rows contain duplicate concept_id values")
    import io
    buffer = io.StringIO(newline="")
    buffer.write(f"# Pipeline commit: {commit_hash}\n")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in sorted_rows:
        writer.writerow({name: _cell(row.get(name)) for name in columns})
    return buffer.getvalue()


def latest_dropflow_records(path: Path) -> list[dict[str, Any]]:
    """Read dropflow JSONL and retain the most recent recorded run of each step."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed dropflow JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("step"), str):
                raise ValueError(f"Malformed dropflow record at {path}:{line_number}")
            records.append(item)
    block_starts: dict[str, list[int]] = defaultdict(list)
    previous_step: str | None = None
    for index, item in enumerate(records):
        if item["step"] != previous_step:
            block_starts[item["step"]].append(index)
            previous_step = item["step"]
    last_start = {step: starts[-1] for step, starts in block_starts.items()}
    return [item for index, item in enumerate(records) if index >= last_start[item["step"]]]


def sha256_file(path: Path, *, chunk_size_bytes: int) -> str:
    """Hash a file with bounded memory."""
    if type(chunk_size_bytes) is not int or chunk_size_bytes < 1:
        raise ValueError("SHA-256 chunk size must be a positive integer")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pipe_table(path: Path) -> list[dict[str, str]]:
    """Parse the pinned SOURCES Markdown table."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith("| source |")), None)
    if header_index is None:
        raise ValueError(f"SOURCES.md has no provenance table header: {path}")
    headers = [normalize_nfc(cell.strip()) for cell in lines[header_index].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2:]:
        if not line.startswith("|"):
            break
        cells = [normalize_nfc(cell.strip().replace("\\|", "|"))
                 for cell in re.split(r"(?<!\\)\|", line.strip("|"))]
        if len(cells) != len(headers):
            raise ValueError(f"Malformed SOURCES.md row: {line!r}")
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def _read_fewshot_metadata(path: Path) -> dict[str, Any]:
    """Load and validate step-12's machine-readable few-shot/cloze evidence."""
    if not path.is_file():
        raise FileNotFoundError(f"Step 14 requires structured step-12 metadata: missing {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed step-12 metadata JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported or malformed step-12 metadata schema in {path}")
    payload = normalize_strings(payload)
    raw_sets = payload.get("fewshot_sets")
    if not isinstance(raw_sets, dict) or not raw_sets:
        raise ValueError(f"Step-12 metadata has no fewshot_sets mapping: {path}")
    seen_ids: set[str] = set()
    fewshot_set_by_id: dict[str, int] = {}
    for raw_number, ids in sorted(raw_sets.items(), key=lambda item: int(item[0])):
        try:
            set_number = int(raw_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed few-shot set number {raw_number!r} in {path}") from exc
        if set_number < 1 or not isinstance(ids, list) or not ids:
            raise ValueError(f"Malformed few-shot set {raw_number!r} in {path}")
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Few-shot set {raw_number!r} has malformed or duplicate concept IDs")
        overlap = seen_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Concept IDs occur in multiple few-shot sets: {sorted(overlap)[:10]!r}")
        seen_ids.update(ids)
        fewshot_set_by_id.update({concept_id: set_number for concept_id in ids})
    for name in ("cloze_demonstration_concept_ids", "test_cloze_concept_ids"):
        ids = payload.get(name)
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(f"Step-12 metadata field {name!r} must be a list of non-empty IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Step-12 metadata field {name!r} contains duplicate IDs")
    if set(payload["cloze_demonstration_concept_ids"]) - seen_ids:
        raise ValueError("Cloze demonstrations must be selected from the few-shot sets")
    if payload.get("cloze_status") not in {"primary", "supplementary"}:
        raise ValueError(f"Malformed cloze_status in step-12 metadata: {payload.get('cloze_status')!r}")
    coverage = payload.get("main_test_cloze_coverage")
    if not isinstance(coverage, dict) or type(coverage.get("n_available")) is not int or type(coverage.get("n_test")) is not int:
        raise ValueError("Step-12 metadata lacks integer main-test cloze coverage counts")
    if coverage["n_available"] < 0 or coverage["n_test"] < coverage["n_available"]:
        raise ValueError("Step-12 metadata has invalid main-test cloze coverage counts")
    if not isinstance(coverage.get("by_stratum"), dict):
        raise ValueError("Step-12 metadata main-test cloze coverage lacks by_stratum")
    payload["fewshot_set_by_id"] = fewshot_set_by_id
    return payload


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        rendered.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(rendered)


def _dropflow_split(stage: str) -> tuple[str, str]:
    """Separate common split suffixes from stage names for the agreement table."""
    for split in ("fewshot_reservoir", "main_test", "directions", "test"):
        suffix = "_" + split
        if stage.endswith(suffix):
            return split, stage[:-len(suffix)]
    return "all", stage


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.4f})" if denominator else "0/0 (NA)"


def _load_step_module(step_file: str) -> Any:
    """Load an adjacent pipeline module for shared, audited calculations."""
    module_path = Path(__file__).resolve().parent / step_file
    spec = importlib.util.spec_from_file_location(f"vilens_release_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _muse_vi_space_summary(config: dict[str, Any]) -> tuple[int, int]:
    """Recompute the MUSE VI-side multiword finding from the two source files."""
    paths = config["attest"]["paths"]
    step04 = _load_step_module("04_attest.py")
    pairs = step04.load_muse_pairs(paths["muse_vi_en"], "vi-en")
    pairs.update(step04.load_muse_pairs(paths["muse_en_vi"], "en-vi"))
    forms = {vi for vi, _en in pairs}
    return sum(" " in form for form in forms), len(forms)


def build_agreement(
    rows: list[dict[str, Any]], *, dropflow: list[dict[str, Any]], sources: list[dict[str, str]],
    config: dict[str, Any], fewshot_metadata: dict[str, Any], fertility_path: Path,
    flores_path: Path, protocol_path: Path,
) -> str:
    """Build the reproducibility report only from structured outputs and source files."""
    main_rows = [row for row in rows if row.get("m1_extension") is not True]
    lines = ["# Dataset agreement and release audit", "",
             f"Protocol: [{protocol_path.name}](../{protocol_path.name})  ",
             f"Source provenance: [{Path(config['release']['paths']['sources']).name}](../build/SOURCES.md)", ""]
    lines += ["## Dropflow (latest run per step)", "",
              _markdown_table(["step", "split", "stage", "unit", "n_in", "n_out", "n_out_by_pos"],
                              ([item.get("step", ""), *_dropflow_split(str(item.get("stage", ""))),
                                item.get("unit", ""),
                                item.get("n_in", ""), item.get("n_out", ""),
                                json.dumps(item.get("n_out_by_pos", {}), ensure_ascii=False, sort_keys=True)] for item in dropflow)), ""]

    lines += ["## Attestation", "", "Shares are computed on the released rows within each split; flags are sourced from the selected VI candidate.", ""]
    attestation_rows = []
    for split in SPLIT_ORDER:
        selected = [row for row in main_rows if row["split"] == split]
        for flag in ATTESTATION_FLAGS:
            count = sum(bool(selected_vi_candidate(row).get(flag)) if flag != "external_attested" else bool(row.get(flag)) for row in selected)
            attestation_rows.append((split, flag, _ratio(count, len(selected))))
    lines += [_markdown_table(["split", "flag", "share"], attestation_rows), ""]
    try:
        muse_multiword, muse_total = _muse_vi_space_summary(config)
        lines += [f"MUSE unique VI-side forms containing a space: {_ratio(muse_multiword, muse_total)}.", ""]
    except (KeyError, FileNotFoundError) as exc:
        lines += [f"MUSE unique VI-side forms containing a space: unavailable ({exc}).", ""]

    lines += ["## Etymology agreement and H3", ""]
    etym_settings = config["etymology"]
    step08 = _load_step_module("08_etymology.py")
    kappa_rows = [row for row in main_rows if row.get("signal_a") in {"sino", "nonsino"}]
    a_labels = [row["signal_a"] for row in kappa_rows]
    kappa_results: dict[str, tuple[float | None, float | None, float | None]] = {}
    for label, field in (("primary", "signal_b"), ("strict", "signal_b_strict"),
                         ("relaxed", "signal_b_relaxed")):
        result = step08.bootstrap_kappa_ci(
            a_labels, [row[field] for row in kappa_rows],
            resamples=int(etym_settings["bootstrap_resamples"]),
            confidence=float(etym_settings["bootstrap_confidence"]), seed=int(config["seed"]),
        )
        kappa_results[label] = result
    crosstab = Counter((row.get("signal_a"), row.get("signal_b")) for row in main_rows)
    lines += ["Signal A × primary Signal B:", "",
              _markdown_table(["A", "B=sino", "B=nonsino"],
                              ((label, crosstab[(label, "sino")], crosstab[(label, "nonsino")])
                               for label in ("sino", "nonsino", "other_loan", "conflict"))), "",
              "Cohen's κ (A ∈ {sino, nonsino}); seeded percentile bootstrap:", "",
              _markdown_table(["B variant", "n", "κ", "95% CI"],
                              ((label, len(kappa_rows),
                                f"{estimate:.6f}" if estimate is not None else "NA",
                                f"[{lower:.6f}, {upper:.6f}]" if lower is not None and upper is not None else "NA")
                               for label, (estimate, lower, upper) in kappa_results.items())), ""]
    primary_verified = [item for row in main_rows for item in (row.get("han_verifications") or [])
                        if isinstance(item, dict) and item.get("primary_pass") is True]
    char_readings = sum(item.get("reading_source") == "wiktionary_char" for item in primary_verified)
    lines += [f"Primary-verified reading_source = wiktionary_char for {_ratio(char_readings, len(primary_verified))}.", ""]
    primary_lower = kappa_results["primary"][1]
    test_rows = [row for row in main_rows if row.get("split") == "test"]
    n_sino = sum(row.get("stratum") == "sino" for row in test_rows)
    n_nonsino = sum(row.get("stratum") == "nonsino" for row in test_rows)
    n_native = sum(row.get("native_strict") is True for row in test_rows)
    sino_via_counts = Counter(value for row in main_rows for value in (row.get("sino_via") or []))
    lines += [f"sino_via counts: `{json.dumps(dict(sorted(sino_via_counts.items())), ensure_ascii=False, sort_keys=True)}`.",
              f"TEST native_strict concepts: {n_native}.", ""]
    power_alpha = float(etym_settings["power_alpha"])
    power_divisor = int(etym_settings["power_holm_comparisons"])
    target_power = float(etym_settings["power_target"])
    power_rows = []
    adjusted_sino_nonsino: float | None = None
    for comparison, n1, n2 in (("sino_vs_nonsino", n_sino, n_nonsino),
                               ("sino_vs_native_strict", n_sino, n_native)):
        unadjusted = step08.minimum_detectable_difference(n1, n2, alpha=power_alpha, power=target_power)
        adjusted = step08.minimum_detectable_difference(n1, n2, alpha=power_alpha / power_divisor, power=target_power)
        if comparison == "sino_vs_nonsino":
            adjusted_sino_nonsino = adjusted
        power_rows.append((comparison, n1, n2,
                           f"{unadjusted:.6f}" if unadjusted is not None else "NA",
                           f"{adjusted:.6f}" if adjusted is not None else "NA"))
    confirmatory = (primary_lower is not None
                    and primary_lower >= float(etym_settings["h3_confirmatory_kappa_lower_min"])
                    and adjusted_sino_nonsino is not None
                    and adjusted_sino_nonsino <= float(etym_settings["h3_confirmatory_holm_mde_max"]))
    primary_lower_text = f"{primary_lower:.6f}" if primary_lower is not None else "NA"
    adjusted_mde_text = f"{adjusted_sino_nonsino:.6f}" if adjusted_sino_nonsino is not None else "NA"
    lines += ["Power (two-sided standardized MDE, 80% target):", "",
              _markdown_table(["comparison", "n1", "n2", "d at α=.05", "d at Holm α"], power_rows), "",
              f"H3 status by configured rule: **{'confirmatory' if confirmatory else 'exploratory'}** "
              f"(primary κ lower bound={primary_lower_text}; Holm MDE={adjusted_mde_text}).", ""]

    lines += ["## Strata", ""]
    strata_rows = []
    for split in SPLIT_ORDER:
        subset = [row for row in main_rows if row["split"] == split]
        counts = Counter(row["stratum"] for row in subset)
        for stratum in STRATUM_ORDER:
            strata_rows.append((split, stratum, counts[stratum]))
    lines += [_markdown_table(["split", "stratum", "concepts"], strata_rows), ""]
    syllable_edges = config["split"]["syllable_bins"]
    if len(syllable_edges) != 2 or syllable_edges != sorted(set(syllable_edges)):
        raise ValueError(f"Invalid split.syllable_bins in release config: {syllable_edges!r}")
    syllable_labels = (str(syllable_edges[0]), str(syllable_edges[1]), f"{syllable_edges[1] + 1}+")
    syllable_rows = []
    for row in main_rows:
        if row["split"] != "test":
            continue
        syllables = row.get("n_syllables")
        if syllables is None:
            syllables = len(vi_orth_key(row["vi_canonical"]).split())
        bucket = (syllable_labels[0] if syllables <= syllable_edges[0] else
                  syllable_labels[1] if syllables <= syllable_edges[1] else syllable_labels[2])
        syllable_rows.append((row["pos"], bucket, row["stratum"]))
    counts = Counter(syllable_rows)
    lines += ["TEST stratum by POS × syllable bucket:", "",
              _markdown_table(["POS", "syllables", "stratum", "concepts"],
                              ((pos, bucket, stratum, counts[(pos, bucket, stratum)])
                               for pos, bucket, stratum in sorted(counts))), ""]

    lines += ["## Diacritic collapse", ""]
    collapse_rows = []
    for split in SPLIT_ORDER:
        for stratum in STRATUM_ORDER:
            selected = [row for row in main_rows if row["split"] == split and row["stratum"] == stratum]
            collapsed = sum(bool(row.get("collapsed")) for row in selected)
            if selected:
                collapse_rows.append((split, stratum, _ratio(collapsed, len(selected))))
    lines += [_markdown_table(["split", "stratum", "collapsed"], collapse_rows), ""]

    lines += ["## E1 tokenization and M1", "", "Token counts and single-token coverage are computed on the TEST rows.", ""]
    model_names = list(config["tokens"].get("models", ["gemma", "qwen", "llama"]))
    token_rows = []
    test_rows = [row for row in main_rows if row["split"] == "test"]
    for model in model_names:
        for language in config["langs"]:
            field = f"ntok_{model}_{language}"
            values = [int(row[field]) for row in test_rows if isinstance(row.get(field), (int, float))]
            if values:
                token_rows.append((model, language, f"{np.mean(values):.3f}", f"{np.median(values):.3f}", max(values)))
    lines += [_markdown_table(["model", "lang", "ntok mean", "median", "max"], token_rows), "", "Single-token VI coverage by stratum:", ""]
    single_rows = []
    for model in model_names:
        field = f"single_token_vi_{model}"
        for stratum in STRATUM_ORDER:
            selected = [row for row in test_rows if row["stratum"] == stratum]
            n = sum(row.get(field) is True for row in selected)
            if selected:
                single_rows.append((model, stratum, _ratio(n, len(selected))))
    lines += [_markdown_table(["model", "stratum", "single-token VI"], single_rows), ""]
    single_by_syllable = []
    for model in model_names:
        field = f"single_token_vi_{model}"
        for bucket in syllable_labels:
            selected = []
            for row in test_rows:
                count = row.get("n_syllables")
                if count is None:
                    count = len(vi_orth_key(row["vi_canonical"]).split())
                row_bucket = (syllable_labels[0] if count <= syllable_edges[0] else
                              syllable_labels[1] if count <= syllable_edges[1] else syllable_labels[2])
                if row_bucket == bucket:
                    selected.append(row)
            n = sum(row.get(field) is True for row in selected)
            if selected:
                single_by_syllable.append((model, bucket, _ratio(n, len(selected))))
    lines += ["Single-token VI coverage by syllable bucket:", "",
              _markdown_table(["model", "syllable bucket", "single-token VI"], single_by_syllable), ""]
    m1_rows = []
    for model in model_names:
        field = f"m1_eligible_{model}"
        selected = [row for row in rows if row.get("split") == "test" or row.get("m1_extension") is True]
        m1_rows.append((model, sum(row.get(field) is True for row in selected)))
    extension_n = sum(row.get("m1_extension") is True for row in rows)
    lines += [f"M1 extension concepts: {extension_n}", "", _markdown_table(["model", "M1 eligible (main test + extension)"], m1_rows), ""]

    lines += ["## FLORES fertility", ""]
    if fertility_path.is_file():
        with fertility_path.open("r", encoding="utf-8", newline="") as handle:
            fertility = list(csv.DictReader(handle))
        lines += [_markdown_table(["model", "lang", "tokens/character", "tokens/syllable"],
                                  ((item["model"], item["lang"], item["tokens_per_character"], item["tokens_per_syllable"] or "NA") for item in fertility)), ""]
    else:
        lines += ["Not yet available; `make fertility` writes `data/interim/e1_fertility.csv`.", ""]
    if flores_path.is_file():
        lines += [f"FLORES direction-prompt JSONL SHA-256 (text not reproduced here): `"
                  f"{sha256_file(flores_path, chunk_size_bytes=int(config['downloads']['chunk_size_bytes']))}`.", ""]
    else:
        lines += ["FLORES direction-prompt JSONL SHA-256: not yet available.", ""]

    coverage = fewshot_metadata["main_test_cloze_coverage"]
    lines += ["## Cloze and few-shot", "", f"Recorded cloze status: **{fewshot_metadata['cloze_status']}**.", "",
              f"Step-12 main-test cloze coverage: {coverage['n_available']}/{coverage['n_test']}.", "",
              "Main-test cloze coverage by stratum:", "",
              _markdown_table(["stratum", "available", "main test"],
                              ((stratum, counts["n_available"], counts["n_test"])
                               for stratum, counts in sorted(coverage["by_stratum"].items()))), ""]
    rows_by_id = {row["concept_id"]: row for row in rows}
    lines += ["Selected few-shot concepts (set | en | vi | ru | stratum):", "", "```text"]
    for set_number, ids in sorted(fewshot_metadata["fewshot_sets"].items(), key=lambda item: int(item[0])):
        for concept_id in ids:
            row = rows_by_id.get(concept_id)
            if row is None:
                raise ValueError(f"Selected few-shot concept {concept_id!r} is absent from release parquet")
            lines.append(" | ".join((set_number, row["en_lemma"], row["vi_canonical"],
                                      row.get("ru_canonical") or "", row["stratum"])))
    lines += ["```", ""]

    lines += ["## Sources", "", "See the full file-level table in [SOURCES.md](../build/SOURCES.md).", ""]
    return "\n".join(lines)


def build_licenses(sources: list[dict[str, str]]) -> str:
    """Build source-level license obligations and the proposed release license."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in sources:
        grouped[row["source"]].add(row["license"])
    if "wordfreq" not in grouped:
        grouped["wordfreq"].add("CC BY-SA 4.0")
    if "flores" not in grouped:
        grouped["flores"].add("Not retrieved; gated dataset terms must be reviewed at the source")
    policies = {
        "wiktextract_en": ("yes", "Attribute Wiktionary/Wiktextract and distribute adapted content under CC BY-SA."),
        "wiktextract_vi": ("yes", "Attribute Wiktionary/Wiktextract and distribute adapted content under CC BY-SA."),
        "wordfreq": ("yes", "Attribute the source and comply with CC BY-SA share-alike terms."),
        "muse": ("flag-only", "Non-commercial license; retain only boolean attestation flags. HUMAN REVIEW."),
        "brysbaert": ("yes", "Concrete scores appear in concepts.tsv; verify publisher terms before redistribution. HUMAN REVIEW."),
        "flores": ("no", "Gated terms apply; FLORES sentence text is never redistributed."),
        "unihan": ("no", "Retain Unicode notices; no Unihan reading table is redistributed."),
        "cedict": ("no", "CC BY-SA attribution/share-alike; CEDICT gloss text is not in concepts.tsv."),
        "wordnet": ("no", "WordNet License; WordNet corpus is not redistributed in concepts.tsv."),
        "nllb": ("no", "Model output used only for preprocessing flags/canonical selection; no weights redistributed."),
        "lid": ("no", "Language-identification model files are not redistributed."),
        "gated": ("no", "Access metadata only; no weights redistributed."),
    }
    rows = []
    for source in sorted(grouped):
        license_text = " / ".join(sorted(grouped[source]))
        redistributed, obligations = policies.get(source, ("no", "Review the source license; source payload is not included."))
        rows.append((source, license_text, redistributed, obligations))
    return "\n".join([
        "# Data licenses", "", "Proposed release license for the derived concept dataset: **CC BY-SA 4.0**.",
        "Human legal review is required for the flagged rows below.", "",
        _markdown_table(["source", "license as recorded", "redistributed in concepts.tsv", "obligations / review"], rows), "",
    ])


def _read_latest_inputs(config: dict[str, Any], logger: logging.Logger) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Load all mandatory release evidence before any output file is written."""
    release = config["release"]
    paths = {name: Path(value) for name, value in release["paths"].items()}
    input_path = paths["input"]
    if not input_path.is_file():
        raise FileNotFoundError(missing_input_message(input_path))
    table = pq.read_table(input_path)
    rows = [normalize_strings(row) for row in table.to_pylist()]
    if not rows:
        raise ValueError(f"Step-12 enriched input contains no rows: {input_path}")
    rows.sort(key=lambda row: normalize_nfc(row.get("concept_id", "")))
    ids = [row.get("concept_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Step-12 enriched input has missing or duplicate concept_id values")
    require_resolved_concreteness(rows)
    required_base = {"concept_id", "split", "vi_canonical", "en_lemma", "zh_canonical", "fr_canonical", "id_canonical",
                     "pos", "sense_gloss", "stratum", "signal_a", "signal_b", "signal_b_strict",
                     "signal_b_relaxed", "sino_via", "native_strict", "n_sources", "bt_route",
                     "bt_pass_strict_v1", "min_surface_dist", "n_senses_vi", "vi_nodiac", "collapsed",
                     "concreteness", "concreteness_match", "m1_extension", "han_verifications"}
    required_base.update(required_release_columns(config["langs"], list(config["models"])))
    missing_fields = sorted(required_base - set(table.column_names))
    if missing_fields:
        raise ValueError(f"Step-12 enriched input is missing required release columns: {missing_fields!r}")
    if any(row.get("split") not in SPLIT_ORDER for row in rows):
        raise ValueError(f"Unexpected split labels in release input: {sorted({str(row.get('split')) for row in rows})!r}")
    allowed_concreteness = {"exact", "head", "none"}
    bad_matches = sorted({row.get("concreteness_match") for row in rows} - allowed_concreteness, key=str)
    if bad_matches:
        raise ValueError(f"Unexpected concreteness_match values: {bad_matches!r}")
    if not paths["dropflow"].is_file():
        raise FileNotFoundError(f"Step 14 requires dropflow history: missing {paths['dropflow']}")
    if not paths["sources"].is_file():
        raise FileNotFoundError(f"Step 14 requires provenance: missing {paths['sources']}")
    sources = _read_pipe_table(paths["sources"])
    dropflow = latest_dropflow_records(paths["dropflow"])
    fewshot_path = Path(config["prompts"]["paths"]["fewshot_metadata"])
    fewshot_metadata = _read_fewshot_metadata(fewshot_path)
    row_ids = set(ids)
    selected_ids = set(fewshot_metadata["fewshot_set_by_id"])
    missing_selected = sorted(selected_ids - row_ids)
    if missing_selected:
        raise ValueError(f"Step-12 metadata selects IDs absent from release parquet: {missing_selected[:10]!r}")
    rows_by_id = {row["concept_id"]: row for row in rows}
    wrong_fewshot_split = sorted(concept_id for concept_id in selected_ids
                                 if rows_by_id[concept_id].get("split") != "fewshot_reservoir")
    if wrong_fewshot_split:
        raise ValueError(f"Step-12 metadata few-shot IDs are not in the reservoir: {wrong_fewshot_split[:10]!r}")
    cloze_ids = set(fewshot_metadata["test_cloze_concept_ids"])
    missing_cloze = sorted(cloze_ids - row_ids)
    wrong_cloze_split = sorted(concept_id for concept_id in cloze_ids & row_ids
                               if rows_by_id[concept_id].get("split") != "test")
    if missing_cloze or wrong_cloze_split:
        raise ValueError(f"Step-12 metadata has invalid test cloze IDs: missing={missing_cloze[:10]!r}; "
                         f"wrong_split={wrong_cloze_split[:10]!r}")
    main_rows = [row for row in rows if row.get("split") == "test" and row.get("m1_extension") is False]
    listed_main_cloze = {concept_id for concept_id in cloze_ids
                         if rows_by_id[concept_id].get("m1_extension") is False}
    coverage = fewshot_metadata["main_test_cloze_coverage"]
    if len(main_rows) != coverage["n_test"] or len(listed_main_cloze) != coverage["n_available"]:
        raise ValueError("Step-12 main-test cloze coverage does not match the structured release parquet")
    main_by_stratum: dict[str, dict[str, int]] = {}
    for row in main_rows:
        stratum = row["stratum"]
        counts = main_by_stratum.setdefault(stratum, {"n_available": 0, "n_test": 0})
        counts["n_test"] += 1
        counts["n_available"] += row["concept_id"] in cloze_ids
    if main_by_stratum != coverage["by_stratum"]:
        raise ValueError("Step-12 per-stratum cloze coverage does not match the structured release parquet")
    logger.info("Validated step-12 rows=%d, dropflow records=%d, source rows=%d", len(rows), len(dropflow), len(sources))
    logger.info("Validated structured step-12 few-shot sets=%d, cloze status=%s, main-test cloze=%d/%d",
                len(fewshot_metadata["fewshot_sets"]), fewshot_metadata["cloze_status"],
                coverage["n_available"], coverage["n_test"])
    return rows, sources, dropflow, fewshot_metadata


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _pipeline_commit(config_path: Path) -> str:
    """Return the current source commit used to identify the release build."""
    repository = config_path.resolve().parent.parent
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, text=True, capture_output=True)
    return result.stdout.strip().lower()


def _render_qa_sample(rows: list[dict[str, Any]], *, n_per_stratum: int, seed: int) -> str:
    """Render a seeded, stable sample of up to n rows for each etymology stratum."""
    if type(n_per_stratum) is not int or n_per_stratum < 1:
        raise ValueError("release.qa_sample_per_stratum must be a positive integer")
    rng = np.random.default_rng(seed)
    output = ["# Seeded release QA sample", f"# seed={seed}; requested_per_stratum={n_per_stratum}",
              "stratum\tsplit\tconcept_id\ten\tvi\tzh\tfr\tid\tru"]
    for stratum in STRATUM_ORDER:
        members = sorted((row for row in rows if row["stratum"] == stratum), key=lambda row: row["concept_id"])
        if not members:
            continue
        chosen_indexes = sorted(rng.choice(len(members), size=min(n_per_stratum, len(members)), replace=False).tolist())
        for index in chosen_indexes:
            row = members[index]
            values = [stratum, row["split"], row["concept_id"], row["en_lemma"], row["vi_canonical"],
                      row["zh_canonical"], row["fr_canonical"], row["id_canonical"], row.get("ru_canonical") or ""]
            output.append("\t".join(_cell(value).replace("\t", " ").replace("\n", " ") for value in values))
    return "\n".join(output) + "\n"


def run(config_path: str | Path) -> dict[str, Any]:
    """Write release artifacts after concreteness and prompt evidence are complete."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    rows, sources, dropflow, fewshot_metadata = _read_latest_inputs(config, logger)
    model_keys = list(config["models"])
    columns = release_columns(config["langs"], model_keys)
    cloze_ids = set(fewshot_metadata["test_cloze_concept_ids"])
    projected = [project_release_row(row, columns=columns,
                                     fewshot_sets=fewshot_metadata["fewshot_set_by_id"],
                                     cloze_ids=cloze_ids) for row in rows]
    commit_hash = _pipeline_commit(Path(config_path))
    tsv = render_concepts_tsv(projected, columns, commit_hash)
    release_paths = {name: Path(value) for name, value in config["release"]["paths"].items()}
    agreement = build_agreement(
        rows, dropflow=dropflow, sources=sources, config=config, fewshot_metadata=fewshot_metadata,
        fertility_path=release_paths["fertility"], flores_path=release_paths["flores_directions"],
        protocol_path=release_paths["protocol"],
    )
    licenses = build_licenses(sources)
    qa = _render_qa_sample(rows, n_per_stratum=int(config["release"]["qa_sample_per_stratum"]), seed=int(config["seed"]))
    artifacts = ((release_paths["output_tsv"], tsv), (release_paths["agreement"], agreement),
                 (release_paths["licenses"], licenses), (release_paths["qa_sample"], qa))
    for path, contents in artifacts:
        _write_text_atomic(path, contents)
        logger.info("Wrote %s (%d bytes)", path, len(contents.encode("utf-8")))
    runtime = time.monotonic() - started
    logger.info("Released %d concepts; pipeline commit=%s", len(projected), commit_hash)
    logger.info("Runtime seconds: %.3f", runtime)
    return {"rows": len(projected), "commit": commit_hash, "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        logger = logging.getLogger(STEP)
        message = f"Step 14 stopped: {type(exc).__name__}: {exc}"
        if logger.handlers:
            logger.error(message)
        else:
            print(f"ERROR {message}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
