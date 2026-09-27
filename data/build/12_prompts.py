"""Build Russian-sourced few-shot prompts and Vietnamese cloze prompts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wordfreq import zipf_frequency

try:
    from common import DropflowLogger, iter_jsonl, load_config, norm_ru, normalize_nfc, normalize_strings, ru_stress_free, setup_logging, strip_diacritics, vi_orth_key
    from cloze_v13 import build_cloze_v13, record_candidate_dropflow
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, iter_jsonl, load_config, norm_ru, normalize_nfc, normalize_strings, ru_stress_free, setup_logging, strip_diacritics, vi_orth_key
    from data.build.cloze_v13 import build_cloze_v13, record_candidate_dropflow


STEP = "12_prompts"
MODEL_SETTINGS_SOURCE = "nllb"


def select_russian_candidate(
    provisional: str, alternatives: list[str], outputs: list[str],
) -> tuple[str, bool, bool, list[str]]:
    """Choose a stress-free RU candidate by earliest stress/ё-insensitive beam match."""
    candidates = [ru_stress_free(provisional).strip(), *(ru_stress_free(item).strip() for item in alternatives)]
    beam_rank: dict[str, int] = {}
    for rank, output in enumerate(outputs):
        key = norm_ru(output)
        if key and key not in beam_rank:
            beam_rank[key] = rank
    matches = [(beam_rank[norm_ru(candidate)], index, candidate)
               for index, candidate in enumerate(candidates) if norm_ru(candidate) in beam_rank]
    if not matches:
        return candidates[0], False, False, candidates[1:]
    _, chosen_index, chosen = min(matches, key=lambda item: (item[0], item[1]))
    return chosen, True, chosen != candidates[0], [item for i, item in enumerate(candidates) if i != chosen_index]


def fewshot_selection_size(
    eligible_count: int, *, requested_n: int, set_size: int, max_sets: int, minimum_n: int,
) -> tuple[int, int]:
    """Return selected count and set count, or fail below the configured minimum."""
    if eligible_count < minimum_n:
        raise ValueError(f"Only {eligible_count} few-shot concepts are eligible; minimum is {minimum_n}")
    if requested_n != set_size * max_sets or minimum_n % set_size:
        raise ValueError("Few-shot requested/minimum sizes must be whole sets and requested_n = set_size × max_sets")
    set_count = min(max_sets, eligible_count // set_size)
    return set_count * set_size, set_count


def select_cloze_demonstrations(
    selected: list[dict[str, Any]], masked_by_id: dict[str, str | None], *, k: int,
) -> list[dict[str, str]]:
    """Take the first k selected few-shot concepts with valid cloze sentences."""
    available = [row for row in selected if masked_by_id.get(row["concept_id"]) is not None]
    if len(available) < k:
        raise ValueError(f"Only {len(available)} selected few-shot concepts have valid cloze examples; {k} required")
    result = [
        {
            "sentence": mask_canonical_occurrence(masked_by_id[row["concept_id"]], row["vi_canonical"]),
            "answer": row["vi_canonical"],
        }
        for row in available[:k]
    ]
    if any(item["sentence"] is None for item in result):
        raise AssertionError("Selected few-shot cloze example lost its canonical occurrence")
    return result


def _load_step_module(step_file: str) -> Any:
    """Load an adjacent numbered pipeline module for its audited helper functions."""
    module_path = Path(__file__).resolve().parent / step_file
    spec = importlib.util.spec_from_file_location(f"vilens_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_translation_prompt(
    examples: list[dict[str, str]], *, ru_test: str, vi_test: str,
    ru_label: str, vi_label: str, k: int,
) -> str:
    """Render the unquoted Russian-to-Vietnamese translation template."""
    if len(examples) != k:
        raise ValueError(f"Translation prompt requires exactly {k} examples; received {len(examples)}")
    lines = [f"{ru_label}: {item['ru']} - {vi_label}: {item['vi']}" for item in examples]
    lines.append(f"{ru_label}: {ru_test} - {vi_label}:")
    return "\n".join(lines)


def render_repetition_prompt(
    examples: list[dict[str, str]], *, vi_test: str, vi_label: str, k: int,
) -> str:
    """Render the Vietnamese repetition template without trailing label space."""
    if len(examples) != k:
        raise ValueError(f"Repetition prompt requires exactly {k} examples; received {len(examples)}")
    lines = [f"{vi_label}: {item['vi']} - {vi_label}: {item['vi']}" for item in examples]
    lines.append(f"{vi_label}: {vi_test} - {vi_label}:")
    return "\n".join(lines)


def render_cloze_prompt(
    examples: list[dict[str, str]], *, sentence: str, answer_label: str, k: int,
) -> str:
    """Render fixed masked cloze demonstrations followed by the target sentence."""
    if len(examples) != k:
        raise ValueError(f"Cloze prompt requires exactly {k} examples; received {len(examples)}")
    lines: list[str] = []
    for item in examples:
        lines.extend((item["sentence"], f"{answer_label} {item['answer']}"))
    lines.extend((sentence, answer_label))
    return "\n".join(lines)


def nodiac_translation_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Strip Vietnamese example forms while leaving Russian text byte-for-byte intact."""
    return [{**item, "vi": strip_diacritics(item["vi"], preserve_case=True)} for item in examples]


def nodiac_cloze_parts(
    examples: list[dict[str, str]], *, sentence: str, answer_label: str,
) -> tuple[list[dict[str, str]], str, str]:
    """Strip Vietnamese marks from cloze demonstrations, target sentence, answers, and label."""
    stripped_examples = [
        {
            "sentence": strip_diacritics(item["sentence"], preserve_case=True),
            "answer": strip_diacritics(item["answer"], preserve_case=True),
        }
        for item in examples
    ]
    return (
        stripped_examples,
        strip_diacritics(sentence, preserve_case=True),
        strip_diacritics(answer_label, preserve_case=True),
    )


def teacher_forcing_targets(row: dict[str, Any], *, vi_text: str) -> dict[str, str]:
    """Return all five study targets with the required leading-space token."""
    result = {
        "target_vi": vi_text,
        "target_en": row["en_lemma"],
        "target_zh": row["zh_canonical"],
        "target_fr": row["fr_canonical"],
        "target_id": row["id_canonical"],
    }
    for key, value in result.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing target text {key} for {row.get('concept_id')!r}")
        result[key] = " " + normalize_nfc(value).strip()
    return result


def select_balanced_fewshot(
    eligible: list[dict[str, Any]], *, n_final: int, set_count: int, set_size: int,
    seed: int, allowed_strata: list[str], allowed_pos: list[str],
) -> list[dict[str, Any]]:
    """Select concepts by seeded balancing over POS × stratum cells."""
    if n_final != set_count * set_size:
        raise ValueError("fewshot.n_final must equal fewshot.n_sets × fewshot.set_size")
    if len(eligible) < n_final:
        raise ValueError(f"Only {len(eligible)} few-shot concepts are eligible; {n_final} are required")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {
        (pos, stratum): [] for pos in allowed_pos for stratum in allowed_strata
    }
    for row in eligible:
        cell = (row["pos"], row["stratum"])
        if cell not in cells:
            raise ValueError(f"Eligible few-shot concept has unsupported POS/stratum: {cell!r}")
        cells[cell].append(row)

    rng = np.random.default_rng(seed)
    available: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in sorted(cells):
        members = sorted(cells[cell], key=lambda row: row["concept_id"])
        order = rng.permutation(len(members)).tolist()
        available[cell] = [members[index] for index in order]
    target_per_cell = n_final / len(cells)
    selected_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {cell: [] for cell in cells}
    tie_order = {cell: float(rng.random()) for cell in sorted(cells)}
    for _ in range(n_final):
        open_cells = [cell for cell in sorted(cells) if available[cell]]
        if not open_cells:
            raise AssertionError("Few-shot allocation exhausted eligible cells early")
        chosen_cell = min(
            open_cells,
            key=lambda cell: (
                -max(0.0, target_per_cell - len(selected_by_cell[cell])),
                len(selected_by_cell[cell]), tie_order[cell], cell,
            ),
        )
        selected_by_cell[chosen_cell].append(available[chosen_cell].pop())
    selected = [row for cell in sorted(selected_by_cell) for row in selected_by_cell[cell]]
    shuffle_order = rng.permutation(len(selected)).tolist()
    return [selected[index] for index in shuffle_order]


def _punctuation_edges(token: str) -> tuple[str, int, int]:
    """Return token text without leading/trailing Unicode punctuation and its span."""
    start, end = 0, len(token)
    while start < end and unicodedata.category(token[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(token[end - 1]).startswith("P"):
        end -= 1
    return token[start:end], start, end


def mask_canonical_occurrence(sentence: str, canonical: str) -> str | None:
    """Mask the first whole-syllable canonical occurrence using VI spelling equivalence."""
    text = normalize_nfc(sentence)
    target = vi_orth_key(canonical).split()
    if not target:
        raise ValueError("Cannot mask an empty canonical Vietnamese form")
    matches = list(re.finditer(r"\S+", text))
    token_data = [_punctuation_edges(match.group(0)) for match in matches]
    for start in range(0, len(matches) - len(target) + 1):
        window = token_data[start:start + len(target)]
        cores = [item[0] for item in window]
        if any(not core for core in cores) or vi_orth_key(" ".join(cores)).split() != target:
            continue
        first_match, last_match = matches[start], matches[start + len(target) - 1]
        first_offset = window[0][1]
        last_offset = window[-1][2]
        left = first_match.start() + first_offset
        right = first_match.start() + last_offset if start + len(target) == 1 else last_match.start() + last_offset
        return text[:left] + "___" + text[right:]
    return None


def _entry_cloze_sentence(entries: list[dict[str, Any]], canonical: str, *, max_syllables: int) -> str | None:
    """Choose the shortest valid Vietnamese Wiktionary example sentence."""
    candidates: set[str] = set()
    for entry in entries:
        senses = entry.get("senses", [])
        if senses is None:
            senses = []
        if not isinstance(senses, list):
            raise ValueError(f"Unexpected senses on Vietnamese entry {entry.get('word')!r}")
        for sense in senses:
            if not isinstance(sense, dict):
                raise ValueError(f"Malformed Vietnamese sense on {entry.get('word')!r}: {sense!r}")
            examples = sense.get("examples", [])
            if examples is None:
                examples = []
            if not isinstance(examples, list):
                raise ValueError(f"Unexpected examples on Vietnamese sense of {entry.get('word')!r}")
            for example in examples:
                if not isinstance(example, dict):
                    raise ValueError(f"Malformed Wiktionary example on {entry.get('word')!r}: {example!r}")
                text = example.get("text")
                if text is None:
                    continue
                if not isinstance(text, str):
                    raise ValueError(f"Vietnamese example text is not a string on {entry.get('word')!r}: {example!r}")
                normalized = normalize_nfc(text).strip()
                if not normalized or len(normalized.split()) > max_syllables:
                    continue
                masked = mask_canonical_occurrence(normalized, canonical)
                if masked is not None:
                    candidates.add(normalized)
    if not candidates:
        return None
    return min(candidates, key=lambda value: (len(value.split()), len(value), value.casefold(), value))


def _collect_russian_candidates(
    dump_path: Path, concepts: list[dict[str, Any]], *, step03: Any,
    config: dict[str, Any], logger: logging.Logger,
) -> tuple[dict[tuple[str, str, str], list[dict[str, str]]], Counter[str]]:
    """Stream English Wiktextract once and collect cleaned RU items for exact sense keys."""
    target_keys = {
        (normalize_nfc(row["en_lemma"]).strip(), normalize_nfc(row["pos"]).strip(), normalize_nfc(row["sense_gloss"]))
        for row in concepts
    }
    target_pairs = {(lemma, pos) for lemma, pos, _sense in target_keys}
    drop_tags = {normalize_nfc(str(tag)) for tag in config["pool"]["drop_tags"]}
    locations = config["pool"]["translation_locations"]
    found: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    cleaning: Counter[str] = Counter()
    entries_seen = 0
    for entries_seen, entry in enumerate(iter_jsonl(dump_path), start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"English Wiktextract row {entries_seen} is not an object")
        word, pos = entry.get("word"), entry.get("pos")
        if not isinstance(word, str) or not isinstance(pos, str):
            raise ValueError(f"English Wiktextract row {entries_seen} lacks string word/POS")
        lemma = normalize_nfc(word).strip()
        normalized_pos = normalize_nfc(pos).strip()
        if (lemma, normalized_pos) not in target_pairs:
            continue
        relevant: list[tuple[dict[str, Any], str]] = []
        for item, location in step03._translation_lists(entry, locations):
            if not isinstance(item, dict):
                cleaning["non_object_item"] += 1
                continue
            code = item.get("lang_code")
            structural = step03._item_structure_rule(item)
            if structural:
                if code == "ru" or code is None:
                    cleaning[structural] += 1
                continue
            if normalize_nfc(code).strip() == "ru":
                relevant.append((item, location))
        for item in step03.deduplicate_translation_items(relevant):
            cleaned, rule = step03.clean_translation_item(item, drop_tags)
            if rule:
                cleaning[rule] += 1
                continue
            assert cleaned is not None
            key = (lemma, normalized_pos, normalize_nfc(cleaned["sense"]))
            if key not in target_keys:
                continue
            word_ru = ru_stress_free(normalize_nfc(cleaned["word"])).strip()
            if not word_ru:
                raise ValueError(f"Step-03 cleaner returned an empty Russian candidate for {key!r}")
            location = cleaned["source_location"]
            previous = found[key].get(word_ru)
            if previous in {"both", location}:
                continue
            if previous is not None and {previous, location} == {"top", "senses"}:
                found[key][word_ru] = "both"
            else:
                found[key][word_ru] = location
        if entries_seen % int(config["logging"]["progress_every"]) == 0:
            logger.info("Streamed English entries for RU candidates: %d", entries_seen)
    logger.info("Streamed English entries for RU candidates: %d (end of dump)", entries_seen)
    return {
        key: [{"word": word, "source_location": location} for word, location in sorted(values.items())]
        for key, values in sorted(found.items())
    }, cleaning


def _atomic_write_parquet(table: pa.Table, path: Path, row_group_size: int) -> None:
    """Atomically write a deterministic Parquet table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as handle:
            temporary = handle.name
        pq.write_table(table, temporary, compression="zstd", version="2.6", row_group_size=row_group_size, write_statistics=True)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Atomically write stable compact JSONL using UTF-8 and LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    """Write deterministic UTF-8 JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def build_fewshot_metadata(
    fewshot_sets: dict[int, list[dict[str, Any]]], *, cloze_demonstration_ids: list[str],
    test_rows: list[dict[str, Any]], main_test_rows: list[dict[str, Any]],
    masked_by_id: dict[str, str | None], cloze_status: str,
) -> dict[str, Any]:
    """Build the structured release evidence emitted alongside step-12 prompts."""
    if cloze_status not in {"primary", "supplementary"}:
        raise ValueError(f"Unexpected cloze status: {cloze_status!r}")
    set_ids: dict[str, list[str]] = {}
    selected_ids: list[str] = []
    for set_number, group in sorted(fewshot_sets.items()):
        ids = [str(row["concept_id"]) for row in group]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Few-shot set {set_number} has missing or duplicate concept IDs")
        set_ids[str(set_number)] = ids
        selected_ids.extend(ids)
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("A concept occurs in more than one selected few-shot set")
    available_test_ids = sorted(row["concept_id"] for row in test_rows
                                if masked_by_id.get(row["concept_id"]) is not None)
    main_test_counts = Counter(row["stratum"] for row in main_test_rows)
    main_cloze_counts = Counter(row["stratum"] for row in main_test_rows
                                if masked_by_id.get(row["concept_id"]) is not None)
    strata = sorted(set(main_test_counts) | set(main_cloze_counts))
    return {
        "schema_version": 1,
        "fewshot_sets": set_ids,
        "cloze_demonstration_concept_ids": list(cloze_demonstration_ids),
        "test_cloze_concept_ids": available_test_ids,
        "cloze_status": cloze_status,
        "main_test_cloze_coverage": {
            "n_available": len([row for row in main_test_rows
                                 if masked_by_id.get(row["concept_id"]) is not None]),
            "n_test": len(main_test_rows),
            "by_stratum": {
                stratum: {"n_available": main_cloze_counts[stratum], "n_test": main_test_counts[stratum]}
                for stratum in strata
            },
        },
    }


def _remove_step_dropflow(path: Path) -> None:
    """Remove old step-12 dropflow records before a rerun."""
    if not path.exists():
        return
    kept: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), start=1):
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("step"), str):
            raise ValueError(f"Malformed dropflow record at {path}:{line_number}")
        if record["step"] != "12":
            kept.append(line if line.endswith("\n") else line + "\n")
    path.write_text("".join(kept), encoding="utf-8", newline="\n")


def _prompt_record(
    row: dict[str, Any], *, prompt: str, format_name: str, condition: str,
    fewshot_set: int, vi_target: str,
) -> dict[str, Any]:
    """Build a schema-checked prompt JSON record with leading-space targets."""
    return {
        "concept_id": row["concept_id"], "split": row["split"],
        "m1_extension": bool(row["m1_extension"]), "format": format_name,
        "condition": condition, "fewshot_set": fewshot_set, "prompt": prompt,
        **teacher_forcing_targets(row, vi_text=vi_target),
    }


def _assert_no_test_fewshot_overlap(target_rows: list[dict[str, Any]], fewshot_rows: list[dict[str, Any]]) -> None:
    """Stop if a TEST concept has leaked into the fixed few-shot demonstration pool."""
    fewshot_ids = {row["concept_id"] for row in fewshot_rows}
    leaked = sorted(row["concept_id"] for row in target_rows if row["split"] == "test" and row["concept_id"] in fewshot_ids)
    if leaked:
        raise AssertionError(f"TEST concepts occur among few-shot examples: {leaked[:10]!r}")


def run(config_path: str | Path, *, cloze_only: bool = False) -> dict[str, Any]:
    """Collect RU translations, select fixed demonstrations, and write test prompts."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["prompts"]
    paths = {key: Path(value) for key, value in settings["paths"].items()}
    for key in ("english_dump", "main_step11", "extension"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Step 12 required input does not exist: {paths[key]}")
    main_table, ext_table = pq.read_table(paths["main_step11"]), pq.read_table(paths["extension"])
    if main_table.schema.names != ext_table.schema.names or main_table.schema.types != ext_table.schema.types:
        raise ValueError("Step-11 main and M1 extension schemas differ")
    concepts = [normalize_strings(row) for row in main_table.to_pylist()]
    extensions = [normalize_strings(row) for row in ext_table.to_pylist()]
    if any(row.get("m1_extension") is not False for row in concepts):
        raise ValueError("Main step-11 table contains a row marked m1_extension")
    if any(row.get("m1_extension") is not True for row in extensions):
        raise ValueError("M1 extension table contains a row not marked m1_extension")
    all_rows = sorted([*concepts, *extensions], key=lambda row: row["concept_id"])
    all_ids = [row["concept_id"] for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate concept_id across main and M1 extension inputs")
    test_rows = [row for row in all_rows if row["split"] in settings["target_splits"]]
    if any(type(row.get("m1_extension")) is not bool for row in test_rows):
        raise ValueError("Step-11 test rows must have boolean m1_extension values")
    main_test_rows = [row for row in test_rows if row["m1_extension"] is False]
    fewshot_pool = [row for row in concepts if row["split"] == "fewshot_reservoir"]
    _assert_no_test_fewshot_overlap(test_rows, fewshot_pool)
    if not test_rows:
        raise ValueError(f"No prompt targets in configured target_splits={settings['target_splits']!r}")

    step03 = _load_step_module("03_pool.py")
    candidates_by_group, cleaning_counts = _collect_russian_candidates(
        paths["english_dump"], all_rows, step03=step03, config=config, logger=logger,
    )
    logger.info("RU item cleaning counts: %s", json.dumps(dict(sorted(cleaning_counts.items())), sort_keys=True))
    candidate_rows = [row for row in all_rows if candidates_by_group.get(
        (normalize_nfc(row["en_lemma"]).strip(), normalize_nfc(row["pos"]).strip(), normalize_nfc(row["sense_gloss"])),
    )]
    logger.info("Concepts with one or more cleaned RU candidates: %d/%d", len(candidate_rows), len(all_rows))
    if not candidate_rows:
        raise ValueError("No RU candidates were found for any concept; refusing to construct prompts")

    model_settings = {
        **config[MODEL_SETTINGS_SOURCE], "progress_every": int(config["logging"]["progress_every"]),
    }
    model_settings["cache_dir"] = config[MODEL_SETTINGS_SOURCE]["cache_dir"]
    step07 = _load_step_module("07_backtranslate.py")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = step07.choose_device()
    logger.info("Russian NLLB canonicalization device=%s; pinned=%s@%s; offline cache-only", device,
                model_settings["model"], model_settings["revision"])
    translator = step07.NLLBTranslator(model_settings, device, logger)
    source_code = config["nllb"]["lang_codes"]["en"]
    target_code = settings["translation"]["nllb_target_code"]
    lemmas = sorted({normalize_nfc(row["en_lemma"]).strip() for row in candidate_rows})
    requests = [(source_code, target_code, lemma) for lemma in lemmas]
    outputs = translator.translate_many(requests)
    nllb_outputs = {lemma: outputs[(source_code, target_code, lemma)] for lemma in lemmas}
    candidate_identity = {row["concept_id"]: candidates_by_group[
        (normalize_nfc(row["en_lemma"]).strip(), normalize_nfc(row["pos"]).strip(), normalize_nfc(row["sense_gloss"]))
    ] for row in candidate_rows}
    for row in all_rows:
        words = candidate_identity.get(row["concept_id"], [])
        row["ru_cands"] = words
        row["ru_candidate_words"] = [item["word"] for item in words]
        if not words:
            row["ru_canonical"] = None
            row["ru_alts"] = []
            row["ru_nllb_agree"] = None
            row["ru_outs"] = None
            continue
        ranked = sorted(words, key=lambda item: (
            -float(zipf_frequency(item["word"], "ru")), len(item["word"]),
            item["word"].casefold(), item["word"],
        ))
        provisional, alternatives = ranked[0]["word"], [item["word"] for item in ranked[1:]]
        canonical, agrees, _, alternatives = select_russian_candidate(
            provisional, alternatives, nllb_outputs[normalize_nfc(row["en_lemma"]).strip()],
        )
        row["ru_canonical"] = canonical
        row["ru_alts"] = alternatives
        row["ru_nllb_agree"] = agrees
        row["ru_outs"] = nllb_outputs[normalize_nfc(row["en_lemma"]).strip()]

    ru_candidate_ids = {row["concept_id"] for row in candidate_rows}
    test_candidate_rows = [row for row in test_rows if row["concept_id"] in ru_candidate_ids]
    test_ru_agree_rows = [row for row in test_candidate_rows if row["ru_nllb_agree"] is True]
    logger.info("TEST RU coverage: candidates=%d/%d (%.6f)", len(test_candidate_rows), len(test_rows), len(test_candidate_rows) / len(test_rows))
    logger.info("TEST RU NLLB agreement: %d/%d candidate concepts (%.6f); share of all TEST=%d/%d (%.6f)",
                len(test_ru_agree_rows), len(test_candidate_rows), len(test_ru_agree_rows) / len(test_candidate_rows) if test_candidate_rows else 0.0,
                len(test_ru_agree_rows), len(test_rows), len(test_ru_agree_rows) / len(test_rows))

    row_group_size = int(config["split"]["parquet_row_group_size"])
    input_schema = main_table.schema
    additions = [
        pa.field("ru_cands", pa.list_(pa.struct([pa.field("word", pa.string()), pa.field("source_location", pa.string())]))),
        pa.field("ru_candidate_words", pa.list_(pa.string())), pa.field("ru_canonical", pa.string()),
        pa.field("ru_alts", pa.list_(pa.string())), pa.field("ru_nllb_agree", pa.bool_()),
        pa.field("ru_outs", pa.list_(pa.string())),
    ]
    name_collisions = sorted(set(field.name for field in additions).intersection(input_schema.names))
    if name_collisions:
        raise ValueError(f"Step-11 input unexpectedly has RU candidate columns: {name_collisions!r}")

    fewshot_settings = settings["fewshot"]
    requested_n = int(fewshot_settings["n_final"])
    max_sets = int(fewshot_settings["n_sets"])
    set_size = int(fewshot_settings["set_size"])
    minimum_n = int(fewshot_settings["minimum_eligible_n"])
    reservoir_rows = [row for row in all_rows if row["split"] == "fewshot_reservoir"]
    eligible_fewshot = [row for row in reservoir_rows
                        if row["ru_canonical"] is not None and row["ru_nllb_agree"] is True
                        and row.get("bt_pass") is True]
    criterion_counts = {
        "reservoir": len(reservoir_rows),
        "ru_candidate": sum(row["ru_canonical"] is not None for row in reservoir_rows),
        "ru_nllb_agree": sum(row["ru_nllb_agree"] is True for row in reservoir_rows),
        "bt_pass_any_route": sum(row.get("bt_pass") is True for row in reservoir_rows),
        "all_criteria": len(eligible_fewshot),
    }
    logger.info("Few-shot eligibility criteria counts: %s (target=%d; minimum=%d)", json.dumps(criterion_counts, sort_keys=True), requested_n, minimum_n)
    logger.info("Few-shot eligible strata: %s", json.dumps(dict(sorted(Counter(row["stratum"] for row in eligible_fewshot).items())), sort_keys=True))
    n_final, set_count = fewshot_selection_size(
        len(eligible_fewshot), requested_n=requested_n, set_size=set_size,
        max_sets=max_sets, minimum_n=minimum_n,
    )
    if n_final < requested_n:
        logger.info("Few-shot fallback: selecting %d from %d eligible concepts as %d sets of %d",
                    n_final, len(eligible_fewshot), set_count, set_size)
    selected = select_balanced_fewshot(
        eligible_fewshot, n_final=n_final, set_count=set_count, set_size=set_size,
        seed=int(config["seed"]), allowed_strata=sorted({row["stratum"] for row in eligible_fewshot}),
        allowed_pos=["adj", "noun", "verb"],
    )
    fewshot_sets = {
        set_number: selected[(set_number - 1) * set_size:set_number * set_size]
        for set_number in range(1, set_count + 1)
    }
    if any(len(group) != set_size for group in fewshot_sets.values()):
        raise AssertionError("Selected few-shot sets do not have configured fixed size")
    selected_ids = {row["concept_id"] for row in selected}
    if selected_ids.intersection(row["concept_id"] for row in test_rows):
        raise AssertionError("A TEST concept was selected for few-shot examples")
    logger.info("Selected few-shot set allocation (set | POS/stratum counts):")
    for set_number, group in fewshot_sets.items():
        counts = Counter((row["pos"], row["stratum"]) for row in group)
        logger.info("%d | %s", set_number, json.dumps({f"{pos}/{stratum}": count for (pos, stratum), count in sorted(counts.items())}, sort_keys=True))
    logger.info("Few-shot list: set | en | vi | ru | stratum")
    for set_number, group in fewshot_sets.items():
        for row in group:
            logger.info("%d | %s | %s | %s | %s", set_number, row["en_lemma"], row["vi_canonical"], row["ru_canonical"], row["stratum"])

    # Rebuild v1.3 cloze candidates under the ordered C1–C8 rules.
    cloze_k = int(settings["cloze"]["k"])
    cloze_v13_settings = settings["cloze_v13"]
    if cloze_k != int(cloze_v13_settings["demo_count"]):
        raise ValueError(
            f"prompts.cloze.k={cloze_k} must match prompts.cloze_v13.demo_count="
            f"{cloze_v13_settings['demo_count']}"
        )
    step08 = _load_step_module("08_etymology.py")
    step04 = _load_step_module("04_attest.py")
    translator.max_new_tokens = int(cloze_v13_settings["nllb_max_new_tokens"])
    logger.info("Cloze C7 NLLB max_new_tokens=%d", translator.max_new_tokens)
    cloze_result = build_cloze_v13(
        all_rows=all_rows,
        test_rows=test_rows,
        fewshot_sets=fewshot_sets,
        config=config,
        translator=translator,
        step07=step07,
        step08=step08,
        attest_matcher=step04.contains_whole_word,
        logger=logger,
    )
    masked_by_id = cloze_result["masked_by_id"]
    test_cloze_rows = cloze_result["test_cloze_rows"]
    test_cloze_nodiac_rows = cloze_result["test_nodiac_rows"]
    main_cloze_rows = [row for row in main_test_rows if masked_by_id.get(row["concept_id"]) is not None]
    demo_concept_ids = cloze_result["demo_concept_ids"]
    if cloze_result["stop_reason"]:
        if cloze_result["stop_reason"]:
            logger.error(
                "Cloze v1.3 cannot render prompts: stop_reason=%s; selected %d/%d demonstrations; leaving prompt files untouched",
                cloze_result["stop_reason"],
                cloze_result["report"]["demo_count_available"], cloze_result["report"]["demo_count_required"],
            )
        runtime = time.monotonic() - started
        logger.info("Runtime seconds: %.3f", runtime)
        return {
            "status": cloze_result["stop_reason"],
            "main_survivors": cloze_result["main_survivors"],
            "demos_available": cloze_result["report"]["demo_count_available"],
            "demos_required": cloze_result["report"]["demo_count_required"],
            "cloze_report_path": cloze_v13_settings["paths"]["audit_report"],
            "runtime_seconds": runtime,
        }

    cloze_output_dir = Path(cloze_v13_settings["paths"]["output_dir"])
    if cloze_only:
        cloze_output_dir.mkdir(parents=True, exist_ok=True)
        for condition in ("diac", "nodiac"):
            filename = f"cloze_{condition}_set{int(fewshot_settings['primary_set'])}.jsonl"
            _atomic_write_jsonl(cloze_result["records"][condition], cloze_output_dir / filename)
            logger.info("Prompt output %s | records=%d", cloze_output_dir / filename, len(cloze_result["records"][condition]))
        runtime = time.monotonic() - started
        logger.info("Cloze-only build completed without rewriting other prompt or data outputs")
        logger.info("Runtime seconds: %.3f", runtime)
        return {
            "status": "complete",
            "main_survivors": cloze_result["main_survivors"],
            "extension_survivors": cloze_result["report"]["final_counts"]["extension"],
            "file_counts": {
                f"cloze_{condition}_set{int(fewshot_settings['primary_set'])}.jsonl": len(cloze_result["records"][condition])
                for condition in ("diac", "nodiac")
            },
            "cloze_report_path": cloze_v13_settings["paths"]["audit_report"],
            "runtime_seconds": runtime,
        }

    labels = settings["label"]
    translation_k = int(settings["translation"]["k"])
    repetition_k = int(settings["repetition"]["k"])
    primary_set = int(fewshot_settings["primary_set"])
    output_dir = paths["output_dir"]
    file_rows: dict[str, list[dict[str, Any]]] = {}
    sample_prompts: dict[tuple[str, str], str] = {}
    test_translation_rows = [row for row in test_rows if row["ru_canonical"] is not None]
    test_cloze_rows = [row for row in test_rows if masked_by_id[row["concept_id"]] is not None]
    test_cloze_nodiac_rows = [row for row in test_cloze_rows if row["collapsed"] is False]
    main_cloze_rows = [row for row in main_test_rows if masked_by_id[row["concept_id"]] is not None]

    for set_number, fewshot_group in fewshot_sets.items():
        examples = [{"ru": row["ru_canonical"], "vi": row["vi_canonical"]} for row in fewshot_group]
        for condition in ("diac", "nodiac"):
            vi_label = labels["vi"] if condition == "diac" else strip_diacritics(labels["vi"], preserve_case=True)
            condition_examples = examples if condition == "diac" else nodiac_translation_examples(examples)
            records: list[dict[str, Any]] = []
            for row in test_rows:
                if condition == "nodiac" and row["collapsed"]:
                    continue
                vi_text = row["vi_canonical"] if condition == "diac" else strip_diacritics(row["vi_canonical"], preserve_case=True)
                prompt = render_repetition_prompt(
                    [{"vi": item["vi"]} for item in condition_examples], vi_test=vi_text,
                    vi_label=vi_label, k=repetition_k,
                )
                records.append(_prompt_record(row, prompt=prompt, format_name="repetition", condition=condition,
                                              fewshot_set=set_number, vi_target=vi_text))
            file_rows[f"repetition_{condition}_set{set_number}.jsonl"] = records
            if records:
                sample_prompts[("repetition", condition)] = records[0]["prompt"]

            translation_records: list[dict[str, Any]] = []
            for row in test_translation_rows:
                if condition == "nodiac" and row["collapsed"]:
                    continue
                vi_text = row["vi_canonical"] if condition == "diac" else strip_diacritics(row["vi_canonical"], preserve_case=True)
                prompt = render_translation_prompt(
                    condition_examples, ru_test=row["ru_canonical"], vi_test=vi_text,
                    ru_label=labels["ru"], vi_label=vi_label, k=translation_k,
                )
                translation_records.append(_prompt_record(row, prompt=prompt, format_name="translation", condition=condition,
                                                          fewshot_set=set_number, vi_target=vi_text))
            file_rows[f"translation_{condition}_set{set_number}.jsonl"] = translation_records
            if translation_records:
                sample_prompts[("translation", condition)] = translation_records[0]["prompt"]

    for condition in ("diac", "nodiac"):
        records = cloze_result["records"][condition]
        file_rows[f"cloze_{condition}_set{primary_set}.jsonl"] = records
        if records:
            sample_prompts[("cloze", condition)] = records[0]["prompt"]

    # Write enriched intermediate RU data and deterministic prompt files only after all
    # mandatory few-shot/cloze prerequisites have passed.
    ru_schema = pa.schema([*input_schema, *additions])
    ru_rows = all_rows
    ru_table = pa.Table.from_pylist(ru_rows, schema=ru_schema)
    _atomic_write_parquet(ru_table, paths["ru_output"], row_group_size)
    for filename, records in sorted(file_rows.items()):
        destination_dir = cloze_output_dir if filename.startswith("cloze_") else output_dir
        _atomic_write_jsonl(records, destination_dir / filename)
        logger.info("Prompt output %s | records=%d", destination_dir / filename, len(records))

    logger.info("Rendered prompt examples (first deterministic TEST row per format/condition):")
    for key in sorted(sample_prompts):
        logger.info("--- %s/%s ---\n%s", key[0], key[1], sample_prompts[key])
    logger.info("Few-shot final selection: %d examples, %d sets × %d; primary set=%d", n_final, set_count, set_size, primary_set)
    balance_counts = Counter((row["pos"], row["stratum"]) for row in selected)
    logger.info("Few-shot selection balancing counts: %s", json.dumps(
        {f"{pos}/{stratum}": count for (pos, stratum), count in sorted(balance_counts.items())}, sort_keys=True,
    ))
    main_stratum_counts = Counter(row["stratum"] for row in main_test_rows)
    main_cloze_counts = Counter(row["stratum"] for row in main_cloze_rows)
    logger.info("MAIN TEST cloze coverage by stratum:")
    for stratum in sorted(main_stratum_counts):
        logger.info("%s | %d/%d (%.6f)", stratum, main_cloze_counts[stratum], main_stratum_counts[stratum], main_cloze_counts[stratum] / main_stratum_counts[stratum])
    threshold = int(settings["cloze"]["min_test_concepts_primary"])
    cloze_role = "primary" if len(main_cloze_rows) >= threshold else "supplementary"
    configured_cloze_role = settings["cloze"].get("cloze_status")
    if configured_cloze_role != cloze_role:
        raise ValueError(
            "Configured prompts.cloze.cloze_status does not match the main-test result: "
            f"configured={configured_cloze_role!r}, computed={cloze_role!r}"
        )
    metadata = build_fewshot_metadata(
        fewshot_sets, cloze_demonstration_ids=demo_concept_ids, test_rows=test_rows,
        main_test_rows=main_test_rows, masked_by_id=masked_by_id, cloze_status=cloze_role,
    )
    _atomic_write_json(metadata, paths["fewshot_metadata"])
    logger.info("Wrote structured few-shot/cloze metadata: %s", paths["fewshot_metadata"])
    logger.info("Cloze coverage all TEST=%d; MAIN TEST=%d/%d; threshold=%d; cloze_status=%s",
                len(test_cloze_rows), len(main_cloze_rows), len(main_test_rows), threshold, cloze_role)
    logger.info("NLLB Russian calls=%d; cache hits=%d", translator.translation_calls, translator.cache_hits)

    flow_path = Path(config["paths"]["dropflow"])
    _remove_step_dropflow(flow_path)
    flow = DropflowLogger(flow_path)
    flow_stages = [
        ("translation_ru_available_test", len(test_rows), len(test_translation_rows), test_translation_rows),
        ("cloze_sentence_available_test", len(test_rows), len(test_cloze_rows), test_cloze_rows),
        ("cloze_sentence_available_main_test", len(main_test_rows), len(main_cloze_rows), main_cloze_rows),
        ("cloze_nodiac_noncollapsed_test", len(test_cloze_rows), len(test_cloze_nodiac_rows), test_cloze_nodiac_rows),
    ]
    for stage, n_in, n_out, rows in flow_stages:
        by_pos = Counter(row["pos"] for row in rows)
        flow.record(step="12", stage=stage, unit="concepts", n_in=n_in, n_out=n_out,
                    n_out_by_pos=dict(sorted(by_pos.items())))
    record_candidate_dropflow(flow, cloze_result)

    runtime = time.monotonic() - started
    logger.info("Output RU intermediate: %s (%d concepts)", paths["ru_output"], len(ru_rows))
    logger.info("Runtime seconds: %.3f", runtime)
    return {"ru_coverage_test": (len(test_candidate_rows), len(test_rows)),
            "ru_agreement_test": (len(test_ru_agree_rows), len(test_candidate_rows)),
            "fewshot": selected, "file_counts": {name: len(rows) for name, rows in file_rows.items()},
            "cloze_test_n": len(test_cloze_rows), "cloze_main_test_n": len(main_cloze_rows),
            "cloze_role": cloze_role, "metadata_path": str(paths["fewshot_metadata"]),
            "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--cloze-only", action="store_true", help="Rebuild cloze files without rewriting other prompt/data outputs")
    args = parser.parse_args()
    try:
        run(args.config, cloze_only=args.cloze_only)
    except Exception as exc:
        logger = logging.getLogger(STEP)
        if logger.handlers:
            logger.exception("Prompt generation stopped: %s", exc)
        else:
            print(f"ERROR Prompt generation stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
