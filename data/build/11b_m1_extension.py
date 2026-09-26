"""Build a separate, filtered M1-only extension without changing the main pool."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    from common import DropflowLogger, load_config, normalize_nfc, normalize_strings, setup_logging, vi_orth_key
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, load_config, normalize_nfc, normalize_strings, setup_logging, vi_orth_key


STEP = "11b_m1_extension"
MODEL_KEYS = ("gemma", "qwen", "llama")
SPLITS = ("fewshot_reservoir", "directions", "test")


def extension_eligible_flags(row: dict[str, Any], *, extension: bool) -> dict[str, Any]:
    """Return extension membership and test-only single-token eligibility flags."""
    flags: dict[str, Any] = {"m1_extension": extension}
    is_test = row.get("split") == "test"
    for model in MODEL_KEYS:
        source = row.get(f"single_token_vi_{model}")
        if type(source) is not bool:
            raise ValueError(f"Missing single_token_vi_{model} for {row.get('concept_id')!r}")
        flags[f"m1_eligible_{model}"] = source if is_test else None
    return flags


def assert_extension_disjoint(main_rows: list[dict[str, Any]], extension_rows: list[dict[str, Any]]) -> None:
    """Assert extension identifiers, VI orthographic keys, and English lemmas are disjoint."""
    main_ids = {row["concept_id"] for row in main_rows}
    main_vi = {vi_orth_key(row["vi_canonical"]) for row in main_rows}
    main_en = {normalize_nfc(row["en_lemma"]).casefold() for row in main_rows}
    seen_ids: set[str] = set()
    seen_vi: set[str] = set()
    seen_en: set[str] = set()
    for row in extension_rows:
        concept_id = row["concept_id"]
        vi_key = vi_orth_key(row["vi_canonical"])
        en_key = normalize_nfc(row["en_lemma"]).casefold()
        if concept_id in main_ids or concept_id in seen_ids:
            raise AssertionError(f"M1 extension concept_id collision: {concept_id!r}")
        if vi_key in main_vi or vi_key in seen_vi:
            raise AssertionError(f"M1 extension Vietnamese spelling collision: {vi_key!r}")
        if en_key in main_en or en_key in seen_en:
            raise AssertionError(f"M1 extension English lemma collision: {en_key!r}")
        seen_ids.add(concept_id)
        seen_vi.add(vi_key)
        seen_en.add(en_key)


def _write_config(config: dict[str, Any], path: Path) -> None:
    """Write a temporary complete YAML config for an existing pipeline stage."""
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path, row_group_size: int) -> None:
    """Atomically write a deterministic Zstandard Parquet table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as handle:
            temporary = handle.name
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd", version="2.6",
                       row_group_size=row_group_size, write_statistics=True)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _replace_step_dropflow(path: Path, step: str) -> None:
    """Remove prior records for this step while preserving all other dropflow history."""
    if not path.exists():
        return
    retained: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), start=1):
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("step"), str):
            raise ValueError(f"Malformed dropflow record at {path}:{line_number}")
        if record["step"] != step:
            retained.append(line if line.endswith("\n") else line + "\n")
    path.write_text("".join(retained), encoding="utf-8", newline="\n")


def _temp_config(config: dict[str, Any], root: Path, *, stage: str, input_path: Path, output_path: Path) -> Path:
    """Clone the config with isolated stage paths and logs under a temporary directory."""
    cloned = yaml.safe_load(yaml.safe_dump(config, allow_unicode=True))
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cloned["paths"]["logs"] = str(logs)
    cloned["paths"]["dropflow"] = str(root / "dropflow.jsonl")
    if stage == "06":
        cloned["filter"]["paths"]["input"] = str(input_path)
        cloned["filter"]["paths"]["output"] = str(output_path)
        cloned["filter"]["paths"]["loanword_audit"] = str(root / "loanword_audit.parquet")
    elif stage == "08":
        cloned["etymology"]["paths"]["input"] = str(input_path)
        cloned["etymology"]["paths"]["output"] = str(output_path)
    elif stage == "09":
        cloned["covariates"]["paths"]["input"] = str(input_path)
        cloned["covariates"]["paths"]["output"] = str(output_path)
    elif stage == "10":
        cloned["tokens"]["paths"]["input"] = str(input_path)
        cloned["tokens"]["paths"]["output"] = str(output_path)
    elif stage == "11":
        cloned["diacritics"]["paths"]["input"] = str(input_path)
        cloned["diacritics"]["paths"]["output"] = str(output_path)
    else:
        raise ValueError(f"Unsupported temporary pipeline stage: {stage!r}")
    result = root / f"config_{stage}.yaml"
    _write_config(cloned, result)
    return result


def _load_step_module(step_file: str) -> Any:
    """Load a numbered pipeline script as a module for reuse by the extension run."""
    module_path = Path(__file__).resolve().parent / step_file
    name = f"vilens_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline step module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path: str | Path) -> dict[str, Any]:
    """Build extension candidates, run stages 07–11, and mark the main-test union."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    settings = config["m1_extension"]
    paths = settings["paths"]
    target_senses = settings["target_n_senses_vi"]
    target_syllables = settings["target_syllables"]
    diagnostic_percentile = settings["candidate_polysemy_percentile"]
    if type(target_senses) is not int or target_senses < 1 or type(target_syllables) is not int or target_syllables < 1:
        raise ValueError("m1_extension target senses/syllables must be positive integers")
    if not 0 <= float(diagnostic_percentile) <= 100:
        raise ValueError("m1_extension.candidate_polysemy_percentile must be within [0, 100]")

    main_paths = {name: Path(paths[name]) for name in ("step05", "main_step06", "main_step07", "main_step11", "output")}
    for name, path in main_paths.items():
        if name != "output" and not path.is_file():
            raise FileNotFoundError(f"M1 extension requires {name}: {path}")
    main05 = pq.read_table(main_paths["step05"]).to_pylist()
    main06 = [normalize_strings(row) for row in pq.read_table(main_paths["main_step06"]).to_pylist()]
    main07_table = pq.read_table(main_paths["main_step07"])
    main07 = [normalize_strings(row) for row in main07_table.to_pylist()]
    main11_table = pq.read_table(main_paths["main_step11"])
    main11 = [normalize_strings(row) for row in main11_table.to_pylist()]
    if not main05 or not main06 or not main07 or not main11:
        raise ValueError("One or more required upstream parquet files are empty")
    cutoffs: dict[str, float] = {}
    for row in main06:
        pos, cutoff = row["pos"], row["polysemy_cutoff"]
        if not isinstance(cutoff, (int, float)):
            raise ValueError(f"Missing frozen step-06 cutoff for {pos!r}")
        if pos in cutoffs and cutoffs[pos] != float(cutoff):
            raise ValueError(f"Inconsistent frozen step-06 cutoffs for POS {pos!r}")
        cutoffs[pos] = float(cutoff)
    original06_ids = {row["concept_id"] for row in main06}

    step06 = _load_step_module("06_filter.py")
    step07 = _load_step_module("07_backtranslate.py")
    step08 = _load_step_module("08_etymology.py")
    step09 = _load_step_module("09_covariates.py")
    step10 = _load_step_module("10_tokens.py")
    step11 = _load_step_module("11_diacritics.py")

    with tempfile.TemporaryDirectory(prefix="vilens-m1-extension-") as temporary_dir:
        root = Path(temporary_dir)
        config06 = yaml.safe_load(yaml.safe_dump(config, allow_unicode=True))
        config06["filters"]["polysemy_percentile"] = diagnostic_percentile
        config06["paths"]["logs"] = str(root / "logs")
        config06["paths"]["dropflow"] = str(root / "dropflow.jsonl")
        (root / "logs").mkdir()
        config06["filter"]["paths"]["input"] = str(main_paths["step05"])
        config06["filter"]["paths"]["output"] = str(root / "06_relaxed.parquet")
        config06["filter"]["paths"]["loanword_audit"] = str(root / "06_loan_audit.parquet")
        path06 = root / "config_06.yaml"
        _write_config(config06, path06)
        logger.info("Candidate discovery uses step 06 with a temporary polysemy percentile %s; main config/output are unchanged", diagnostic_percentile)
        step06.run(path06)
        candidate_rows = [normalize_strings(row) for row in pq.read_table(config06["filter"]["paths"]["output"]).to_pylist()]
        candidates = []
        for row in candidate_rows:
            if row["split"] != "test" or row["concept_id"] in original06_ids:
                continue
            if len(vi_orth_key(row["vi_canonical"]).split()) != target_syllables:
                continue
            if int(row["n_senses_vi"]) != target_senses:
                continue
            if row["pos"] not in cutoffs:
                raise ValueError(f"No frozen step-06 cutoff for extension POS {row['pos']!r}")
            if int(row["n_senses_vi"]) <= cutoffs[row["pos"]]:
                continue
            row["polysemy_cutoff"] = cutoffs[row["pos"]]
            candidates.append(row)
        candidates.sort(key=lambda row: row["concept_id"])
        if not candidates:
            raise ValueError("No extension candidates remain after selecting monosyllabic exact-2-sense concepts dropped only by filter3")
        logger.info("Step-06 extension candidates (pre-filter-2): %d", len(candidates))

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        nllb = {
            **config["nllb"], "progress_every": int(config["logging"]["progress_every"]),
            "bt": config["backtranslate"]["bt"],
        }
        device = step07.choose_device()
        logger.info("M1 extension NLLB device=%s; offline cache-only inference", device)
        translator = step07.NLLBTranslator(nllb, device, logger)
        processed, rows_by_split, info = step07.process_rows(candidates, translator=translator, config=nllb, logger=logger)
        postpromotion, post_flow, _ = step07.post_promotion_filters(rows_by_split, config=config)
        after_bt = [row for row in processed if row["bt_pass"]]
        after_post = postpromotion["test"]
        logger.info("Extension filter 2: %d/%d pass", len(after_bt), len(candidates))
        logger.info("Extension post-promotion filters: %d/%d remain", len(after_post), len(after_bt))

        main_keys = {"ids": {row["concept_id"] for row in main11}}
        main_vi = {vi_orth_key(row["vi_canonical"]) for row in main11}
        main_en = {normalize_nfc(row["en_lemma"]).casefold() for row in main11}
        disjoint = [row for row in after_post if row["concept_id"] not in main_keys["ids"]
                    and vi_orth_key(row["vi_canonical"]) not in main_vi
                    and normalize_nfc(row["en_lemma"]).casefold() not in main_en]
        logger.info("Extension main-set disjointness: %d/%d remain", len(disjoint), len(after_post))
        assert_extension_disjoint(main11, disjoint)
        if not disjoint:
            raise ValueError("All post-filter M1 extension concepts collide with the main set")

        # Step 08 requires the complete declared split vocabulary. Add one already-surviving
        # representative from each non-test split to the isolated temporary input only.
        representatives = []
        for split in ("fewshot_reservoir", "directions"):
            choices = sorted((row for row in main07 if row["split"] == split), key=lambda row: row["concept_id"])
            if not choices:
                raise ValueError(f"No existing {split} representative is available for isolated step-08 execution")
            representatives.append(choices[0])
        extension_ids = {row["concept_id"] for row in disjoint}
        temp08_input = root / "07_extension_plus_split_representatives.parquet"
        pq.write_table(
            pa.Table.from_pylist([*disjoint, *representatives], schema=main07_table.schema), temp08_input,
            compression="zstd", version="2.6", row_group_size=int(config["split"]["parquet_row_group_size"]),
        )
        temp08_output = root / "08_extension.parquet"
        config08 = _temp_config(config, root, stage="08", input_path=temp08_input, output_path=temp08_output)
        step08.run(config08)
        rows08 = [normalize_strings(row) for row in pq.read_table(temp08_output).to_pylist() if row["concept_id"] in extension_ids]
        if len(rows08) != len(disjoint):
            raise AssertionError("Isolated step 08 changed extension row count unexpectedly")

        temp09_input = root / "08_extension_only.parquet"
        pq.write_table(pa.Table.from_pylist(rows08, schema=pq.read_table(temp08_output).schema), temp09_input,
                       compression="zstd", version="2.6", row_group_size=int(config["split"]["parquet_row_group_size"]))
        temp09_output = root / "09_extension.parquet"
        step09.run(_temp_config(config, root, stage="09", input_path=temp09_input, output_path=temp09_output))
        temp10_output = root / "10_extension.parquet"
        step10.run(_temp_config(config, root, stage="10", input_path=temp09_output, output_path=temp10_output))
        temp11_output = root / "11_extension.parquet"
        step11.run(_temp_config(config, root, stage="11", input_path=temp10_output, output_path=temp11_output))
        ext_table = pq.read_table(temp11_output)
        extension_rows = [normalize_strings(row) for row in ext_table.to_pylist()]
        if len(extension_rows) != len(disjoint):
            raise AssertionError("Steps 08–11 changed extension row count unexpectedly")

    # Use the same final table schema for the extension and main data, with eligibility
    # defined only on main-test and extension rows; non-test main rows remain NA.
    additions = [pa.field("m1_extension", pa.bool_(), nullable=False)]
    additions.extend(pa.field(f"m1_eligible_{model}", pa.bool_(), nullable=True) for model in MODEL_KEYS)
    collisions = sorted(set(field.name for field in additions).intersection(main11_table.schema.names))
    if collisions:
        raise ValueError(f"Main step-11 table already contains M1 extension columns: {collisions!r}")
    final_schema = pa.schema([*main11_table.schema, *additions])
    for row in main11:
        row.update(extension_eligible_flags(row, extension=False))
    for row in extension_rows:
        row.update(extension_eligible_flags(row, extension=True))
    main11.sort(key=lambda row: row["concept_id"])
    extension_rows.sort(key=lambda row: row["concept_id"])
    if ext_table.schema.names != main11_table.schema.names or ext_table.schema.types != main11_table.schema.types:
        raise ValueError("Extension step-11 schema differs from the main 11_diac schema")
    assert_extension_disjoint(main11, extension_rows)
    _write_parquet(extension_rows, final_schema, main_paths["output"], int(config["split"]["parquet_row_group_size"]))
    _write_parquet(main11, final_schema, main_paths["main_step11"], int(config["split"]["parquet_row_group_size"]))

    eligible_counts = {model: sum(bool(row[f"m1_eligible_{model}"]) for row in [*main11, *extension_rows]
                                  if row["split"] == "test") for model in MODEL_KEYS}
    extension_strata = Counter(row["stratum"] for row in extension_rows)
    flow_path = Path(config["paths"]["dropflow"])
    _replace_step_dropflow(flow_path, "11b")
    flow = DropflowLogger(flow_path)
    stages = [
        ("extension_candidates", len([row for row in main05 if row["split"] == "test"]), len(candidates), candidates),
        ("filter2_backtranslation", len(candidates), len(after_bt), after_bt),
        ("post_promotion_filters", len(after_bt), len(after_post), after_post),
        ("disjoint_main_set", len(after_post), len(disjoint), disjoint),
        ("extension_enriched_08_11", len(disjoint), len(extension_rows), extension_rows),
    ]
    for stage, n_in, n_out, selected in stages:
        pos_counts = Counter(row["pos"] for row in selected)
        flow.record(step="11b", stage=stage, unit="concepts", n_in=n_in, n_out=n_out,
                    n_out_by_pos=dict(sorted(pos_counts.items())))
        logger.info("dropflow %s | n_in=%d n_out=%d n_out_by_pos=%s", stage, n_in, n_out, dict(sorted(pos_counts.items())))
    logger.info("Extension strata: %s", json.dumps(dict(sorted(extension_strata.items())), sort_keys=True))
    logger.info("M1 eligible counts (main test + extension): %s", json.dumps(eligible_counts, sort_keys=True))
    for model in MODEL_KEYS:
        by_stratum = Counter(row["stratum"] for row in [*main11, *extension_rows]
                             if row["split"] == "test" and row[f"m1_eligible_{model}"])
        logger.info("m1_eligible_%s by stratum: %s", model, json.dumps(dict(sorted(by_stratum.items())), sort_keys=True))
    logger.info("Output: %s (%d extension concepts); main flags updated in %s", main_paths["output"], len(extension_rows), main_paths["main_step11"])
    runtime = time.monotonic() - started
    logger.info("NLLB extension translation calls=%d cache_hits=%d", translator.translation_calls, translator.cache_hits)
    logger.info("Runtime seconds: %.3f", runtime)
    return {"extension_n": len(extension_rows), "eligible_counts": eligible_counts,
            "extension_strata": dict(extension_strata), "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        logger = logging.getLogger(STEP)
        if logger.handlers:
            logger.exception("M1 extension stopped: %s", exc)
        else:
            print(f"ERROR M1 extension stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
