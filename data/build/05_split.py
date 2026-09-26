"""Create leakage-safe few-shot, direction, and test concept partitions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wordfreq import zipf_frequency

try:  # Direct script execution and package-based tests.
    from common import DropflowLogger, load_config, normalize_nfc, normalize_strings, setup_logging
except ModuleNotFoundError:
    from data.build.common import DropflowLogger, load_config, normalize_nfc, normalize_strings, setup_logging


STEP = "05_split"
SPLITS = ("fewshot_reservoir", "directions", "test")


@dataclass(frozen=True)
class Component:
    """A leakage-connected set of concepts and its representative stratum."""

    component_id: str
    members: tuple[dict[str, Any], ...]
    stratum: tuple[str, str]
    mixed_pos: bool
    mixed_syllable: bool


def _term_key(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Concept has missing or malformed {field}: {value!r}")
    return normalize_nfc(value.strip()).casefold()


def syllable_bucket(word: str, edges: list[int]) -> str:
    """Bucket a whitespace-separated Vietnamese form as 1, 2, or 3+."""
    if len(edges) != 2 or any(type(edge) is not int for edge in edges) or edges != sorted(set(edges)) or edges[0] < 1:
        raise ValueError(f"syllable bins must be two ascending positive integers, got {edges!r}")
    count = len(normalize_nfc(word).strip().split())
    if count <= edges[0]:
        return str(edges[0])
    if count <= edges[1]:
        return str(edges[1])
    return f"{edges[1] + 1}+"


def concept_syllable_bucket(row: dict[str, Any], edges: list[int]) -> str:
    """Use the first surviving VI candidate as the pre-canonicalization proxy."""
    candidates = row.get("vi_cands")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Concept {row.get('concept_id')!r} has no VI candidates")
    candidate = candidates[0]
    if not isinstance(candidate, dict) or not isinstance(candidate.get("word"), str):
        raise ValueError(f"Concept {row.get('concept_id')!r} has malformed first VI candidate")
    return syllable_bucket(candidate["word"], edges)


def connected_components(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Return row-index components linked by any VI form or English lemma."""
    n_rows = len(rows)
    parents = list(range(n_rows))
    ranks = [0] * n_rows

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if ranks[root_left] < ranks[root_right]:
            root_left, root_right = root_right, root_left
        parents[root_right] = root_left
        if ranks[root_left] == ranks[root_right]:
            ranks[root_left] += 1

    seen_ids: set[str] = set()
    seen_terms: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        concept_id = _term_key(row.get("concept_id"), field="concept_id")
        if concept_id in seen_ids:
            raise ValueError(f"Duplicate concept_id in split input: {concept_id!r}")
        seen_ids.add(concept_id)
        _term_key(row.get("en_lemma"), field="en_lemma")
        pos = _term_key(row.get("pos"), field="pos")
        candidates = row.get("vi_cands")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Concept {concept_id!r} has missing/empty vi_cands")
        terms = {("en", _term_key(row["en_lemma"], field="en_lemma"))}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"Concept {concept_id!r} contains a non-object VI candidate")
            terms.add(("vi", _term_key(candidate.get("word"), field="vi candidate word")))
        for term in sorted(terms):
            previous = seen_terms.setdefault(term, index)
            union(index, previous)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(n_rows):
        grouped[find(index)].append(index)
    components = list(grouped.values())
    for component in components:
        component.sort(key=lambda idx: _term_key(rows[idx]["concept_id"], field="concept_id"))
    components.sort(key=lambda group: _term_key(rows[group[0]]["concept_id"], field="concept_id"))
    return components


def describe_components(rows: list[dict[str, Any]], edges: list[int]) -> list[Component]:
    """Attach representative POS/syllable strata and mixed-label diagnostics."""
    result: list[Component] = []
    for indexes in connected_components(rows):
        members = tuple(rows[index] for index in indexes)
        first = members[0]
        stratum = (normalize_nfc(first["pos"]), concept_syllable_bucket(first, edges))
        pos_values = {normalize_nfc(member["pos"]) for member in members}
        syllable_values = {concept_syllable_bucket(member, edges) for member in members}
        result.append(Component(
            component_id=normalize_nfc(first["concept_id"]),
            members=members,
            stratum=stratum,
            mixed_pos=len(pos_values) > 1,
            mixed_syllable=len(syllable_values) > 1,
        ))
    return result


def stratified_component_selection(
    candidates: list[Component],
    target_n: int,
    pool_stratum_counts: Counter[tuple[str, str]],
    rng: np.random.Generator,
) -> list[Component]:
    """Seededly select whole components to proportional largest-remainder quotas."""
    if target_n < 0:
        raise ValueError("Split target must be non-negative")
    if target_n == 0 or not candidates:
        return []
    total_pool = sum(pool_stratum_counts.values())
    if total_pool <= 0:
        raise ValueError("Cannot stratify against an empty pool")
    keys = sorted(pool_stratum_counts)
    exact = {key: target_n * pool_stratum_counts[key] / total_pool for key in keys}
    quotas = {key: math.floor(exact[key]) for key in keys}
    remainder = target_n - sum(quotas.values())
    for key in sorted(keys, key=lambda item: (-(exact[item] - quotas[item]), item))[:remainder]:
        quotas[key] += 1

    by_stratum: dict[tuple[str, str], list[Component]] = defaultdict(list)
    for component in candidates:
        by_stratum[component.stratum].append(component)
    ordered: dict[tuple[str, str], list[Component]] = {}
    for key in sorted(by_stratum):
        stable = sorted(by_stratum[key], key=lambda item: item.component_id)
        order = rng.permutation(len(stable)).tolist()
        ordered[key] = [stable[index] for index in order]

    selected: list[Component] = []
    selected_ids: set[str] = set()
    selected_counts: Counter[tuple[str, str]] = Counter()
    cursor: Counter[tuple[str, str]] = Counter()
    for key in keys:
        while selected_counts[key] < quotas[key] and cursor[key] < len(ordered.get(key, [])):
            component = ordered[key][cursor[key]]
            cursor[key] += 1
            selected.append(component)
            selected_ids.add(component.component_id)
            selected_counts[key] += len(component.members)

    # If unavailable strata leave a shortfall, fill it from remaining components
    # in the most under-represented available strata, still using whole components.
    while sum(len(component.members) for component in selected) < target_n:
        available = [
            key for key in sorted(ordered)
            if cursor[key] < len(ordered[key])
        ]
        if not available:
            break
        key = max(
            available,
            key=lambda item: (exact.get(item, 0.0) - selected_counts[item], tuple(-ord(ch) for ch in "|".join(item))),
        )
        component = ordered[key][cursor[key]]
        cursor[key] += 1
        if component.component_id not in selected_ids:
            selected.append(component)
            selected_ids.add(component.component_id)
            selected_counts[key] += len(component.members)
    return sorted(selected, key=lambda item: item.component_id)


def fewshot_component_eligible(
    component: Component,
    zipf_by_concept: dict[str, float],
    zipf_threshold: float,
) -> bool:
    """Return true only when every concept meets all reservoir eligibility rules."""
    for row in component.members:
        candidates = row.get("vi_cands")
        if not isinstance(candidates, list) or len(candidates) != 1:
            return False
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ValueError(f"Concept {row.get('concept_id')!r} has malformed VI candidate")
        sources = candidate.get("n_sources")
        external = candidate.get("external_attested")
        if type(sources) is not int or type(external) is not bool:
            raise ValueError(f"Concept {row.get('concept_id')!r} lacks valid attestation fields")
        if sources < 3 and not external:
            return False
        concept_id = normalize_nfc(row["concept_id"])
        if zipf_by_concept[concept_id] < zipf_threshold:
            return False
    return True


def assert_disjoint_splits(rows: list[dict[str, Any]]) -> None:
    """Assert concept IDs, VI surfaces, and English lemmas occur in one split."""
    owners: dict[tuple[str, str], str] = {}
    seen_ids: set[str] = set()
    for row in rows:
        split_name = row.get("split")
        if split_name not in SPLITS:
            raise ValueError(f"Unknown split label: {split_name!r}")
        concept_id = _term_key(row.get("concept_id"), field="concept_id")
        if concept_id in seen_ids:
            raise ValueError(f"Duplicate concept_id in partitioned rows: {concept_id!r}")
        seen_ids.add(concept_id)
        terms = [("en", _term_key(row.get("en_lemma"), field="en_lemma"))]
        candidates = row.get("vi_cands")
        if not isinstance(candidates, list):
            raise ValueError(f"Concept {concept_id!r} has malformed vi_cands")
        terms.extend(("vi", _term_key(item.get("word"), field="vi candidate word")) for item in candidates)
        for term in terms:
            prior = owners.setdefault(term, split_name)
            if prior != split_name:
                raise ValueError(f"Leakage across splits for {term[0]} term {term[1]!r}: {prior} vs {split_name}")


def _replace_dropflow(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace this step's dropflow records with one record per split."""
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
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as temp:
            temp_name = temp.name
            for record in records:
                temp.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    for split_name in SPLITS:
        split_rows = [row for row in rows if row["split"] == split_name]
        by_pos = dict(sorted(Counter(row["pos"] for row in split_rows).items()))
        DropflowLogger(path).record(
            step=STEP, stage=f"split_{split_name}", unit="concepts",
            n_in=len(rows), n_out=len(split_rows), n_out_by_pos=by_pos,
        )


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path, row_group_size: int) -> None:
    """Atomically write deterministic zstd-compressed parquet output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False) as temp:
            temp_name = temp.name
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), temp_name,
            compression="zstd", version="2.6", row_group_size=row_group_size, write_statistics=True,
        )
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _format_words(values: Any) -> str:
    if not isinstance(values, list):
        raise ValueError(f"Candidate language column must be a list, got {values!r}")
    words: list[str] = []
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("word"), str):
            raise ValueError(f"Malformed language candidate: {item!r}")
        words.append(normalize_nfc(item["word"]))
    return " / ".join(sorted(set(words)))


def run(config_path: str) -> dict[str, Any]:
    """Build component-safe splits, convenience parquet files, and diagnostics."""
    started = time.monotonic()
    config = load_config(config_path)
    logger = setup_logging(STEP, config["paths"]["logs"], level=config.get("logging", {}).get("level", "INFO"))
    progress_every = int(config["logging"]["progress_every"])
    if progress_every < 1:
        raise ValueError("logging.progress_every must be at least one")
    try:
        settings = config["heldout"]
        split_settings = config["split"]
        paths = split_settings["paths"]
        if settings["stratify_by"] != ["pos", "syllable_bucket"]:
            raise ValueError("heldout.stratify_by must be [pos, syllable_bucket]")
        if settings["fewshot_min_en_zipf_tercile"] != "top":
            raise ValueError("heldout.fewshot_min_en_zipf_tercile currently supports only 'top'")
        max_component_size = int(split_settings["component_max_size"])
        bin_edges = split_settings["syllable_bins"]
        if split_settings["syllable_source"] != "first_vi_candidate":
            raise ValueError("split.syllable_source must be 'first_vi_candidate'")
        row_group_size = int(split_settings["parquet_row_group_size"])
        if row_group_size < 1:
            raise ValueError("split.parquet_row_group_size must be positive")
        if max_component_size < 1:
            raise ValueError("split.component_max_size must be positive")
        if not isinstance(bin_edges, list) or len(bin_edges) != 2:
            raise ValueError(f"split.syllable_bins must have two edges, got {bin_edges!r}")

        input_path = Path(paths["input"])
        table = pq.read_table(input_path)
        required = {"concept_id", "en_lemma", "pos", "vi_cands", "zh_cands", "fr_cands", "id_cands"}
        missing = sorted(required - set(table.column_names))
        if missing:
            raise ValueError(f"Attested parquet is missing required columns: {missing}")
        if "split" in table.column_names:
            raise ValueError("Attested input already has a split column")
        rows = [normalize_strings(row) for row in table.to_pylist()]
        if not rows:
            raise ValueError("Attested parquet has no concepts")
        logger.info("Loaded %d attested concepts from %s", len(rows), input_path)
        logger.info("Concept syllable_bucket source: %s", split_settings["syllable_source"])

        concepts_by_id: dict[str, dict[str, Any]] = {}
        zipf_by_concept: dict[str, float] = {}
        lemma_zipf: dict[str, float] = {}
        zipf_values: list[float] = []
        strata_by_id: dict[str, tuple[str, str]] = {}
        pool_stratum_counts: Counter[tuple[str, str]] = Counter()
        for index, row in enumerate(sorted(rows, key=lambda item: _term_key(item.get("concept_id"), field="concept_id")), start=1):
            concept_id = normalize_nfc(row["concept_id"])
            if concept_id in concepts_by_id:
                raise ValueError(f"Duplicate concept_id: {concept_id!r}")
            concepts_by_id[concept_id] = row
            lemma = normalize_nfc(row["en_lemma"]).strip()
            if lemma not in lemma_zipf:
                lemma_zipf[lemma] = float(zipf_frequency(lemma, "en"))
            zipf_value = lemma_zipf[lemma]
            if not math.isfinite(zipf_value):
                raise ValueError(f"wordfreq returned a non-finite Zipf score for {lemma!r}")
            zipf_by_concept[concept_id] = zipf_value
            zipf_values.append(zipf_value)
            stratum = (normalize_nfc(row["pos"]), concept_syllable_bucket(row, bin_edges))
            strata_by_id[concept_id] = stratum
            pool_stratum_counts[stratum] += 1
            if index % progress_every == 0 or index == len(rows):
                logger.info("Prepared concept metadata: %d/%d", index, len(rows))

        # The top tercile starts at the 2/3 quantile of concept-weighted pool Zipf scores.
        zipf_threshold = float(np.quantile(np.asarray(zipf_values, dtype=np.float64), 2.0 / 3.0))
        components = describe_components(rows, bin_edges)
        component_distribution = Counter(
            "1" if len(component.members) == 1 else
            "2" if len(component.members) == 2 else
            "3-5" if len(component.members) <= max_component_size else ">5"
            for component in components
        )
        component_concepts = Counter(
            "1" if len(component.members) == 1 else
            "2" if len(component.members) == 2 else
            "3-5" if len(component.members) <= max_component_size else ">5"
            for component in components
            for _ in component.members
        )
        mixed_components = sum(component.mixed_pos or component.mixed_syllable for component in components)
        logger.info("Connected components=%d; mixed components=%d (mixed POS=%d; mixed syllable=%d)",
                    len(components), mixed_components,
                    sum(component.mixed_pos for component in components),
                    sum(component.mixed_syllable for component in components))
        for size_bin in ("1", "2", "3-5", ">5"):
            logger.info("Component sizes %s: components=%d concepts=%d", size_bin,
                        component_distribution[size_bin], component_concepts[size_bin])

        forced_test = [component for component in components if len(component.members) > max_component_size]
        logger.info("Forced to test due to component size > %d: components=%d concepts=%d",
                    max_component_size, len(forced_test), sum(len(component.members) for component in forced_test))
        eligible_components = [component for component in components if len(component.members) <= max_component_size]
        rng = np.random.default_rng(config["seed"])
        direction_components = stratified_component_selection(
            eligible_components, int(settings["directions_n"]), pool_stratum_counts, rng,
        )
        direction_ids = {component.component_id for component in direction_components}
        remaining_components = [component for component in eligible_components if component.component_id not in direction_ids]
        fewshot_eligible = [
            component for component in remaining_components
            if fewshot_component_eligible(component, zipf_by_concept, zipf_threshold)
        ]
        fewshot_components = stratified_component_selection(
            fewshot_eligible, int(settings["fewshot_reservoir_n"]), pool_stratum_counts, rng,
        )
        fewshot_ids = {component.component_id for component in fewshot_components}

        assignment_by_concept: dict[str, str] = {}
        for component in components:
            if component.component_id in direction_ids:
                split_name = "directions"
            elif component.component_id in fewshot_ids:
                split_name = "fewshot_reservoir"
            else:
                split_name = "test"
            for row in component.members:
                concept_id = normalize_nfc(row["concept_id"])
                if concept_id in assignment_by_concept:
                    raise ValueError(f"Concept assigned by multiple components: {concept_id!r}")
                assignment_by_concept[concept_id] = split_name

        partitioned_rows = [
            {**row, "split": assignment_by_concept[normalize_nfc(row["concept_id"])]}
            for row in sorted(rows, key=lambda item: normalize_nfc(item["concept_id"]))
        ]
        assert_disjoint_splits(partitioned_rows)
        if len(assignment_by_concept) != len(rows):
            raise ValueError("Some concepts were not assigned to exactly one split")

        row_counts: dict[str, list[dict[str, Any]]] = {
            split_name: [row for row in partitioned_rows if row["split"] == split_name]
            for split_name in SPLITS
        }
        schema = table.schema.append(pa.field("split", pa.string(), nullable=False))
        _write_parquet(partitioned_rows, schema, Path(paths["output"]), row_group_size)
        for split_name, path_key in (("fewshot_reservoir", "fewshot_reservoir"),
                                     ("directions", "directions"), ("test", "test_pool")):
            _write_parquet(row_counts[split_name], schema, Path(paths[path_key]), row_group_size)
        _replace_dropflow(Path(config["paths"]["dropflow"]), partitioned_rows)

        split_concept_counts: dict[str, Counter[tuple[str, str]]] = {
            split_name: Counter(strata_by_id[normalize_nfc(row["concept_id"])] for row in row_counts[split_name])
            for split_name in SPLITS
        }
        logger.info("Split sizes by POS x syllable bucket (pool share; split counts):")
        for key in sorted(pool_stratum_counts):
            pool_count = pool_stratum_counts[key]
            logger.info("%s | %s | pool=%d (%.6f) | fewshot_reservoir=%d | directions=%d | test=%d",
                        key[0], key[1], pool_count, pool_count / len(rows),
                        split_concept_counts["fewshot_reservoir"][key],
                        split_concept_counts["directions"][key], split_concept_counts["test"][key])

        logger.info("External attestation share by split (concept has any external_attested VI candidate):")
        for split_name in SPLITS:
            split_rows = row_counts[split_name]
            externally_attested = sum(
                any(candidate.get("external_attested") is True for candidate in row["vi_cands"])
                for row in split_rows
            )
            logger.info("%s | %d/%d (%.6f)", split_name, externally_attested, len(split_rows),
                        externally_attested / len(split_rows) if split_rows else 0.0)

        reservoir_rows = row_counts["fewshot_reservoir"]
        logger.info("Fewshot reservoir selected=%d concepts (target=%d); zipf_top_tercile_threshold=%.6f",
                    len(reservoir_rows), int(settings["fewshot_reservoir_n"]), zipf_threshold)
        logger.info("Fewshot reservoir list: en | pos | vi | zh | fr | id | n_sources")
        for row in reservoir_rows:
            if len(row["vi_cands"]) != 1:
                raise ValueError(f"Reservoir concept {row['concept_id']!r} does not have exactly one VI candidate")
            candidate = row["vi_cands"][0]
            logger.info("%s | %s | %s | %s | %s | %s | %s",
                        row["en_lemma"], row["pos"], candidate["word"],
                        _format_words(row["zh_cands"]), _format_words(row["fr_cands"]),
                        _format_words(row["id_cands"]), candidate["n_sources"])
        logger.info("Disjointness assertions passed: concept_id, VI forms, and English lemmas are split-exclusive")
        logger.info("Outputs: %s; %s; %s; %s",
                    paths["output"], paths["fewshot_reservoir"], paths["directions"], paths["test_pool"])
        elapsed = time.monotonic() - started
        logger.info("Runtime seconds: %.3f", elapsed)
        return {
            "components": len(components), "rows": len(rows),
            "split_sizes": {key: len(value) for key, value in row_counts.items()},
            "zipf_threshold": zipf_threshold, "runtime_seconds": elapsed,
        }
    except Exception as exc:
        logger.error("Split step stopped: %s", exc)
        logger.error("Runtime seconds: %.3f", time.monotonic() - started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
