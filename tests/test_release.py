"""End-to-end integrity checks for the frozen release and generated prompts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from data.build.common import strip_diacritics, vi_orth_key


ROOT = Path(__file__).resolve().parents[1]
CONCEPTS_PATH = ROOT / "data" / "concepts.tsv"
PROMPTS_DIR = ROOT / "data" / "prompts"
FEWSHOT_METADATA = ROOT / "data" / "interim" / "12_fewshot.json"
DROPflow_PATH = ROOT / "data" / "interim" / "dropflow.jsonl"
AGREEMENT_PATH = ROOT / "data" / "agreement.md"


def _read_concepts() -> list[dict[str, str]]:
    with CONCEPTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        assert first_line.startswith("# Pipeline commit: ")
        return list(csv.DictReader(handle, delimiter="\t"))


def _prompt_files() -> list[Path]:
    return sorted(path for path in PROMPTS_DIR.glob("*.jsonl") if path.is_file())


def _iter_prompt_records():
    for path in _prompt_files():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                assert isinstance(record, dict), f"{path}:{line_number} is not an object"
                yield path, line_number, record


def _vietnamese_fragments(record: dict[str, object]) -> list[str]:
    """Extract only Vietnamese portions, excluding Russian source text."""
    prompt = record.get("prompt")
    assert isinstance(prompt, str)
    format_name = record.get("format")
    fragments: list[str] = []
    if format_name == "translation":
        for line in prompt.splitlines():
            assert " - Tieng Viet:" in line, "nodiac translation label is not stripped"
            fragments.append(line.split(" - Tieng Viet:", 1)[1].strip())
    elif format_name == "repetition":
        for line in prompt.splitlines():
            pieces = line.split(" - Tieng Viet:")
            assert len(pieces) == 2 and pieces[0].startswith("Tieng Viet:"), line
            fragments.extend((pieces[0].removeprefix("Tieng Viet:").strip(), pieces[1].strip()))
    elif format_name == "cloze":
        for line in prompt.splitlines():
            if line.startswith("Dap an:"):
                fragments.append(line.removeprefix("Dap an:").strip())
            else:
                fragments.append(line)
    else:
        raise AssertionError(f"Unknown nodiac prompt format: {format_name!r}")
    target_vi = record.get("target_vi")
    if target_vi is not None:
        assert isinstance(target_vi, str)
        fragments.append(target_vi)
    return fragments


def _split_stage(stage: str) -> tuple[str, str]:
    for split in ("fewshot_reservoir", "main_test", "directions", "test"):
        suffix = "_" + split
        if stage.endswith(suffix):
            return split, stage[:-len(suffix)]
    return "all", stage


def _latest_dropflow(records: list[dict[str, object]]) -> list[dict[str, object]]:
    starts: dict[str, list[int]] = {}
    previous = None
    for index, record in enumerate(records):
        step = record["step"]
        if step != previous:
            starts.setdefault(str(step), []).append(index)
            previous = step
    last_start = {step: indices[-1] for step, indices in starts.items()}
    return [record for index, record in enumerate(records) if index >= last_start[str(record["step"])]]


def _dropflow_signature(record: dict[str, object]) -> tuple[object, ...]:
    split, stage = _split_stage(str(record["stage"]))
    return (
        str(record["step"]), split, stage, str(record.get("unit", "")),
        int(record["n_in"]), int(record["n_out"]), record.get("n_out_by_pos", {}),
    )


def test_release_split_disjointness_by_id_vi_orth_key_and_english() -> None:
    rows = _read_concepts()
    assert rows
    for field, normalize in (
        ("concept_id", lambda value: value),
        ("vi", vi_orth_key),
        ("en", lambda value: value),
    ):
        split_by_value: dict[str, set[str]] = {}
        for row in rows:
            value = normalize(row[field])
            assert value, f"empty {field} in concept row {row.get('concept_id')}"
            split_by_value.setdefault(value, set()).add(row["split"])
        collisions = {value: splits for value, splits in split_by_value.items() if len(splits) > 1}
        assert not collisions, f"{field} occurs across splits: {list(collisions.items())[:10]!r}"


def test_fewshot_concepts_are_never_prompt_test_items() -> None:
    metadata = json.loads(FEWSHOT_METADATA.read_text(encoding="utf-8"))
    fewshot_ids = {concept_id for ids in metadata["fewshot_sets"].values() for concept_id in ids}
    assert fewshot_ids
    conflicts = []
    for path, line_number, record in _iter_prompt_records():
        if record.get("split") == "test" and record.get("concept_id") in fewshot_ids:
            conflicts.append((str(path.relative_to(ROOT)), line_number, record.get("concept_id")))
    assert not conflicts, f"Few-shot concepts appear as test items: {conflicts[:10]!r}"


def test_prompt_concept_ids_and_targets_are_valid() -> None:
    concept_ids = {row["concept_id"] for row in _read_concepts()}
    for path, line_number, record in _iter_prompt_records():
        if "concept_id" in record:
            assert record["concept_id"] in concept_ids, (
                f"{path}:{line_number} has unknown concept_id {record['concept_id']!r}"
            )
        for key, value in record.items():
            if key == "target" or key.startswith("target_"):
                assert isinstance(value, str) and len(value) > 1, f"{path}:{line_number} has empty {key}"
                assert value.startswith(" ") and not value.startswith("  "), (
                    f"{path}:{line_number} {key} must start with exactly one space"
                )
                assert not value[1].isspace(), f"{path}:{line_number} {key} has extra leading whitespace"


def test_nodiac_prompts_strip_vietnamese_text_only() -> None:
    files = [path for path in _prompt_files() if "_nodiac_" in path.name]
    assert files
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                for fragment in _vietnamese_fragments(record):
                    assert strip_diacritics(fragment, preserve_case=True) == fragment, (
                        f"Vietnamese diacritic remains in {path.name}:{line_number}: {fragment!r}"
                    )


def test_release_has_resolved_concreteness_and_boolean_muse_flag() -> None:
    rows = _read_concepts()
    for row in rows:
        assert row["concreteness_match"] in {"exact", "head", "none"}
        assert row["in_muse"] in {"true", "false"}


def test_agreement_dropflow_matches_latest_structured_dropflow() -> None:
    records = [json.loads(line) for line in DROPflow_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = [_dropflow_signature(record) for record in _latest_dropflow(records)]
    agreement = AGREEMENT_PATH.read_text(encoding="utf-8")
    anchor = "## Dropflow (latest run per step)"
    assert anchor in agreement
    all_lines = agreement.splitlines()
    start = all_lines.index(anchor) + 1
    while start < len(all_lines) and not all_lines[start].startswith("|"):
        start += 1
    lines = []
    for line in all_lines[start:]:
        if not line.startswith("|"):
            break
        lines.append(line)
    assert len(lines) >= 3
    rendered = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 7, line
        rendered.append((
            cells[0], cells[1], cells[2], cells[3], int(cells[4]), int(cells[5]), json.loads(cells[6]),
        ))
    assert rendered == expected


def test_no_flores_text_is_tracked_by_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True,
    )
    paths = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    flores_paths = [path for path in paths if path.startswith("data/raw/flores/")
                    or path == "data/prompts/directions_flores.jsonl"]
    assert not flores_paths, f"FLORES text is tracked by git: {flores_paths!r}"
