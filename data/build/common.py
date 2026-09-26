"""Shared, deterministic helpers for the data-building pipeline."""

from __future__ import annotations

import json
import logging
import sys
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


class DataConfigError(ValueError):
    """Raised when a pipeline configuration cannot be loaded."""


def _unique_mapping(pairs: Iterable[tuple[Any, Any]]) -> dict[Any, Any]:
    """Build an NFC-normalized mapping without silently overwriting keys."""

    result: dict[Any, Any] = {}
    for key, value in pairs:
        key = normalize_nfc(key) if isinstance(key, str) else key
        if key in result:
            raise ValueError(f"Duplicate mapping key after NFC normalization: {key!r}")
        result[key] = value
    return result


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate or NFC-colliding keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        return _unique_mapping(
            (self.construct_object(key, deep=deep), self.construct_object(value, deep=deep))
            for key, value in node.value
        )


def normalize_nfc(value: str) -> str:
    """Return *value* normalized to Unicode NFC."""

    if not isinstance(value, str):
        raise TypeError(f"normalize_nfc expects str, got {type(value).__name__}")
    return unicodedata.normalize("NFC", value)


def vi_orth_key(value: str) -> str:
    """Return a normalized Vietnamese orthographic-equivalence key.

    Older tone placement in open Vietnamese data is mapped to the modern
    first-vowel placement for open ``oa``, ``oe``, and ``uy`` rimes. A final
    ``y`` after the specified consonants is folded to ``i``; standalone ``y``
    and ``qu`` spellings are left intact.
    """
    if not isinstance(value, str):
        raise TypeError(f"vi_orth_key expects str, got {type(value).__name__}")

    tone_marks = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
    modern_rimes = {"oa", "oe", "uy"}
    final_y_onsets = set("bcđdhklmnstvx")
    modern_syllables: list[str] = []

    for syllable in normalize_nfc(value).casefold().split():
        graphemes: list[list[str]] = []
        for char in unicodedata.normalize("NFD", syllable):
            if unicodedata.combining(char):
                if not graphemes:
                    graphemes.append([char])
                else:
                    graphemes[-1].append(char)
            else:
                graphemes.append([char])

        bases = "".join(group[0] for group in graphemes)
        if len(graphemes) >= 2 and bases[-2:] in modern_rimes and not bases.startswith("qu"):
            first, second = graphemes[-2], graphemes[-1]
            old_tone = [mark for mark in second[1:] if mark in tone_marks]
            if old_tone:
                second[:] = [second[0], *(mark for mark in second[1:] if mark not in tone_marks)]
                first.extend(old_tone)

        bases = "".join(group[0] for group in graphemes)
        if bases.endswith("y") and bases[0] in final_y_onsets and not bases.startswith("qu"):
            graphemes[-1][0] = "i"

        modern_syllables.append(normalize_nfc("".join("".join(group) for group in graphemes)))

    return " ".join(modern_syllables)


def strip_diacritics(value: str) -> str:
    """Remove combining marks and Vietnamese đ/Đ from a string."""

    if not isinstance(value, str):
        raise TypeError(f"strip_diacritics expects str, got {type(value).__name__}")
    decomposed = unicodedata.normalize("NFD", normalize_nfc(value))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.replace("đ", "d").replace("Đ", "D").lower()


def contextual_tokenization(tokenizer: Any, prefix: str, word: str) -> tuple[int, list[str]]:
    """Return the no-special-token contextual token-count difference and suffix tokens."""
    prefix = normalize_nfc(prefix)
    word = normalize_nfc(word)
    if not prefix.strip():
        raise ValueError("Token context prefix must be non-empty")
    if not word.strip():
        raise ValueError("Cannot tokenize an empty canonical form")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    context_ids = tokenizer.encode(f"{prefix} {word}", add_special_tokens=False)
    if len(context_ids) < len(prefix_ids):
        raise ValueError(
            "Context encoding is shorter than prefix encoding for "
            f"prefix={prefix!r}, word={word!r}: {len(context_ids)} < {len(prefix_ids)}"
        )
    token_strings = tokenizer.convert_ids_to_tokens(context_ids)
    if isinstance(token_strings, str):
        token_strings = [token_strings]
    if not isinstance(token_strings, list) or len(token_strings) != len(context_ids):
        raise ValueError(
            "Tokenizer returned an unexpected token-string sequence for "
            f"prefix={prefix!r}, word={word!r}"
        )
    if any(not isinstance(token, str) for token in token_strings):
        raise ValueError(f"Tokenizer returned a non-string token for {word!r}: {token_strings!r}")
    count = len(context_ids) - len(prefix_ids)
    incremental_tokens = [normalize_nfc(token) for token in token_strings[len(prefix_ids):]]
    if len(incremental_tokens) != count:
        raise ValueError(f"Tokenizer count/token-string mismatch for {word!r}")
    return count, incremental_tokens


def require_resolved_concreteness(rows: Iterable[Mapping[str, Any]]) -> None:
    """Refuse final release assembly while any row has pending concreteness."""
    pending: list[str] = []
    for row in rows:
        if row.get("concreteness_match") == "pending":
            identifier = row.get("concept_id", "<unknown concept>")
            pending.append(str(identifier))
    if pending:
        preview = ", ".join(pending[:10])
        suffix = "" if len(pending) <= 10 else f" (and {len(pending) - 10} more)"
        raise ValueError(
            "Refusing final dataset assembly: concreteness_match is pending for "
            f"{len(pending)} concepts: {preview}{suffix}"
        )


def normalized_levenshtein(left: str, right: str) -> float:
    """Return Levenshtein distance divided by the longer string length."""

    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("normalized_levenshtein expects two strings")
    left, right = normalize_nfc(left), normalize_nfc(right)
    if left == right:
        return 0.0
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 0.0
    # Two-row dynamic program keeps memory linear in the shorter input.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1] / denominator


def normalize_strings(value: Any) -> Any:
    """Recursively normalize every string in a JSON/YAML-compatible value."""

    if isinstance(value, str):
        return normalize_nfc(value)
    if isinstance(value, Mapping):
        return _unique_mapping(
            (key, normalize_strings(item)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [normalize_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_strings(item) for item in value)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and normalize all strings to Unicode NFC."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8", newline="") as handle:
            loaded = yaml.load(handle, Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise DataConfigError(f"Could not read config {config_path}: {exc}") from exc
    except (yaml.YAMLError, ValueError, TypeError) as exc:
        raise DataConfigError(f"Invalid YAML in config {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise DataConfigError(f"Config {config_path} must contain a top-level mapping")
    return normalize_strings(loaded)


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    """Stream NFC-normalized JSONL; stop with a diagnostic on malformed input."""

    jsonl_path = Path(path)
    try:
        handle = jsonl_path.open("rb")
    except OSError as exc:
        raise OSError(f"Could not read JSONL file {jsonl_path}: {exc}") from exc

    with handle:
        # Read bytes so invalid UTF-8 can be attributed to the exact record.
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_unique_mapping,
                    parse_constant=_reject_json_constant,
                )
                record = normalize_strings(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc}"
                ) from exc
            yield record


def _reject_json_constant(value: str) -> Any:
    """Reject the non-standard NaN/Infinity values accepted by json.loads."""

    raise ValueError(f"Non-standard JSON constant: {value}")


# Explicit alias for callers that prefer the "stream" naming.
stream_jsonl = iter_jsonl


class DropflowLogger:
    """Append one deterministic record per stage; audit history grows on reruns."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        step: str,
        stage: str,
        unit: str,
        n_in: int,
        n_out: int,
        n_out_by_pos: Mapping[str, int],
    ) -> None:
        normalized_unit = normalize_nfc(unit)
        if normalized_unit not in {"entries", "groups", "concepts", "items"}:
            raise ValueError(f"Dropflow unit must be entries/groups/concepts/items, got {unit!r}")
        counts = normalize_strings(dict(n_out_by_pos))
        if any(not isinstance(key, str) for key in counts):
            raise ValueError("n_out_by_pos keys must be POS strings")
        if any(type(value) is not int or value < 0 for value in (n_in, n_out, *counts.values())):
            raise ValueError("Dropflow counts must be non-negative integers")
        if n_out > n_in:
            raise ValueError("Dropflow n_out cannot exceed n_in for a filter stage")
        if sum(counts.values()) != n_out:
            raise ValueError("Dropflow n_out_by_pos counts must sum to n_out")
        payload = {
            "step": normalize_nfc(step),
            "stage": normalize_nfc(stage),
            "unit": normalized_unit,
            "n_in": n_in,
            "n_out": n_out,
            "n_out_by_pos": counts,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def log_dropflow(
    path: str | Path,
    *,
    step: str,
    stage: str,
    unit: str,
    n_in: int,
    n_out: int,
    n_out_by_pos: Mapping[str, int],
) -> None:
    """Append one dropflow record; convenient functional API."""

    DropflowLogger(path).record(
        step=step,
        stage=stage,
        unit=unit,
        n_in=n_in,
        n_out=n_out,
        n_out_by_pos=n_out_by_pos,
    )


def setup_logging(
    step_name: str,
    log_dir: str | Path,
    *,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Log to stdout and a per-run file, without nondeterministic timestamps.

    Call once at step startup. Reconfiguration closes old handlers and replaces
    the log file, so identical runs produce identical log bytes.
    """

    normalized_step = normalize_nfc(step_name)
    if not normalized_step or Path(normalized_step).name != normalized_step:
        raise ValueError("step_name must be a non-empty filename stem")
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{normalized_step}.log"

    logger = logging.getLogger(normalized_step)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
