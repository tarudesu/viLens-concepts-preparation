"""Restore omitted concreteness fields from a user's original Brysbaert workbook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


# Exact spaCy en_core_web_sm head lemmas used by step 09b for the frozen rows
# whose original concreteness_match is "head". Embedding these 135 short lemmas
# keeps this restoration utility independent of spaCy and its model download.
HEAD_LEMMA_BY_CONCEPT = json.loads(r'''{"0043e6e50b1c":"acid","00cea385a34c":"plane","011e3ec152c3":"number","03be33f1068e":"language","03fbf58b1039":"distance","09f0172c771d":"radiation","0aecc9bd333b":"death","0b30e78e941f":"son","0e28b35ce2db":"number","115215ca2516":"wheel","115452f5039f":"system","1487dd54564b":"piety","15442cfab330":"powder","15fead9f58cd":"past","168e30faf59a":"wind","17781bd98b3c":"past","17a9a7b571c2":"license","17baef84280c":"security","1ad7dff397f7":"engineering","20a767540f31":"biology","20f7e5b26663":"type","2248a8f11b57":"government","23f9cba619fa":"sister","2788a6f53477":"transport","281efbf0693d":"straw","2b544d68941e":"water","2e5fb5ade429":"circuit","30b565729474":"family","35723aa3920d":"pyramid","366709c42b47":"network","36b019ddd0f9":"pad","37b994ef1743":"engineering","3ad25b6e80f2":"art","3b65a8210e1f":"being","3c530da0988f":"crime","3cff44322c86":"sprout","3df297ade35c":"milk","43283a567aff":"language","43591d2a79c4":"socialism","468114afd863":"page","4af27cfb3573":"gas","4b370af7492f":"commander","4fb5f13429f6":"eclipse","51671681bca2":"network","534c7bda8088":"learning","53d0391ed774":"law","5534232d1148":"market","56485ebf2145":"medicine","57cfeeb2f5cd":"economy","5a8533180bbc":"carbon","5aa4f38597d8":"stem","5fdd8b593240":"expression","6046c987e08a":"earth","6120c9ee8abf":"rhinoceros","612c7148ea5a":"alarm","65f53187cd15":"horizon","65fc74075bed":"science","67ce60a27765":"structure","69a31becddb5":"algebra","6bd2faab1902":"term","6e7d9343315a":"letter","700dd7987578":"bladder","7686e96124e8":"park","78919f1a38c0":"appendix","78a06d04289a":"camp","7afdcd672858":"wave","7b6f2d1cb268":"life","7b8e15434d0d":"bladder","7d7ff1b2abb2":"intestine","84c5fb3e0d64":"water","858f4dcd986a":"intercourse","883cf62afc34":"property","8a5339facc63":"throw","8c3aa09c2ef7":"science","8cc4fceeadfd":"cigarette","8f79a9486d4a":"mechanic","905cb15ad577":"immunity","94eb52627ff2":"clothe","97cce6cd052a":"tense","9b992c6083eb":"year","9caac4110154":"season","9e1196d4841f":"marry","a09a6f119445":"minister","a1fe1e53e637":"intestine","a46f67ee2976":"car","aa52dc113981":"tea","aae2048a7467":"pearl","ac146be2bbea":"bean","ad57a3be3566":"code","ae8a3e439a9d":"system","aee093794731":"science","afa5735f1a1f":"chain","b0278fa807f4":"system","b1abcee0e231":"war","b59af4ada628":"rail","b5a73aa18a26":"look","b8705a66c2b8":"currency","bb52be72aecd":"design","bc51343a3fa4":"tea","bc634d6b554a":"boat","c1ebe2859d7e":"field","c1fbd5c3dab6":"movie","c43fee483c50":"bomb","c4ddd88c886a":"milk","c71f93a6b774":"chemistry","c77cd8153fdd":"sauce","ccb657477d47":"intelligence","d1d9ec3861b9":"city","d35eac2c1dec":"radical","d530875cb44e":"screen","d90d2d6b4b3f":"matter","da350733c8d4":"tooth","ddb3b2177ee4":"computer","dde3e1431245":"enterprise","df507a2b0d4b":"technology","dfac41abdad6":"prince","e0d2b9ef3a16":"group","e2b1afd33c35":"number","e55effb31258":"pencil","ebb2861f368a":"right","f0f453781731":"cycle","f105ab2e4b7e":"boar","f1268a5ebb20":"warehouse","f1321a4895f7":"come","f1a1ccd56890":"park","f53a343fa6b3":"whale","f5411853b246":"do","f7033acfe5d8":"paralysis","f918ffa42beb":"bean","fa1e8c328762":"bear","fac66ca0e79c":"lose","fadf2b9010ce":"range","fcabbcc365db":"cucumber","fe9df4bb97a2":"chain","ffa089fd025d":"election"}''')


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
    """Warn if the workbook digest differs from its provenance record."""
    candidates = [row for row in source_rows(sources_path)
                  if row["source"] == "brysbaert" and Path(row["file"]).name == path.name]
    if len(candidates) != 1:
        print(f"WARNING: no unique Brysbaert provenance row for {path.name}; continuing.", file=sys.stderr)
        return
    actual = sha256_file(path)
    if actual != candidates[0]["SHA-256"]:
        print(f"WARNING: SHA-256 mismatch for {path.name}; expected {candidates[0]['SHA-256']}, got {actual}.", file=sys.stderr)


def norm_key(value: str) -> str:
    """Normalize a source word using the step-09b exact-match rule."""
    return unicodedata.normalize("NFC", value).strip().lower()


def load_norms(path: Path) -> dict[tuple[str, int], float | None]:
    """Load workbook scores with Bigram filtering and highest-Total duplicates."""
    frame = pd.read_excel(path, engine="openpyxl", keep_default_na=False, na_values={})
    required = {"Word", "Conc.M", "Bigram", "Total"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Workbook missing required columns {missing}; found {list(frame.columns)}")
    grouped: dict[tuple[str, int], list[tuple[int, float | None, float]]] = {}
    for row_number, record in enumerate(frame.to_dict(orient="records"), start=2):
        raw_word = record["Word"]
        if not isinstance(raw_word, str) or not raw_word.strip():
            raise ValueError(f"Blank Word cell at workbook row {row_number}")
        word = norm_key(raw_word)
        try:
            bigram_float = float(record["Bigram"])
            total = float(record["Total"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Bigram or Total at workbook row {row_number}") from exc
        if bigram_float not in (0.0, 1.0) or not math.isfinite(total) or total < 0:
            raise ValueError(f"Invalid Bigram or Total at workbook row {row_number}")
        bigram = int(bigram_float)
        raw_score: Any = record["Conc.M"]
        if raw_score is None or (isinstance(raw_score, str) and not raw_score.strip()):
            score = None
        else:
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid Conc.M at workbook row {row_number}") from exc
            if math.isnan(score):
                score = None
        grouped.setdefault((word, bigram), []).append((row_number, score, total))
    norms: dict[tuple[str, int], float | None] = {}
    for key, candidates in sorted(grouped.items()):
        max_total = max(candidate[2] for candidate in candidates)
        winners = [candidate for candidate in candidates if candidate[2] == max_total]
        scores = {candidate[1] for candidate in winners}
        if len(scores) > 1:
            raise ValueError(f"Conflicting tied highest-Total duplicate for {key[0]!r}, Bigram={key[1]}")
        norms[key] = winners[0][1]
    if not norms:
        raise ValueError("Workbook contains no usable norms")
    return norms


def read_concepts(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    """Read the release TSV while preserving its pipeline comment line."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        comment = handle.readline()
        if not comment.startswith("# Pipeline commit: "):
            raise ValueError(f"Missing pipeline commit comment in {path}")
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing TSV header in {path}")
        required = {"concept_id", "en", "pos", "concreteness_match"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Input is missing required columns: {sorted(required - set(reader.fieldnames))}")
        return comment, list(reader.fieldnames), list(reader)


def restore_concreteness(row: dict[str, str], norms: dict[tuple[str, int], float | None]) -> tuple[float | None, str]:
    """Match exact one/two-word entries, then the frozen step-09b head lemma."""
    exact = norm_key(row["en"])
    word_count = len(exact.split())
    if word_count in (1, 2):
        key = (exact, word_count - 1)
        if key in norms:
            return norms[key], "exact"
    head = HEAD_LEMMA_BY_CONCEPT.get(row["concept_id"])
    if head is not None and (norm_key(head), 0) in norms:
        return norms[(norm_key(head), 0)], "head"
    return None, "none"


def write_concreteness(
    comment: str, columns: list[str], rows: list[dict[str, str]], norms: dict[tuple[str, int], float | None], output: Path,
) -> None:
    """Insert/replace concreteness fields and write a deterministic UTF-8 TSV."""
    if "concreteness" in columns:
        columns.remove("concreteness")
    position = columns.index("concreteness_match")
    output_columns = columns[:position] + ["concreteness", "concreteness_match"] + columns[position + 1:]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(comment)
        writer = csv.DictWriter(handle, fieldnames=output_columns, delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            value, match = restore_concreteness(row, norms)
            row["concreteness"] = "" if value is None else format(value, ".12g")
            row["concreteness_match"] = match
            writer.writerow(row)


def main() -> int:
    """Run the standalone concreteness restoration command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_source(args.xlsx, args.sources)
    norms = load_norms(args.xlsx)
    comment, columns, rows = read_concepts(args.concepts)
    write_concreteness(comment, columns, rows, norms, args.output)
    print(f"Wrote concreteness fields for {len(rows)} concepts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
