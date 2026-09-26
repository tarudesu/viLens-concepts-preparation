from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_script(name: str):
    path = ROOT / "data" / "build" / name
    spec = importlib.util.spec_from_file_location(f"fixture_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text)))


def test_09b_concreteness_exact_head_none_and_blank_score() -> None:
    step = load_script("09b_concreteness.py")
    norms = {("ice cream", 1): 4.31, ("food", 0): 4.80, ("known blank", 1): None}

    class FakeNLP:
        def __call__(self, text: str):
            if text == "cat food":
                return [SimpleNamespace(pos_="NOUN", dep_="compound", lemma_="cat", text="cat"),
                        SimpleNamespace(pos_="NOUN", dep_="ROOT", lemma_="food", text="food")]
            return [SimpleNamespace(pos_="VERB", dep_="ROOT", lemma_="run", text="ran")]

    assert step.match_concreteness("ice cream", "noun", norms, FakeNLP()) == (4.31, "exact")
    assert step.match_concreteness("cat food", "noun", norms, FakeNLP()) == (4.80, "head")
    assert step.match_concreteness("known blank", "noun", norms, FakeNLP()) == (None, "exact")
    assert step.match_concreteness("unlisted", "verb", norms, FakeNLP()) == (None, "none")


def test_09b_reads_fixture_norms_and_rejects_unexpected_headers(tmp_path: Path) -> None:
    step = load_script("09b_concreteness.py")
    norms = step.load_norms(FIXTURES / "brysbaert_concreteness.csv", word_column="Word", value_column="Conc.M")
    assert norms[("ice cream", 1)] == 4.31
    assert norms[("known blank", 1)] is None
    invalid = tmp_path / "wrong.csv"
    invalid.write_text("term,score\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected Brysbaert CSV headers"):
        step.load_norms(invalid, word_column="Word", value_column="Conc.M")


def test_09b_xlsx_bigram_exact_head_fallback_and_duplicate_resolution(tmp_path: Path, caplog) -> None:
    import pandas as pd

    step = load_script("09b_concreteness.py")
    workbook = tmp_path / "norms.xlsx"
    pd.DataFrame([
        {"Word": " CAT ", "Bigram": 0, "Conc.M": 4.2, "Total": 10},
        {"Word": "cat", "Bigram": 0, "Conc.M": 3.1, "Total": 8},
        {"Word": "ice cream", "Bigram": 1, "Conc.M": 4.8, "Total": 23},
        {"Word": "food", "Bigram": 0, "Conc.M": 4.0, "Total": 12},
    ]).to_excel(workbook, index=False)
    with caplog.at_level(logging.WARNING, logger=step.STEP):
        norms = step.load_norms(
            workbook, word_column="Word", value_column="Conc.M", bigram_column="Bigram",
            raters_column="Total",
        )

    class FakeNLP:
        def __call__(self, text: str):
            assert text == "cat food"
            return [SimpleNamespace(pos_="NOUN", dep_="compound", lemma_="cat", text="cat"),
                    SimpleNamespace(pos_="NOUN", dep_="ROOT", lemma_="food", text="food")]

    assert norms[("cat", 0)] == 4.2
    assert step.match_concreteness("cat", "noun", norms, FakeNLP()) == (4.2, "exact")
    assert step.match_concreteness("ice cream", "noun", norms, FakeNLP()) == (4.8, "exact")
    assert step.match_concreteness("cat food", "noun", norms, FakeNLP()) == (4.0, "head")
    assert "Duplicate normalized Brysbaert Word='cat'" in caplog.text


def test_09b_missing_inputs_diagnostic(tmp_path: Path) -> None:
    step = load_script("09b_concreteness.py")
    missing = [tmp_path / "11_diac.parquet", tmp_path / "11_m1_ext.parquet"]
    assert step.missing_inputs_message(missing) == (
        f"Step 09b requires step-11 main and M1 extension inputs: missing {missing[0]}, {missing[1]}"
    )
    assert step.missing_norms_message(tmp_path / "brysbaert") == (
        f"Step 09b requires Brysbaert file(s) in {tmp_path / 'brysbaert'}/; none found. "
        "Place the original norms file there, then run make concreteness."
    )


def test_09b_missing_manual_norms_stops_before_download(tmp_path: Path) -> None:
    step = load_script("09b_concreteness.py")
    main_input, extension_input = tmp_path / "main.parquet", tmp_path / "extension.parquet"
    main_input.touch()
    extension_input.touch()
    raw_dir = tmp_path / "brysbaert"
    config = _write_config(tmp_path, {
        "paths": {"logs": str(tmp_path / "logs")},
        "concreteness": {"paths": {"main_input": str(main_input), "extension_input": str(extension_input),
                                     "raw_dir": str(raw_dir)}},
    })
    with pytest.raises(FileNotFoundError, match=r"Step 09b requires Brysbaert file\(s\)"):
        step.run(config)


def test_10b_sentence_and_aggregate_fertility() -> None:
    step = load_script("10b_fertility.py")
    assert step.sentence_counts(CharTokenizer(), "xin chào", language="vi") == (8, 8, 2)
    result = step.aggregate_fertility(["xin chào", "Việt Nam"], CharTokenizer(), language="vi")
    assert result == {
        "n_sentences": 2, "total_tokens": 16, "total_characters": 16,
        "tokens_per_character": 1.0, "total_syllables": 4, "tokens_per_syllable": 4.0,
    }
    assert step.aggregate_fertility(["hello"], CharTokenizer(), language="en")["tokens_per_syllable"] is None


def test_10b_missing_input_diagnostic(tmp_path: Path) -> None:
    step = load_script("10b_fertility.py")
    paths = [tmp_path / "devtest_vie_Latn.parquet", tmp_path / "devtest_eng_Latn.parquet"]
    assert step.missing_flores_message(paths) == (
        f"Step 10b requires FLORES devtest input files: missing {paths[0]}, {paths[1]}"
    )


def test_13b_same_seeded_split_for_every_language() -> None:
    step = load_script("13b_direction_flores.py")
    source = {f"s{i}": f"Sentence {i}" for i in range(10)}
    datasets = {language: [{"sentence_id": sid, "sentence": f"{language} {text}"} for sid, text in source.items()]
                for language in ("vi", "en", "zh")}
    records, counts = step.build_direction_records(datasets, fit_fraction=0.8, seed=20260925)
    assert counts == {"fit": 8, "validate": 2}
    splits_by_id: dict[str, set[str]] = {}
    for record in records:
        splits_by_id.setdefault(record["sentence_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in splits_by_id.values())
    assert len(records) == 30


def test_13b_rejects_different_language_sentence_ids() -> None:
    step = load_script("13b_direction_flores.py")
    with pytest.raises(ValueError, match="sentence IDs differ"):
        step.build_direction_records(
            {"en": [{"sentence_id": "1", "sentence": "a"}],
             "vi": [{"sentence_id": "2", "sentence": "b"}]},
            fit_fraction=0.8, seed=1,
        )


def test_13b_missing_input_diagnostic(tmp_path: Path) -> None:
    step = load_script("13b_direction_flores.py")
    paths = [tmp_path / "dev_vie_Latn.parquet", tmp_path / "dev_eng_Latn.parquet"]
    assert step.missing_flores_message(paths) == f"Step 13b requires FLORES dev input files: missing {paths[0]}, {paths[1]}"


def test_14_structured_fewshot_metadata_release_projection_and_tsv(tmp_path: Path) -> None:
    step = load_script("14_release.py")
    metadata_path = tmp_path / "12_fewshot.json"
    metadata_path.write_text(json.dumps({
        "schema_version": 1,
        "fewshot_sets": {"1": ["b"], "2": ["c"]},
        "cloze_demonstration_concept_ids": ["b"],
        "test_cloze_concept_ids": ["test-x"],
        "cloze_status": "supplementary",
        "main_test_cloze_coverage": {"n_available": 1, "n_test": 4,
                                      "by_stratum": {"sino": {"n_available": 1, "n_test": 2}}},
    }, ensure_ascii=False), encoding="utf-8")
    metadata = step._read_fewshot_metadata(metadata_path)
    fewshot = metadata["fewshot_set_by_id"]
    assert fewshot == {"b": 1, "c": 2}
    columns = step.release_columns(["vi", "en", "zh", "fr", "id"], ["gemma"])
    row = {
        "concept_id": "b", "split": "test", "vi_canonical": "mèo", "en_lemma": "cat",
        "zh_canonical": "猫", "fr_canonical": "chat", "id_canonical": "kucing", "ru_canonical": "кошка",
        "vi_cands": [{"word": "mèo", "in_vi_gloss": True, "in_muse": False, "in_wikidata": True,
                      "external_attested": True}],
        "vi_alts": [], "vi_nodiac": "meo", "collapsed": False, "pos": "noun", "sense_gloss": "feline",
        "stratum": "sino", "signal_a": "sino", "signal_b": "sino", "signal_b_strict": "nonsino",
        "signal_b_relaxed": "sino", "sino_via": ["template"], "native_strict": False,
        "han_verifications": [{"primary_pass": True, "reading_source": "wiktionary_char"}], "n_sources": 3,
        "external_attested": True, "bt_route": "strict", "bt_pass_strict_v1": True,
        "zh_nllb_agree": True, "fr_nllb_agree": False, "id_nllb_agree": True, "ru_nllb_agree": True,
        "min_surface_dist": 0.6, "n_senses_vi": 1, "n_syllables": 1,
        "concreteness": 4.0, "concreteness_match": "exact", "m1_extension": False,
        "zipf_vi": 5.0, "zipf_en": 5.0, "zipf_zh": 5.0, "zipf_fr": 5.0, "zipf_id": 5.0,
        "n_chars_vi": 3, "n_chars_en": 3, "n_chars_zh": 1, "n_chars_fr": 4, "n_chars_id": 5,
        "single_token_vi_gemma": True, "m1_eligible_gemma": True,
        **{f"ntok_gemma_{language}": 1 for language in ("vi", "en", "zh", "fr", "id")},
    }
    projected = step.project_release_row(row, columns=columns, fewshot_sets=fewshot, cloze_ids={"b"})
    assert projected["reading_source"] == "wiktionary_char"
    assert projected["fewshot_set"] == 1
    assert projected["cloze_available"] is True
    rendered = step.render_concepts_tsv([projected], columns, "a" * 40)
    assert rendered.startswith("# Pipeline commit: " + "a" * 40 + "\n")
    assert "concept_id\tsplit\tvi\ten" in rendered


def test_14_latest_dropflow_retains_latest_contiguous_step_run(tmp_path: Path) -> None:
    step = load_script("14_release.py")
    path = tmp_path / "dropflow.jsonl"
    records = [
        {"step": "06", "stage": "old_a"}, {"step": "06", "stage": "old_b"},
        {"step": "07", "stage": "current"},
        {"step": "06", "stage": "new_a"}, {"step": "06", "stage": "new_b"},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    assert [item["stage"] for item in step.latest_dropflow_records(path)] == ["current", "new_a", "new_b"]


def test_14_pending_guard_and_missing_input_message(tmp_path: Path) -> None:
    step = load_script("14_release.py")
    path = tmp_path / "12_ru.parquet"
    assert step.missing_input_message(path) == f"Step 14 requires step-12 enriched input: missing {path}"
    from data.build.common import require_resolved_concreteness
    with pytest.raises(ValueError, match="concreteness_match is pending"):
        require_resolved_concreteness([{"concept_id": "x", "concreteness_match": "pending"}])


def test_14_release_input_refuses_pending_concreteness(tmp_path: Path) -> None:
    step = load_script("14_release.py")
    input_path = tmp_path / "12_ru.parquet"
    row = {
        "concept_id": "fixture", "split": "test", "vi_canonical": "mèo", "en_lemma": "cat",
        "zh_canonical": "猫", "fr_canonical": "chat", "id_canonical": "kucing", "pos": "noun",
        "sense_gloss": "feline", "stratum": "sino", "signal_a": "sino", "signal_b": "sino",
        "signal_b_strict": "nonsino", "signal_b_relaxed": "sino", "sino_via": [],
        "native_strict": False, "n_sources": 2, "bt_route": "strict", "bt_pass_strict_v1": True,
        "min_surface_dist": 1.0, "n_senses_vi": 1, "vi_nodiac": "meo", "collapsed": False,
        "concreteness": None, "concreteness_match": "pending", "m1_extension": False,
        "han_verifications": [],
    }
    pq.write_table(pa.Table.from_pylist([row]), input_path)
    release_paths = {name: str(tmp_path / name) for name in (
        "input", "dropflow", "sources", "protocol", "output_tsv", "agreement", "licenses",
        "qa_sample", "fertility", "flores_directions", "prompts_dir", "logs_dir",
    )}
    release_paths["input"] = str(input_path)
    config = {"release": {"paths": release_paths}}
    with pytest.raises(ValueError, match="Refusing final dataset assembly: concreteness_match is pending"):
        step._read_latest_inputs(config, logger=logging.getLogger("fixture-release"))


def test_dropflow_split_suffix_parser() -> None:
    step = load_script("14_release.py")
    assert step._dropflow_split("filter4_surface_test") == ("test", "filter4_surface")
    assert step._dropflow_split("extension_candidates") == ("all", "extension_candidates")


def test_14_agreement_contains_flores_hash_only(tmp_path: Path) -> None:
    step = load_script("14_release.py")
    flores = tmp_path / "directions_flores.jsonl"
    flores.write_text('{"sentence":"fixture FLORES sentence"}\n', encoding="utf-8")
    row = {
        "concept_id": "fixture", "split": "test", "vi_canonical": "mèo", "en_lemma": "cat",
        "pos": "noun", "stratum": "sino", "n_syllables": 1, "collapsed": False,
        "external_attested": True,
        "vi_cands": [{"word": "mèo", "in_vi_gloss": True, "in_muse": False,
                      "in_wikidata": True, "external_attested": True}],
        "han_verifications": [{"primary_pass": True, "reading_source": "unihan"}],
        "single_token_vi_gemma": True, "m1_eligible_gemma": True,
    }
    config = {
        "seed": 20260925,
        "langs": ["vi", "en", "zh", "fr", "id"],
        "downloads": {"chunk_size_bytes": 4},
        "split": {"syllable_bins": [1, 2]},
        "tokens": {"models": ["gemma"]},
        "etymology": {"bootstrap_resamples": 20, "bootstrap_confidence": 0.95,
                       "power_alpha": 0.05, "power_holm_comparisons": 6, "power_target": 0.8,
                       "h3_confirmatory_kappa_lower_min": 0.6, "h3_confirmatory_holm_mde_max": 0.3},
        "release": {"paths": {"sources": "data/build/SOURCES.md"}},
        "prompts": {"cloze": {"cloze_status": "supplementary"}},
    }
    fewshot_metadata = {
        "fewshot_sets": {}, "cloze_status": "supplementary",
        "main_test_cloze_coverage": {"n_available": 0, "n_test": 0, "by_stratum": {}},
    }
    report = step.build_agreement(
        [row], dropflow=[{"step": "06", "stage": "filter4_test", "unit": "concepts", "n_in": 1,
                          "n_out": 1, "n_out_by_pos": {"noun": 1}}],
        sources=[], config=config, fewshot_metadata=fewshot_metadata,
        fertility_path=tmp_path / "missing.csv",
        flores_path=flores, protocol_path=Path("PROTOCOL_AMENDMENTS.md"),
    )
    assert step.sha256_file(flores, chunk_size_bytes=4) in report
    assert "fixture FLORES sentence" not in report
    assert "filter4" in report and "wiktionary_char for 0/1" in report
    assert "Step-12 main-test cloze coverage: 0/0" in report
    assert "Selected few-shot concepts" in report


def test_ignored_flores_output() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/data/prompts/directions_flores.jsonl" in ignore


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    return path


def test_scripts_stop_with_clear_missing_input_diagnostics(tmp_path: Path) -> None:
    step09b = load_script("09b_concreteness.py")
    config09b = _write_config(tmp_path, {
        "paths": {"logs": str(tmp_path / "logs")},
        "concreteness": {"paths": {"main_input": str(tmp_path / "missing-main.parquet"),
                                     "extension_input": str(tmp_path / "missing-ext.parquet")}},
    })
    with pytest.raises(FileNotFoundError, match="Step 09b requires step-11 main and M1 extension inputs"):
        step09b.run(config09b)

    step10b = load_script("10b_fertility.py")
    flores_dir = tmp_path / "no-flores"
    languages = ["vi", "en", "zh", "fr", "id"]
    codes = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "cmn_Hans", "fr": "fra_Latn", "id": "ind_Latn"}
    config10b = _write_config(tmp_path, {
        "paths": {"logs": str(tmp_path / "logs")}, "langs": languages,
        "fertility": {"paths": {"flores_dir": str(flores_dir)}},
        "direction_flores": {"codes": codes},
    })
    with pytest.raises(FileNotFoundError, match="Step 10b requires FLORES devtest input files"):
        step10b.run(config10b)

    step13b = load_script("13b_direction_flores.py")
    config13b = _write_config(tmp_path, {
        "paths": {"logs": str(tmp_path / "logs")}, "langs": languages,
        "direction_flores": {"languages": languages, "codes": codes,
                             "paths": {"flores_dir": str(flores_dir)},
                             "id_column": "id", "sentence_column": "sentence"},
    })
    with pytest.raises(FileNotFoundError, match="Step 13b requires FLORES dev input files"):
        step13b.run(config13b)

    step14 = load_script("14_release.py")
    config14 = _write_config(tmp_path, {
        "paths": {"logs": str(tmp_path / "logs")},
        "release": {"paths": {"input": str(tmp_path / "missing-12-ru.parquet")}},
    })
    with pytest.raises(FileNotFoundError, match="Step 14 requires step-12 enriched input"):
        step14.run(config14)
