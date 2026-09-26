# AGENTS.md — viLens data pipeline

Standing instructions for any AI coding agent working in this repo. Read this file in full before every task. Task-specific instructions arrive separately, one step at a time; where a task and this file conflict, **stop and ask** rather than choosing yourself.

---

## 1. What this project is

A research codebase for the paper *"The Latent Language of Vietnamese Prompts: Where and When Do LLMs Translate Back?"* (protocol v1.1). This repo currently builds the **dataset**: a fully automated, five-way aligned (vi, en, zh, fr, id), etymology-tagged set of Vietnamese concepts, built only from open resources and used later for logit-lens, language-direction and steering experiments on Gemma 3 4B, Qwen3 4B and Llama 3.2 3B.

The dataset is itself a contribution of the paper, so **reproducibility, auditability and honesty about data loss matter more than speed or elegance.** Every row dropped must be counted, and every threshold must come from the config.

---

## 2. Environment

- Developed on macOS, Apple M4 Pro, 24 GB unified memory. Must also run unchanged on Linux + CUDA.
- Python version as pinned in `pyproject.toml`. Dependencies are managed with **uv**; always commit `uv.lock`. Run everything through `uv run ...`.
- Device selection for any torch code: `cuda` if available, else `mps` if available, else `cpu`. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the Makefile environment. Never assume CUDA.
- The data pipeline is CPU/IO-bound. The only model inference is NLLB back-translation (step 07). Step 10 loads **tokenizers only**, never model weights.
- Large dumps (Wiktextract) may be tens of GB uncompressed. Keep them compressed where possible and stream them with `gzip.open` / line iteration. **Never load a full dump into memory.**

---

## 3. Repository layout

```
configs/data.yaml        # single source of truth for thresholds, sizes, seeds, paths
data/build/NN_<name>.py  # one script per pipeline step
data/build/common.py     # shared helpers (config, NFC, dropflow, JSONL streaming, logging)
data/build/SOURCES.md    # provenance of every downloaded file
data/raw/                # downloads (gitignored)
data/interim/            # intermediate parquet, logs, dropflow.jsonl (gitignored)
data/interim/logs/       # one log file per step
data/prompts/            # generated prompt JSONL
data/concepts.tsv        # final released concept set
data/agreement.md        # data-flow table, κ, stratum sizes
data/LICENSES.md         # per-source licensing
tests/                   # pytest
Makefile                 # one target per step + `all`
```

Do not create new top-level directories without being asked.

---

## 4. Pipeline map

Each step reads only the outputs of earlier steps (or `data/raw`) and writes to `data/interim` unless stated otherwise.

| Step | Script | Purpose | Main output | Network |
|---|---|---|---|---|
| 00 | scaffold | Repo, config, helpers | — | no |
| 01 | `01_download.py` | Fetch and hash all sources | `data/raw/*`, `SOURCES.md` | **yes** |
| 02 | `02_inspect.py` | Read-only schema diagnostics | stdout only | no |
| 03 | `03_pool.py` | Five-way candidate pool from EN translation tables + VI entry join | `03_pool.parquet` | no |
| 04 | `04_attest.py` | Cross-source attestation (MUSE, Wikidata) — filter 1 | `04_attested.parquet` | **yes** (Wikidata, cached) |
| 05 | `05_split.py` | Held-out few-shot and direction pools | `05_*.parquet` | no |
| 06 | `06_filter.py` | Canonical form; filters 3 (polysemy), 4 (surface), 5 (proper noun) | `06_filtered.parquet` | no |
| 07 | `07_backtranslate.py` | NLLB round-trip — filter 2 | `07_bt.parquet` | model download only |
| 08 | `08_etymology.py` | Sino/native/ambiguous tagging, Cohen's κ | `08_etym.parquet` | no |
| 09 | `09_covariates.py` | Frequency, syllables, concreteness | `09_cov.parquet` | no |
| 10 | `10_tokens.py` | Token counts per model tokenizer; fertility (E1 data) | `10_tok.parquet`, `e1_fertility.csv` | tokenizer download only |
| 11 | `11_diacritics.py` | Diacritic-stripped variants, collapse detection | `11_diac.parquet` | no |
| 12 | `12_prompts.py` | Translation / repetition / cloze prompts | `data/prompts/*.jsonl` | no |
| 13 | `13_direction_data.py` | Data for language-direction estimation | `data/prompts/directions_*.jsonl` | no |
| 14 | `14_report.py` | Final assembly and reports | `concepts.tsv`, `agreement.md`, `LICENSES.md` | no |

**Checkpoints:** after steps **02, 06, 08, 12** a human reviews the output before work continues. At a checkpoint, finish the step, print the REPORT BACK items, commit, and **stop**. Do not start the next step, and do not "pre-implement" later steps.

A checkpoint that has been reviewed and approved does not block later re-runs of that step when a task explicitly instructs the re-run. Checkpoints still block starting a NEW step beyond the latest approved checkpoint unless the task says otherwise.

---

## 5. Coding conventions (apply to every script)

1. **Config-driven.** Every threshold, size, seed, path, model ID and revision is read from `configs/data.yaml`. No magic numbers in code. If a task needs a new parameter, add it to the config with a comment.
2. **CLI.** Every script takes `--config configs/data.yaml` and is runnable as `uv run python data/build/NN_name.py --config configs/data.yaml` and via its Makefile target.
3. **Idempotent and deterministic.** Re-running with the same inputs and config produces byte-identical outputs. Sort outputs by a stable key (usually `concept_id`). Use `numpy.random.default_rng(config.seed)`; never touch global random state; never depend on dict or set iteration order for output ordering.
4. **Unicode.** Normalize every string to NFC on read. All files are UTF-8.
5. **Streaming.** Iterate large JSONL line by line. Parquet (pyarrow) for all intermediate tables.
6. **Dropflow logging.** Every filter stage appends one JSON line to `data/interim/dropflow.jsonl`:
   `{"step": "06", "stage": "filter3_polysemy", "n_in": ..., "n_out": ..., "n_out_by_pos": {...}}`.
   Re-running a step must first remove that step's previous dropflow lines, so there are no duplicates.
7. **Logging.** Log to stdout and to `data/interim/logs/NN_<name>.log`.
8. **Caching network and model calls.** Any HTTP response or model output is cached on disk, keyed by a hash of (source/model, settings, input). Re-runs must not re-query. Be polite to public endpoints: rate-limit, and send a descriptive User-Agent.
9. **Shared helpers live in `common.py`.** In particular `strip_diacritics` (NFD → remove combining marks → map `đ→d`, `Đ→D`) must exist **once** and be reused by steps 06 and 11.
10. **Pinned revisions.** Hugging Face models and tokenizers are loaded at a **commit hash** from config, never `main`. If the config says `TODO`, resolve the current hash, write it back into the config, and report it.
11. **Type hints and docstrings** on public functions. Keep scripts readable; a reviewer should be able to audit each filter rule against the spec.

---

## 6. Data-integrity rules (non-negotiable)

- **Never silently drop, fill or coerce data.** Every removal is a logged dropflow stage. Missing values stay NA. **Do not impute** (e.g. concreteness, frequency).
- **Held-out disjointness.** Few-shot, direction and test pools are disjoint by `concept_id` **and** by Vietnamese surface form. Assert this in code and in a test.
- **Filter order is fixed** as specified per step. Don't reorder filters for efficiency.
- **Unexpected data → stop.** If field names, language codes, counts or formats differ from what the task describes, stop and print a diagnostic. Do not guess a workaround. Decisions about the data belong to the human at checkpoints.
- **Do not change** thresholds, filter logic, stratum definitions or split sizes on your own initiative, even if results look bad. Report the problem instead.
- **No test-set peeking.** Nothing in the data build may use model outputs from the experiments (lens scores, alignment scores, etc.).
- **The English entries are the alignment pivot** (translation tables grouped by sense). Vietnamese-edition or other sources are used for joins and cross-checks only, unless a task says otherwise.

---

## 7. Testing

- `uv run pytest` must pass before every commit.
- Write tests for every pure function, at minimum: NFC normalization, diacritic stripping (`strip_diacritics("Đường phố") == "duong pho"`), normalized Levenshtein edge cases, Traditional→Simplified conversion, sense grouping on a toy entry, the attestation matching rule, the back-translation pass rule, both etymology signals on hand-made toy entries, collapse detection, and split disjointness.
- Tests use small hand-made fixtures in `tests/fixtures/`, never the real dumps.

---

## 8. Git

- One commit per completed step: `step NN: <short description>`.
- Never commit `data/raw/` or `data/interim/`. Do commit `SOURCES.md`, the config, final outputs in `data/`, and prompt files (unless they exceed 50 MB; if so, ask).
- Do not rewrite history or amend earlier step commits.

---

## 9. Reporting back

Each task ends with a **REPORT BACK** list. Print exactly those items, as plain text or markdown tables, so a human can paste them into a review. Also report:
- any deviation from the task spec, and why;
- any `TODO` you resolved (e.g. revision hashes) and its value;
- runtime of the step and peak memory if it is notable.

Keep reports factual. Do not interpret the results or recommend threshold changes unless asked.

---

## 10. Licensing awareness

- Wiktionary / Wiktextract: CC BY-SA (share-alike applies to the released concept set).
- `wordfreq` data: CC BY-SA.
- MUSE: reportedly non-commercial. Used as a **filter only**; its pairs must not be redistributed in `concepts.tsv` beyond a boolean flag.
- Unihan: Unicode license. CC-CEDICT: CC BY-SA.
- Brysbaert concreteness norms: check terms before redistributing values.
- FLORES+: gated, with terms of use; never commit FLORES text.

Record the license of every source in `SOURCES.md` exactly as stated at the source.

---

## 11. Open decisions (set by the human at checkpoints — do not decide these)

| Decision | Config key | Set at |
|---|---|---|
| Chinese language code / field in translation tables | `chinese.code_field` | Checkpoint A (after 02) |
| Sense-grouping rule for translations | TBD | Checkpoint A |
| Surface edit-distance threshold (final) | `filters.surface_edit_threshold` | Checkpoint B (after 06) |
| Signal B rule adjustments; H3 confirmatory vs exploratory | TBD | Checkpoint C (after 08) |
| Translation-prompt source language | `prompts.translation.src_lang` | Checkpoint D (after 12) |
| Whether cloze format is enabled, and its sentence source | `prompts.cloze.*` | Checkpoint D |
| Model revision hashes | `models.*.revision` | Resolved in step 10, reported |

If a task needs one of these and the config still says `TODO`, stop and ask.
