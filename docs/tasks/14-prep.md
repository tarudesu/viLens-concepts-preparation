TASK 14-prep: implement (do NOT run on real data yet) steps 09b, 10b, 13b and 14.
Each must be complete, tested on fixtures, and runnable with one make target once
its input exists. If the input is missing, each script exits with a clear message.

09b (09b_concreteness.py): hash data/raw/brysbaert/* via `01_download.py --only
brysbaert`; fill concreteness + concreteness_match per the TASK 09 rules
(exact → head → none) for 11_diac.parquet AND 11_m1_ext.parquet.
10b (10b_fertility.py): per model × language, on FLORES devtest: tokens per
character, and for vi tokens per syllable. Output data/interim/e1_fertility.csv.
13b (13b_direction_flores.py): FLORES dev sentences for vi/en/zh/fr/id, the same
80/20 fit/validate split method (seeded, by sentence id).
Output data/prompts/directions_flores.jsonl. Never commit FLORES text: add the
file to .gitignore and store only its SHA-256 in agreement.md.
14 (14_release.py):
  - Call require_resolved_concreteness(); refuse while it is pending.
  - data/concepts.tsv: one row per concept (all splits + m1 extension), with a
    `split` column. Columns: every canonical form (vi, en, zh, fr, id, ru), vi_alts,
    vi_nodiac, collapsed, pos, en_sense_gloss, stratum, sigA, sigB, sigB_strict,
    sigB_relaxed, sino_via, native_strict, reading_source, n_sources, in_vi_gloss,
    in_muse (boolean only), in_wikidata, external_attested, bt_route,
    bt_pass_strict_v1, <lang>_nllb_agree, min_surface_dist, n_senses_vi,
    zipf_<lang>, n_syllables, n_chars_<lang>, concreteness, concreteness_match,
    ntok_<model>_<lang>, single_token_vi_<model>, m1_eligible_<model>,
    m1_extension, cloze_available, fewshot_set (for the few-shot concepts).
    UTF-8, sorted by concept_id, with a header comment line giving the pipeline
    commit hash.
  - data/agreement.md: the full dropflow table per split (latest run only), with
    by-POS counts; attestation shares; the MUSE syllable finding; the κ section
    (primary/strict/relaxed with CIs, reading_source share, the H3 status); strata
    tables; power; collapse rates; the E1 tokenizer table (ntok, single-token
    coverage, fertility once 10b exists); M1 set sizes; cloze status; the few-shot
    sets; and links to PROTOCOL_AMENDMENTS.md and SOURCES.md.
  - data/LICENSES.md: one row per source: license; redistributed in concepts.tsv
    (yes/no/flag-only); obligations. Proposed release license: CC BY-SA 4.0
    (Wiktionary share-alike). Flag for human review: MUSE (NC, flag only),
    Brysbaert (check terms before redistributing values), FLORES (never redistributed).
  - data/interim/qa_sample.txt: 50 random rows per stratum (seeded).
Commit as "step 14-prep: release tooling". REPORT BACK: the make targets, and
the exact message each script prints when its input is missing.
