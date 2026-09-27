# Protocol amendments

v1.2 (2026-09-26; pre-data; no model outputs exist at the time of this amendment)

- §4.3 filter 1: attestation = ≥2 of {EN-entry translation table, VI-entry
  English gloss, MUSE, Wikidata}. Reason: MUSE's Vietnamese side has 0/73,874
  multi-syllable entries; the v1.1 rule would have limited the concept set to
  monosyllables, undermining C1. Robustness subset: external_attested
  (MUSE or Wikidata).
- §4.4: strata are sino vs nonsino (not native). native_strict (Proto-Vietic/
  Mon-Khmer/Austroasiatic evidence or an inh template) is reported as a
  robustness subset. Reason: only ~7% of pool entries carry native-origin evidence.
- §4.2: POS limited to noun/verb/adj; English lemmas ≤ 2 words; Chinese = Mandarin
  (cmn), Simplified.
- §4.3 filter 4: threshold set to 0.45 at an empty histogram bin (0.45–0.50),
  after inspecting items around 0.30 (predominantly loans). Added a hyphenated-
  transliteration filter. Pre-declared robustness check: primary results
  re-run excluding min_surface_dist < 0.60.
- §4.3 filter 2: pass = round-trip
  whole-syllable containment (≤ +4 syllables) OR forward en→vi hit. Reason: NLLB
  hallucinates on isolated words, and exact-match round-trip failed 71% of
  monosyllables vs 47% of disyllables. Pre-declared robustness subset:
  bt_pass_strict_v1. Directions pool exempt from filter 2.
- Vietnamese spelling variants (tone placement, final i/y) are treated as equivalent; the display form is the more frequent one. If back-translation validates only an alternative form, it becomes the canonical form and is re-filtered.
- Split components and all disjointness checks use the Vietnamese spelling-equivalence key; splits were regenerated (same seed) before any model outputs existed.
- 'Model outputs' in this document means outputs of the study LLMs (Gemma,
  Qwen, Llama); NLLB preprocessing outputs are not included.
- §4.4 H3 status rule (fixed before the corrected Signal B was run): H3 is
  confirmatory iff the lower 95% CI bound of Cohen's κ(A, B) ≥ 0.60 AND the
  Holm-level minimum detectable d (sino vs nonsino, test) ≤ 0.30; otherwise H3
  is reported as exploratory.
- §4.3 filter 3 unchanged (≤ median senses). The adequacy of the M1
  single-token subset is assessed at the E1 gate (minimum 150 items per model).
- zh/fr/id NLLB agreement flags are robustness variables, not filters.
  Pre-declared robustness subset: all three agree.
- Checkpoint C outcome: primary κ(A,B) = 0.594, 95% CI [0.557, 0.631]; Holm-level
  MDE (sino vs nonsino, test) = 0.206. Under the pre-committed rule, H3 is
  EXPLORATORY. Strata frozen with the primary Signal B (test: sino 481, nonsino 706,
  ambiguous 347 excluded from H3). 25.8% of verified Han strings used Wiktionary
  character entries for readings (Unihan kVietnamese coverage gaps); Signal B is
  therefore only partly independent of Signal A. No further changes to the
  etymology rules will be made.
- E1 tokenizer audit (no model weights loaded): single-token Vietnamese coverage
  is 7.0% (Gemma), 6.3% (Qwen), 6.3% (Llama); all single-token items are
  monosyllabic. M1 adequacy (≥150) is not met by the main set. Planned remedy (main
  dataset unchanged): an M1-only extension set of monosyllabic concepts with
  exactly 2 senses that pass all other filters, used solely for the M1-vs-M2
  comparison (C1).
- M1 set = single-token items of the main test set ∪ M1-only extension (monosyllabic, exactly 2 senses, all other filters passed). Sizes: Gemma 186, Qwen 166, Llama 162 (final counts in agreement.md). M1-vs-M2 is also reported on main-set items only.
- Russian forms are stored without stress marks; ё/е are treated as equivalent when matching. Few-shot eligibility: ru canonical with NLLB agreement and filter-2 pass (collapse status is irrelevant, because few-shot examples are never diacritic-stripped).
- Checkpoint D outcome: Russian is the translation-prompt source; templates are unquoted; cloze examples come from Vietnamese Wiktionary examples. Cloze coverage is 337 test concepts (primary; threshold 300). Few-shot selection uses 30 examples in six sets of five, with set 1 primary.
- E9 nodiac condition: all Vietnamese text in the prompt and the Vietnamese target are diacritic-stripped (simulating undiacritized input); items whose stripped test form collides with another word are excluded. The cloze primary/supplementary status is determined on the main test set only.

### Data build complete (2026-09-26)

- Frozen data sizes: main test = 1,535; M1-only extension = 123; directions = 132; few-shot = 30 (six sets of five). `concepts.tsv` has 1,837 rows including the extension.
- Checkpoint C: primary κ(A, B) = 0.594470, 95% CI [0.557133, 0.631475]; H3 is exploratory under the pre-committed rule. Test strata: sino = 481, nonsino = 706, ambiguous = 347, other_loan = 1.
- Cloze is supplementary: 265/1,535 main-test concepts (17.26%) have a qualifying cloze example.
- E1 single-token Vietnamese coverage on the 1,535-item main test set: Gemma = 108 (7.04%); Qwen = 96 (6.25%); Llama = 96 (6.25%). M1 eligible counts including the extension: Gemma = 186; Qwen = 166; Llama = 162.
- FLORES+ devtest fertility (tokens/character; Vietnamese tokens/syllable):

  | model | vi | en | zh | fr | id |
  |---|---:|---:|---:|---:|---:|
  | Gemma | 0.268329 / 1.210382 | 0.205462 | 0.676101 | 0.242200 | 0.219199 |
  | Qwen | 0.286035 / 1.290252 | 0.209304 | 0.640515 | 0.276869 | 0.298768 |
  | Llama | 0.272802 / 1.230561 | 0.205894 | 0.804846 | 0.275771 | 0.293554 |

- Concreteness matching across the release: exact = 1,653/1,837 (89.98%); head = 135/1,837 (7.35%); none = 49/1,837 (2.67%). Counts and within-group rates by split/scope and stratum:

  | split/scope | stratum | n | exact | head | none |
  |---|---|---:|---:|---:|---:|
  | fewshot_reservoir | ambiguous | 14 | 13 (92.86%) | 1 (7.14%) | 0 (0.00%) |
  | fewshot_reservoir | nonsino | 17 | 13 (76.47%) | 4 (23.53%) | 0 (0.00%) |
  | fewshot_reservoir | sino | 16 | 14 (87.50%) | 2 (12.50%) | 0 (0.00%) |
  | directions | ambiguous | 34 | 30 (88.24%) | 4 (11.76%) | 0 (0.00%) |
  | directions | nonsino | 57 | 48 (84.21%) | 6 (10.53%) | 3 (5.26%) |
  | directions | other_loan | 1 | 1 (100.00%) | 0 (0.00%) | 0 (0.00%) |
  | directions | sino | 40 | 38 (95.00%) | 1 (2.50%) | 1 (2.50%) |
  | main_test | ambiguous | 347 | 323 (93.08%) | 13 (3.75%) | 11 (3.17%) |
  | main_test | nonsino | 706 | 594 (84.14%) | 94 (13.31%) | 18 (2.55%) |
  | main_test | other_loan | 1 | 1 (100.00%) | 0 (0.00%) | 0 (0.00%) |
  | main_test | sino | 481 | 460 (95.63%) | 9 (1.87%) | 12 (2.49%) |
  | m1_extension | ambiguous | 27 | 24 (88.89%) | 1 (3.70%) | 2 (7.41%) |
  | m1_extension | nonsino | 68 | 67 (98.53%) | 0 (0.00%) | 1 (1.47%) |
  | m1_extension | other_loan | 2 | 2 (100.00%) | 0 (0.00%) | 0 (0.00%) |
  | m1_extension | sino | 26 | 25 (96.15%) | 0 (0.00%) | 1 (3.85%) |

### v1.2 corrections (2026-09-27, before any study-LLM run)

- Correction to the Checkpoint D line: the 337 cloze count included the M1
  extension. On the main test set, cloze coverage is 265/1,535; cloze is
  SUPPLEMENTARY (see step 12b).
- §4.3 filter 4 (recorded late; applied since step 06): surface distance is
  normalized Levenshtein after removing diacritics, spaces, hyphens and
  apostrophes, with Vietnamese 'ph'→'f'. An etymology-based loanword filter
  drops borrowings from European languages and Malay/Indonesian
  (bor/bor+/lbor/der/der+/obor templates); calques and Sinitic, Sino-Japanese
  and areal/proto-language derivations are retained.
- §7.1 log_freq = Vietnamese Zipf frequency in the primary H3 model; a
  pre-declared sensitivity run adds the readout language's own Zipf frequency
  (e.g. Chinese for LA_zh). Matched-subsample check (§4.5): nearest-neighbour
  matching without replacement, caliper 0.2 SD on the four §4.5 covariates
  (Vietnamese frequency, syllables, concreteness, mean Vietnamese token count).
  Reason: the pre-model data analysis found sino vs nonsino imbalance of
  d = +0.89 (Vietnamese frequency), +0.83 (Chinese frequency), −0.83
  (concreteness), −0.62 (Vietnamese tokens).

### v1.3 cloze rebuild (2026-09-27; before any study-LLM run)

- **Decision:** v1.3 includes the rebuilt cloze files as supplementary,
  descriptive data, before any study-LLM run. The v1.2 cloze files remain at
  tag `v1.2` for provenance and are not used by any analysis.
- **Evidence from the v1.2 audit:** 239 of 337 queries have four or fewer
  tokens; 11 prompt strings are shared by 30 concepts with different targets;
  four queries are multiline; five queries repeat the target unblanked; four
  queries contain bracket markup; and the third demonstration is defective
  (`console thuộc ___ thứ 3` → `thế hệ`).
- **First rebuild and upper bound:** the initial stricter-rule run yielded 61
  main concepts and five extension concepts, with main coverage of 6.0% sino
  (29/481) and 2.8% nonsino (20/706); the other 12 main survivors were
  ambiguous. The 61 main count was an upper bound, not the final v1.3 main
  count. The initial five extension concepts were not an upper bound: fixing
  NLLB truncation allowed three additional extension concepts to pass.
  The audit found that examples for items 15 (`trinh nữ`) and 16
  (`phương pháp`) include citations and English translations, exposing the
  English targets in the prompts and making C7 pass trivially. Items 4
  (`chỉ`/point), 19 (`tượng đài`/monument), and 22 (`hội chứng`/syndrome)
  passed C7 although the reported NLLB output lacked the target; NLLB outputs
  were also truncated mid-sentence. Item 8 (`hôm qua`) is a multiline verse
  that passed C2 after newlines were joined. C6's one-neighbour check lets
  fixed multi-word terms pass, including items 12 (`hội đồng`) and 22
  (`hội chứng`).
- **Rebuild rules:** C1 uses only Wiktionary usage-example `text` values from
  aligned senses, excluding examples with `ref` and quotations; it never joins
  citation or translation fields. C2 rejects raw line breaks before cleanup.
  C4 counts letter-bearing tokens plus the blank and requires six. C5b rejects
  letter-bearing non-Vietnamese query tokens. C6 searches contiguous
  headword windows containing the fill and at least one neighbouring
  syllable, up to the longest headword. C7 excludes token-limit hits and
  requires both a match on the untruncated top beam and a match on at least
  three of the five returned beams, using the target lemma, permitted
  two-word head lemma, or a same-POS WordNet lemma. The truncation fix
  re-admitted `tỉnh` (`3f42cf2315a1`), `gián` (`7d62efb774af`), and
  `tịch thu` (`a734dc647ae3`).
- **Beam-agreement correction:** `mục tiêu` (`fec047a4ea9f`, English `goal`)
  had passed on its top output, “The chessmen missed the first goal,” while
  the other four outputs said “shot” or “target.” The sentence uses the
  physical-target sense, so the candidate fails the three-of-five agreement
  rule (one of five beams matches). After this correction, C1–C8 plus beam
  agreement are frozen for v1.3.
- **C8 context rule:** C8 is displayed last in the funnel but is computed
  against all candidates that pass C3 with a valid blank, before C4–C7. It
  rejects identical queries with different Vietnamese fills; identical
  Vietnamese fills are sense collisions and are left for C7. For example,
  `xã hội chủ nghĩa` (`a5fcf428ddad`) is rejected because its query also
  occurs for `chủ nghĩa xã hội` (`f22472e471f9`), even though the latter
  candidate fails C7.
- **Final candidate funnel:** counts are example candidates, not concepts.

  | Stage | Main | Extension |
  |---|---:|---:|
  | Candidates | 462 | 199 |
  | After C1 | 382 | 93 |
  | After C2 | 351 | 92 |
  | After C3 | 336 | 92 |
  | After C4 | 67 | 9 |
  | After C5 | 59 | 9 |
  | After C6 | 58 | 9 |
  | After C7 | 36 | 7 |
  | After C8 | 35 | 7 |

- **Final concept counts and coverage:** the diac files contain 34 main and
  seven extension concepts (41 total); the nodiac files contain 29 main and
  no extension concepts. Main coverage is 34/1,535 (2.21%), with 17/481
  sino (3.53%) and 10/706 nonsino (1.42%); the remaining seven main concepts
  are ambiguous. Diac counts by stratum are main: 17 sino, 10 nonsino, 7
  ambiguous; extension: 2 sino, 3 nonsino, 2 ambiguous. Nodiac counts are
  main: 15 sino, 8 nonsino, 6 ambiguous; extension: zero in each stratum.
- **Demonstrations:** `1a9ec27bf14d` from few-shot set 1, query
  `___ không giải quyết được gì cả.` / answer `bạo lực`; `8fa44725013f` from
  few-shot set 5, query `Mới mưa xong nền ___, coi chừng té.` / answer `ướt`;
  and `0a52c03dec8c` from the directions split, query
  `Làm theo ___, hưởng theo nhu cầu` / answer `năng lực`. All three pass the
  rules and have `collapsed == false`.
- Cloze is supplementary and descriptive only. It is excluded from H1/H2/H3
  decision rules, Holm families, the §7.1 mixed model, and all stratum
  analyses; it is reported per model with bootstrap confidence intervals.
  The frozen `cloze_available` column refers to v1.2 and is deprecated. In
  v1.3, cloze availability means membership in the new prompt files. The
  v1.3 files are `data/prompts/v1.3/cloze_diac_set1.jsonl` and
  `data/prompts/v1.3/cloze_nodiac_set1.jsonl`, with the v1.2 schema plus
  `demo_concept_ids` and `fewshot_set: null`; collapsed items are omitted from
  nodiac.
- The builder and historical first-run evidence are in
  `data/build/cloze_v13.py` and `docs/data-archive/cloze_v13/`. Final-run
  evidence is archived as `docs/data-archive/cloze_v13/cloze_v13_rebuild_report.json`,
  `docs/data-archive/cloze_v13/cloze_v13_rebuild_candidate_failures.jsonl`,
  and `docs/data-archive/cloze_v13/cloze_v13_rebuild_dropped.csv`.
