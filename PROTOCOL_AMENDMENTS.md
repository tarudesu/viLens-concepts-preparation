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

### v1.3 cloze prompt correction (2026-09-27; before any study-LLM run)

- Cloze examples are restricted to the Vietnamese Wiktionary sense whose
  English gloss matches the aligned English lemma using the existing
  `in_vi_gloss` whole-word rule; when no sense matches, only single-sense
  entries are eligible (C1). Examples are NFC-normalized, whitespace-cleaned
  to one line (C2), and must contain the canonical Vietnamese form exactly
  once, case-insensitively, on syllable boundaries (C3). Queries require at
  least six whitespace tokens including the blank (C4), exclude `()[]/|~`
  (C5), and are rejected when the blank with adjacent syllables forms another
  Vietnamese Wiktextract headword (C6). A filled query must translate through
  the pinned NLLB-200 distilled-600M decode and match a spaCy lemma, the head
  lemma for a two-word English lemma, or a WordNet synonym (C7). Candidate
  queries shared by distinct concepts are removed (C8). Example selection is
  deterministic: shortest with at least eight tokens, otherwise longest,
  with string-order tie-breaking.
- Cloze demonstrations are selected from few-shot set 1, preferring the
  existing `nghiên cứu` and `bạo lực` concepts; set 2 is the seeded fallback.
  Nodiac cloze excludes collapsed forms and strips diacritics from all
  Vietnamese prompt text and the Vietnamese target.
- The v1.2 audit was reproduced (337 records: 265 main and 72 extension;
  239 queries of four or fewer tokens, median 3; 11 repeated query strings
  across 30 concepts; four multiline, five unblanked-target, and four
  parenthesized queries; demo 3 was `console thuộc ___ thứ 3`). The v1.3
  candidate audit yielded 61/1,535 main and 5/123 extension candidates, below
  the pre-set 100-main-item decision threshold; only one of three required
  demonstrations qualified after checking set 2. The v1.2 prompt files remain
  in place pending the human decision on whether to retain cloze.
