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
