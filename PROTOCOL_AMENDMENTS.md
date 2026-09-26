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
