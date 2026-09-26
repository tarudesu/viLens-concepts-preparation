# Pre-registration v1.2

## Frozen protocol text (v1.1)

## 2. Hypotheses `[PREREG]`

### H1 — High-resource pivot

Middle-layer representations of a Vietnamese prompt occupy a region aligned with a high-resource language (English by default, Chinese for Qwen), before moving into a Vietnamese-specific region in late layers.

**Predicts:**
- Normalized language-alignment to the pivot language rises well above null in middle layers.
- In the five-way readout, the pivot language's *relative rise* substantially exceeds that of the control languages.
- Steering along the Vietnamese direction has a sharply localized flip-rate peak.

### H2 — Language-neutral concept space

There is no English stage. Middle layers hold an abstract representation roughly orthogonal to all language-specific directions. The logit lens decodes to English merely because the unembedding is English-shaped.

**Predicts:**
- Alignment near null for *all* languages in middle layers, while semantic decodability is already high.
- In the five-way readout, **all** language versions of the correct concept rise together.
- Tuned lens substantially reduces the apparent English dominance.

### H3 — Etymology-conditional routing (Qwen)

Qwen routes Sino-Vietnamese vocabulary through Chinese and native Vietnamese vocabulary through English. Gemma and Llama show no such split.

**Predicts:**
- Qwen: Chinese alignment significantly higher for the Sino stratum than the native stratum, after covariate adjustment.
- Gemma and Llama: no significant stratum difference.

### 2.1 Decision rules `[PREREG]`

Let `LA_L(ℓ)` be the normalized language-alignment score (§5.3), scaled so 0 = random direction, 1 = a native prompt in language `L`.

Let `rise_L(ℓ)` be the within-language relative rise in readout score (§5.2), measured in that language's own z-units from its early-layer baseline. **Relative rise, not absolute score, is the comparison quantity** — see §5.2 for why.

| Outcome | Rule |
|---|---|
| Support H1 | `max_ℓ LA_en(ℓ) ≥ 0.50` on Vietnamese prompts, **and** `max_ℓ rise_en(ℓ) ≥ 2 × mean(max_ℓ rise_fr, max_ℓ rise_id)` |
| Support H2 | `max_ℓ LA_L(ℓ) < 0.20` for all L, **and** all five `max_ℓ rise_L` fall within a 1.5× band |
| Support H3 | Qwen: `LA_zh(Sino) − LA_zh(native) > 0`, bootstrap 95% CI excluding zero; Gemma and Llama CIs include zero |
| Inconclusive | Any pattern between the H1 and H2 bands — report as such, do not reframe post hoc |

All thresholds frozen at v1.1. Revision requires a version bump documented in §14.

---

### 5.2 M2 — Multi-token logit lens (the instrument)

For target string `t = t₁…t_k` in language `L`:

1. Teacher-force the full target after the prompt.
2. At each layer `ℓ`, apply the final norm and unembedding to the residual stream at each target position.
3. Compute `log P_ℓ(tᵢ | prompt, t₁…tᵢ₋₁)` for each `i`.
4. Score for language `L` at layer `ℓ` = mean over `i`.

One forward pass per (prompt, target-language) pair.

#### `[CORRECTION vs v1.0]` Cross-language scores are not directly comparable

**This was a real flaw in v1.0 and it would have broken E4.** Chinese targets are typically 1–2 tokens; French and Indonesian are multi-token; Vietnamese is multi-token. Mean log-probability per token is systematically affected by target length, tokenizer fertility, and unigram priors. Comparing raw M2 scores *across languages* therefore measures tokenization economics as much as internal representation.

**Fix — three layers of normalization, applied in this order:**

1. **Frequency normalization.** Divide out each language's marginal token probability under the unembedding.
2. **Within-language standardization.** Convert each language's layer curve to z-units using that language's own mean and SD across layers. This makes the curve *shape* comparable even when levels are not.
3. **Baseline-relative rise.** Define `rise_L(ℓ) = z_L(ℓ) − z_L(ℓ_early)`, where `ℓ_early` is a fixed early-layer reference (first 15% of depth). **This is the primary statistic for all cross-language comparisons**, including the H1/H2 decision rules in §2.1.

Report raw scores in the appendix for transparency, but never use them for cross-language claims. Within-language comparisons across layers, models or strata may use the frequency-normalized score directly.

### 5.6 Landmark definitions `[PREREG]`

`[CORRECTION vs v1.0]` v1.0 defined `L_concept` by a fixed absolute probability threshold. That is not comparable across models with different vocabulary sizes and calibration. Replaced with a scale-free relative definition.

| Landmark | Definition |
|---|---|
| `L_concept` | First layer where the correct concept in *any* readout language reaches **10% of its own final-layer probability** |
| `L_peak` | Layer of maximum `rise_L` among non-Vietnamese readout languages |
| `L_cross` | First layer after `L_peak` where `rise_vi` exceeds the pivot language's `rise` |
| Pivot width | `(L_cross − L_concept)` as a fraction of total depth |
| Pivot magnitude | `max_ℓ rise` for the leading non-Vietnamese language |

The 10% figure is frozen. Report sensitivity at 5% and 20% as a robustness check.

---

## 7. Analysis plan `[PREREG]`

### 7.1 Primary model

```
rise ~ language * depth_spline + stratum * language
       + log_freq + n_tokens + n_syllables + concreteness
       + (1 | concept) + (1 | task_format)
```

Mixed-effects, concept as random intercept. The same concepts appear across languages and conditions; treating observations as independent will inflate significance.

### 7.2 Uncertainty

Bootstrap over **concepts**, 1,000 resamples. 95% CIs on every landmark and every decision-rule quantity.

### 7.3 Multiple comparisons

Families: (a) H1/H2 tests across 3 models × 5 languages; (b) H3 tests across 3 models × 2 strata. Holm–Bonferroni within each family. Family definitions frozen here.

### 7.4 Power

With ~1,200 concepts and bootstrap over concepts, the H1/H2 contrasts are amply powered. **H3 is the binding constraint** — its power depends on the `sino` and `native` stratum sizes after filtering, which are unknown until §4 runs. After building the concept set and before running E6, compute and report the minimum detectable stratum difference. If it exceeds a plausible effect size, declare H3 exploratory rather than confirmatory.

### 7.5 Robustness checks

- E4 repeated with tuned lens on at least one model.
- All three task formats reported separately — a finding holding for only one format is a format artifact.
- Results with and without frequency normalization.
- `L_concept` at 5%, 10%, 20% thresholds.
- Prompt-matched vs FLORES-based language directions.

---

## Protocol amendments (v1.2)

Note: amendments override v1.1 where they conflict.

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

## Frozen data pointer

- `data/concepts.tsv` SHA-256: `6ed1faf4210e43f7e37058bf975573d43ed76613897b8e6b5d609524546d4de8`
- Pipeline commit: `0629a4519e7cadfd61d370156ca2b76fd926cb8d`
- Data-build-complete amendment commit: `36d88c0ebdeb06161b4ebf1e99326a663b43cde9`
- concepts.tsv was generated at pipeline commit b6ecb6f (the header line) and reproduced byte-for-byte at 0629a45 after the Makefile order fix.

## Protocol amendments (continued)

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
