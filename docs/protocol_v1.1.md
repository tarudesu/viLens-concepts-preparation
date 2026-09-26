# The Latent Language of Vietnamese Prompts: Where and When Do LLMs Translate Back?

**Research protocol — version 1.1**
**Status:** pre-experiment. Sections marked `[PREREG]` should be frozen before any data is collected.

**Changes from v1.0:** see §14. Two substantive methodological corrections (§5.2, §5.6) and a fully automated data pipeline replacing manual annotation (§4).

---

## 1. Overview

### 1.1 Core question

When a multilingual LLM is prompted in Vietnamese, does it internally represent the answer in a high-resource language (English, or Chinese for Chinese-heavy models) before converting back to Vietnamese? If so, **at which layers** does the shift happen, **which components** perform it, and **how wide** is the non-Vietnamese band?

### 1.2 Why Vietnamese

1. **Syllable-level whitespace.** Most content words are multi-syllable (`máy tính`, `bệnh viện`, `nghiên cứu`) and therefore multi-token under every subword tokenizer. The standard latent-language protocol requires single-token unambiguous targets, so it can only see a biased sliver of the Vietnamese lexicon.
2. **Heavy Sino-Vietnamese vocabulary, in Latin script.** Vietnamese has a large Hán-Việt layer with Chinese etymological cognates, written in Latin script. This isolates the etymology effect from the script confound that Japanese and Korean carry.
3. **Diacritics are optional in practice.** Undiacritized Vietnamese is ubiquitous online, giving a free perturbation condition.

### 1.3 Contributions claimed

- **C1.** A multi-token latent-language measure that removes the single-token restriction, validated against the standard measure on the subset where both apply.
- **C2.** A five-way readout test that discriminates an English pivot from a language-neutral concept space — a distinction the logit lens alone cannot make.
- **C3.** Causal localization of the "translate-back" step to specific layers and components, via a steering sweep.
- **C4.** Evidence on whether pretraining language composition interacts with lexical etymology.
- **C5.** A fully automated, reproducible, five-way aligned and etymology-tagged Vietnamese concept set, built entirely from existing open resources.

---

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

## 3. Models

| Alias | Checkpoint | Notes |
|---|---|---|
| `gemma` | `google/gemma-3-4b-it` | Multimodal wrapper — hook `model.language_model`. Uses interleaved local/global attention; see §5.4 caveat |
| `qwen` | `Qwen/Qwen3-4B` | Set `enable_thinking=False`; we want short direct continuations |
| `llama` | `meta-llama/Llama-3.2-3B-Instruct` | Text-only, simplest to hook |

Optional scale check: `Qwen/Qwen3-1.7B`.

**Do not hardcode layer counts.** Read `num_hidden_layers` from config (`config.text_config` for Gemma 3) and report everything at normalized depth `ℓ = layer / L_total`.

**Mandatory pre-check (E1).** Verify final-layer logit lens reproduces the true output distribution per model. Gemma's final-layer handling in particular must be confirmed. Failing models get tuned-lens probes (`tuned-lens` package, a few GPU-hours on a small corpus) or are dropped with a stated reason.

---

## 4. Data — fully automated pipeline

**No manual annotation.** Every field is derived from existing open resources, with cross-source agreement replacing human validation. This is more reproducible than a two-annotator check and removes the critical path from the timeline.

### 4.1 Sources

| Field | Source | Access |
|---|---|---|
| Five-way aligned concepts | Wiktextract (kaikki.org) translation tables, sense-disambiguated | JSON dump |
| Cross-check alignment | Wikidata labels joined by QID | SPARQL / dump |
| Cross-check alignment | MUSE bilingual dictionaries (vi–en) | GitHub |
| Sino-Vietnamese tagging | Wiktextract Vietnamese etymology categories and templates | Same dump |
| Sino-Vietnamese cross-check | Unihan `kVietnamese` field (Hán-Việt readings of Han characters) | Unicode Consortium |
| Word frequency, all 5 languages | `wordfreq` package | pip |
| Concreteness (for stratum matching) | Brysbaert et al. English concreteness norms, mapped via English gloss | Public dataset |
| Polysemy (ambiguity proxy) | Sense count in Wiktextract | Same dump |
| Language ID for steering outcomes | GlotLID or fastText `lid.176` | pip / download |

All are open-licensed. Record licenses in the repo; Wiktionary content is CC BY-SA, which affects how the derived concept set can be released.

### 4.2 Concept selection

**Target:** 1,200–1,500 concepts after filtering.

**Readout languages:** Vietnamese (`vi`), English (`en`), Chinese (`zh`), French (`fr`), Indonesian (`id`).

- `en` and `zh` are the candidate pivots.
- `fr` and `id` are **controls**: not plausible pivots for any of these models. Under H1 they stay flat while English rises; under H2 they rise along with everything. `id` is the stricter control — Latin script, areal proximity to Vietnamese.

### 4.3 Automated filters

Drop a concept if:

1. **Source disagreement.** The vi–en pair is not attested in at least **two** independent sources (Wiktextract, Wikidata, MUSE).
2. **Back-translation failure.** Round-trip vi→en→vi through an open MT model (e.g. NLLB-200) does not return the original lemma. Automated, no human needed.
3. **High polysemy.** Wiktextract sense count above the median for that part of speech.
4. **Surface overlap.** Vietnamese and English forms are identical or differ by less than an edit-distance threshold (catches loanwords and international vocabulary).
5. **Proper noun**, by Wiktextract POS tag.
6. **Missing** in any of the five readout languages.

Report the drop count at each stage as a data-flow table in the appendix.

### 4.4 Automated etymology stratification

Three-way tag, from two independent automated signals:

- `sino` — Wiktextract marks Sino-Vietnamese origin **and** all syllables have Hán-Việt readings attested in Unihan `kVietnamese`.
- `native` — Wiktextract indicates native origin **and** no syllable has an attested Hán-Việt reading.
- `ambiguous` — the two signals disagree, or evidence is absent.

**Report inter-source agreement (Cohen's κ between the Wiktextract signal and the Unihan signal) as the quality metric.** This is the automated replacement for inter-annotator agreement and should be stated in the paper.

**Drop the `ambiguous` stratum from primary analysis** and report its size. Forcing a binary manufactures noise.

### 4.5 Covariate adjustment for H3

Sino and native strata differ systematically in frequency, length and abstractness. Rather than matching (which discards data), include as covariates in the primary regression (§7.1): log frequency, token count per tokenizer, syllable count, concreteness rating.

Additionally report a **matched-subsample robustness check** using nearest-neighbour matching on those four covariates. If the covariate-adjusted and matched estimates disagree, report both and treat H3 as unresolved.

### 4.6 Prompt formats

Three formats, following the established protocol for comparability:

1. **Translation** — few-shot `xx: <word> → vi: <word>`, then the test item.
2. **Repetition** — few-shot `<word> → <word>`, then the test item.
3. **Cloze** — Vietnamese sentence with the target masked at the final position.

Few-shot examples fixed across all conditions, drawn from a held-out pool never used as test items.

### 4.7 Auxiliary data

- **FLORES-200** vi/en/zh/fr/id parallel sentences → language directions, plus prompt-format-matched direction estimation (§5.3).
- **Diacritic-stripped variants** → generated automatically for E9. Record the *collapse rate* (how many distinct words become identical when stripped) and exclude collapsed items from primary analysis.
- **Belebele-vi**, **VMLU** → optional downstream checks.

---

## 5. Measurements

### 5.1 M1 — Single-token logit lens (replication)

Standard protocol, restricted to concepts whose Vietnamese form is a single token. Purpose: comparability with prior work and a validation target for M2.

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

### 5.3 M3 — Language directions (token-free)

For each language `L`, compute `v_L^(ℓ)` per layer as the difference of means between activations on `L` text and activations on all other languages.

**`[REFINEMENT vs v1.0]` Directions must be estimated on prompt-format-matched data.** v1.0 estimated directions on FLORES sentences and applied them to short word-completion prompts — a domain mismatch that could bias the alignment scores. Estimate directions on the same prompt templates used in §4.6, populated with held-out vocabulary. Retain the FLORES-based directions as a robustness check and report both.

**Validation gate:** each `v_L` must classify held-out prompts by language at >95% accuracy before use. If not, the direction is not measuring language and the analysis stops.

Normalized alignment:

```
LA_L(ℓ) = ( cos(h_ℓ^vi-prompt, v_L) − c_random ) / ( cos(h_ℓ^L-prompt, v_L) − c_random )
```

`c_random` = mean cosine to random unit directions of matched norm, per layer.

This measurement adjudicates H1 vs H2. The logit lens cannot, because it reads through an English-shaped unembedding by construction.

### 5.4 M4 — Component attribution (the "where")

Decompose each layer's residual update into attention output and MLP output; project each onto `v_vi`. Report which component type, and which specific heads or MLP layers, drive the Vietnamese-ward shift between `L_peak` and `L_cross`.

**Gemma 3 caveat:** interleaved local and global attention layers mean head-level results are not directly comparable to Llama or Qwen. Report Gemma's attention attribution separately for local and global layers, or restrict head-level claims to Llama and Qwen.

### 5.5 M5 — Steering sweep (the causal "when")

At layer `ℓ`, add `α · (v_vi − v_en)` to the residual stream and generate ~20 tokens. Sweep `ℓ` across all layers and `α` across three magnitudes.

**Outcome measure:** language of the generated continuation, classified automatically by GlotLID or fastText `lid.176`. No manual reading required.

**The peak of the flip-rate curve is the causal location of the language decision.** If it coincides with `L_cross` from M2, the correlational and causal accounts agree — the strongest result available from this project.

Also report a **fluency guard**: perplexity of the steered continuation. A layer where steering flips the language but destroys fluency is not evidence of a language-decision site.

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

## 6. Experiments

### E0 — Pipeline pilot

**Duration:** 3 days. 50 concepts, one model, end to end through E5. Purpose: catch bugs, confirm hook access, confirm runtimes, confirm output formats. Do not analyze; discard results.

### E1 — Tokenizer and lens audit `[GATE]`

**Duration:** 1 week. **Go/no-go gate.**

1. Fertility (tokens per character, tokens per syllable) for all five languages per tokenizer.
2. Fraction of the Vietnamese concept lexicon that is single-token per tokenizer.
3. Logit-lens validity per model: does final-layer logit lens reproduce the true output distribution?
4. Hook access on Gemma 3's multimodal wrapper.

**Gate conditions:**
- Single-token coverage >60% across all three tokenizers → C1 weakens substantially; rescope before proceeding.
- Any model failing the lens check → tuned-lens probes or drop. Decide now.

### E2 — M1 replication

Single-token protocol on the eligible subset, all models, all three task formats.

### E3 — M2 full lexicon, landmark extraction

Full concept set. Extract all five landmarks per model. Report the **M1 vs M2 delta** on matched concepts (contribution C1).

### E4 — Five-way readout `[CENTERPIECE]`

Score all five language versions at every layer using M2, with the three-stage normalization of §5.2. Compare `rise_L` curves.

Discriminates all three hypotheses at once:
- **H1** → English rises alone; fr and id stay flat.
- **H2** → all five rise together.
- **H3** → for Qwen on the Sino stratum, Chinese rises with or instead of English.

### E5 — M3 direction analysis

`LA_L(ℓ)` for all five languages, all models, on Vietnamese prompts. Apply §2.1 decision rules.

**E4 and E5 must agree.** Disagreement (lens says English, directions say neutral) supports H2 and changes the paper's framing — see §10.1.

### E6 — Etymology stratification

Rerun E4 and E5 on `sino` and `native` strata with covariate adjustment (§4.5). Test H3. Include the matched-subsample robustness check.

### E7 — Component attribution

M4 between `L_peak` and `L_cross`. Attention vs MLP split; top contributing components. Respect the Gemma caveat in §5.4.

### E8 — Steering sweep

M5 across all layers, three magnitudes, ~300 concepts, with the fluency guard. Compare flip-rate peak to `L_cross`.

### E9 — Diacritic condition *(optional)*

Rerun E3 and E5 on diacritic-stripped prompts, excluding collapsed items. Does undiacritized input widen the pivot?

### E10 — Downstream ablation *(optional, low priority)*

Language-specific neuron identification and ablation, measured on Belebele-vi and VMLU. Note this space is already well covered by prior work; include only if it strengthens a specific claim.

### Priority order

**E0 → E1 → E3 → E4 → E5 → E8.** Those are a complete paper: descriptive curves, hypothesis adjudication, causal confirmation. E6 makes it distinctive. E2, E7, E9, E10 are enrichment.

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

## 8. Figures

| # | Content | Supports |
|---|---|---|
| 1 | Fertility and single-token coverage by tokenizer, five languages | Motivation, C1 |
| 2 | M1 vs M2 landmark comparison on matched concepts | C1 |
| 3 | Five-way `rise_L` curves, one panel per model | C2, H1/H2 |
| 4 | `LA_L(ℓ)` curves with null band shaded | C2, H1/H2 |
| 5 | Sino vs native curves for Qwen, Gemma and Llama as controls | C4, H3 |
| 6 | Steering flip-rate vs layer, `L_cross` marked, fluency overlaid | C3 |
| 7 | Attention vs MLP contribution to the Vietnamese-ward shift | C3 |

Figure 3 is the paper. Figure 6 makes it causal.

---

## 9. Timeline

Shortened from 10 weeks to **8**, since automated data construction removes the annotation critical path.

| Week | Work |
|---|---|
| 1 | E0 pilot + E1 audit + gate decision |
| 2 | Automated concept set construction, filtering, etymology tagging, agreement reporting |
| 3 | E2, E3 |
| 4 | E4 |
| 5 | E5, hypothesis adjudication, power check for H3 |
| 6 | E6, E7 |
| 7 | E8 |
| 8 | Writing, robustness checks, optional E9 |

**Compute is not the constraint.** All three models run in bf16 on a single 24GB GPU; expect a few days of total GPU time. Data construction is now days, not weeks.

---

## 10. Risks and contingencies

| Risk | Likelihood | Response |
|---|---|---|
| E1 gate fails on single-token coverage | Medium | C1 weakens; C2 and C3 survive. Reframe around hypothesis adjudication and causal localization |
| Logit lens invalid on Gemma | Medium | Tuned-lens probes, or drop Gemma and substitute another family |
| E4 and E5 disagree | Medium | H2 supported. Paper becomes "the apparent English pivot is a logit-lens artifact." Abstract pre-written (§10.1) |
| Automated etymology tagging has low inter-source κ | Medium | Report κ honestly; widen the `ambiguous` bucket; downgrade H3 to exploratory |
| Sino stratum too small after filtering | Medium | Detected by the §7.4 power check before E6 runs. Declare H3 exploratory |
| No pivot in any model | Low–Medium | Publishable as evidence for language-neutral processing; lean on E5 and E8 |
| Language directions fail the 95% gate | Low | Stop and fix. Do not proceed with an unvalidated direction |
| Steering flips language only by destroying fluency | Medium | The fluency guard catches this. Report flip-rate conditional on acceptable perplexity |

### 10.1 Write both abstracts before running E4

Draft the H1-supported and H2-supported abstracts **now**, before any hypothesis-test data exists. One hour, and it removes the temptation to reinterpret ambiguous curves favourably. State in the paper that this was done.

---

## 11. Related work — positioning

Cite and explicitly differentiate:

- **Wendler et al. (ACL 2024)** — the latent-language protocol and three-phase account; the single-token constraint we relax.
- **Zhong et al. (2024)** — language-adapted models default to their dominant training language rather than English.
- **Schut et al. (2025)** — pivot rates differ by training composition.
- **Tang et al. (2024)**, **Kojima et al. (2024)** — language-specific neurons and their layer distribution.
- **Mondal et al. (2025)** — newer architectures concentrate language-specific neurons in later layers.
- **Lindsey et al. (2025)** — language-agnostic conceptual representations; the strongest statement of H2.

**Positioning sentence:** *Prior work asks whether models pivot through English. We ask two questions it could not answer: whether the pivot is real or an artifact of reading through an English-shaped unembedding, and where in the network the conversion back to the input language is causally performed.*

---

## 12. Reproducibility

```
repo/
├── configs/          # frozen hyperparameters, thresholds, seeds
├── data/
│   ├── build/        # scripts: wiktextract → concepts.tsv, fully automated
│   ├── concepts.tsv  # five-way aligned, etymology-tagged, with covariates
│   ├── agreement.md  # inter-source κ, drop-count data-flow table
│   └── LICENSES.md   # per-source licensing, incl. Wiktionary CC BY-SA
├── src/
│   ├── audit.py      # E1
│   ├── lens.py       # M1, M2 (+ three-stage normalization)
│   ├── directions.py # M3
│   ├── attribute.py  # M4
│   └── steer.py      # M5
├── results/          # raw scores, one parquet per experiment
├── analysis/         # notebooks for §7
└── PREREG.md         # frozen copy of §2, §5.6, §7 before data collection
```

Pin exact model revisions by commit hash, not tag — tags move. Fix and log all seeds. Tag the repo at `preprep-v1.1` before collecting data.

**Release the concept set.** A fully automated, five-way aligned, etymology-tagged Vietnamese resource is reusable and is contribution C5. Check the Wiktionary CC BY-SA share-alike obligation before choosing a license.

---

## 13. Open decisions

Items deliberately left unfixed until E1 and the concept build report back:

- Exact concept count (target 1,200–1,500, actual depends on filter yield).
- Whether Gemma stays in the study (depends on lens validity).
- Whether H3 is confirmatory or exploratory (depends on §7.4 power check).
- Whether E9 and E10 are run at all.

---

## 14. Changelog

### v1.1

- **Data pipeline fully automated (§4).** All manual annotation removed. Etymology tagging now comes from Wiktextract categories cross-checked against Unihan `kVietnamese`; translation validity from three-source agreement plus automated back-translation; frequency from `wordfreq`; concreteness from published norms; steering outcomes from automated language ID. Inter-source κ replaces inter-annotator agreement as the quality metric.
- **`[CORRECTION]` Cross-language M2 comparability (§5.2).** v1.0 compared raw multi-token scores across languages, which confounds tokenization economics with internal representation and would have invalidated E4. Replaced with a three-stage normalization; `rise_L` is now the primary cross-language statistic and the decision rules in §2.1 are restated in those terms.
- **`[CORRECTION]` `L_concept` definition (§5.6).** v1.0 used a fixed absolute probability threshold, not comparable across models with different vocabulary sizes. Replaced with a scale-free relative definition (10% of final-layer probability), with sensitivity analysis.
- **`[REFINEMENT]` Language directions (§5.3).** Now estimated on prompt-format-matched data rather than FLORES sentences, removing a domain mismatch. FLORES-based directions retained as a robustness check.
- **Added** E0 pipeline pilot; fluency guard on the steering sweep (§5.5); Gemma local/global attention caveat (§5.4); diacritic collapse-rate handling (§4.7); §7.4 power analysis for H3; §13 open decisions.
- **Changed** H3 stratum handling from matching to covariate adjustment with matched-subsample robustness check (§4.5).
- **Timeline** shortened 10 → 8 weeks.
- **Added** contribution C5 (the released concept set).

### v1.0

Initial protocol. Models fixed at `gemma-3-4b-it`, `Qwen3-4B`, `Llama-3.2-3B-Instruct`. Hypotheses, decision rules, landmark definitions and analysis plan frozen pending E1 results.
