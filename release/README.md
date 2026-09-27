---
license: cc-by-sa-4.0
language: [vi, en, zh, fr, id, ru]
pretty_name: "viLens Concepts"
size_categories: "1K<n<10K"
tags: [interpretability, multilingual, vietnamese, logit-lens, sino-vietnamese]
configs:
  - config_name: concepts
    data_files:
      - split: train
        path: concepts.jsonl
    default: true
  - config_name: repetition
    data_files:
      - split: diac_set1
        path: prompts/repetition_diac_set1.jsonl
      - split: diac_set2
        path: prompts/repetition_diac_set2.jsonl
      - split: diac_set3
        path: prompts/repetition_diac_set3.jsonl
      - split: diac_set4
        path: prompts/repetition_diac_set4.jsonl
      - split: diac_set5
        path: prompts/repetition_diac_set5.jsonl
      - split: diac_set6
        path: prompts/repetition_diac_set6.jsonl
      - split: nodiac_set1
        path: prompts/repetition_nodiac_set1.jsonl
      - split: nodiac_set2
        path: prompts/repetition_nodiac_set2.jsonl
      - split: nodiac_set3
        path: prompts/repetition_nodiac_set3.jsonl
      - split: nodiac_set4
        path: prompts/repetition_nodiac_set4.jsonl
      - split: nodiac_set5
        path: prompts/repetition_nodiac_set5.jsonl
      - split: nodiac_set6
        path: prompts/repetition_nodiac_set6.jsonl
  - config_name: translation
    data_files:
      - split: diac_set1
        path: prompts/translation_diac_set1.jsonl
      - split: diac_set2
        path: prompts/translation_diac_set2.jsonl
      - split: diac_set3
        path: prompts/translation_diac_set3.jsonl
      - split: diac_set4
        path: prompts/translation_diac_set4.jsonl
      - split: diac_set5
        path: prompts/translation_diac_set5.jsonl
      - split: diac_set6
        path: prompts/translation_diac_set6.jsonl
      - split: nodiac_set1
        path: prompts/translation_nodiac_set1.jsonl
      - split: nodiac_set2
        path: prompts/translation_nodiac_set2.jsonl
      - split: nodiac_set3
        path: prompts/translation_nodiac_set3.jsonl
      - split: nodiac_set4
        path: prompts/translation_nodiac_set4.jsonl
      - split: nodiac_set5
        path: prompts/translation_nodiac_set5.jsonl
      - split: nodiac_set6
        path: prompts/translation_nodiac_set6.jsonl
  - config_name: directions
    data_files:
      - split: matched
        path: prompts/directions_matched.jsonl
---

# viLens Concepts

## Summary

This release contains **1,837 aligned concepts** with canonical forms in the five readout languages (Vietnamese, English, Mandarin Chinese, French, Indonesian) and Russian as the translation-prompt source. The frozen partitions contain 1,535 main-test concepts, 123 M1-extension concepts, 132 directions concepts, and 47 few-shot-reservoir concepts.

## Construction

The English Wiktionary/Wiktextract translation tables provide the alignment pivot; Vietnamese Wiktextract entries supply etymology and gloss cross-checks. Additional attestation uses MUSE and Wikidata; Mandarin strings are checked against Unihan readings and CC-CEDICT, with WordNet used for single-syllable meaning checks. Frequency comes from wordfreq, while Brysbaert values are withheld in this package. FLORES+ is used for tokenizer fertility and a separate gated direction dataset; its sentence text is not included here.

Selection keeps noun/verb/adjective concepts with English lemmas of at most two words and requires two configured attestation sources. The pipeline applies the frozen back-translation, median-polysemy, proper-noun, surface-distance, loanword, and hyphenated-transliteration rules, then assigns the frozen sino/nonsino/ambiguous strata. The v1.2 amendments, analysis plan, provenance notes, and construction materials are maintained in the [preparation repository at the frozen `prereg-v1.2` tag](https://github.com/tarudesu/viLens-concepts-preparation/tree/prereg-v1.2).

## Fields

The `concepts` configuration is a UTF-8 JSON Lines file: each line is one concept object, and every value is stored as a string.

| Column | Meaning |
|---|---|
| `concept_id` | Stable 12-character identifier for this concept. |
| `split` | Leakage-safe partition: test, directions, or fewshot_reservoir. |
| `vi` | Canonical Vietnamese form. |
| `en` | Canonical English lemma used as the alignment pivot. |
| `zh` | Canonical simplified Mandarin form. |
| `fr` | Canonical French form. |
| `id` | Canonical Indonesian form. |
| `ru` | Canonical Russian prompt-source form; may be empty if unavailable. |
| `vi_alts` | Other surviving Vietnamese forms for the concept, stored as a JSON list. |
| `vi_nodiac` | Vietnamese canonical form with diacritics removed (đ→d). |
| `collapsed` | Whether the stripped Vietnamese form collides with another real form. |
| `pos` | Part of speech: noun, verb, or adjective. |
| `en_sense_gloss` | English gloss associated with the aligned sense. |
| `stratum` | Frozen etymological stratum: sino, nonsino, ambiguous, or other_loan. |
| `sigA` | Signal A: Wiktionary/Wiktextract etymological classification. |
| `sigB` | Primary Signal B: external Han-spelling verification classification. |
| `sigB_strict` | Signal B requiring the strict reading, dictionary, and meaning checks. |
| `sigB_relaxed` | Signal B requiring a verified Vietnamese reading and CEDICT headword. |
| `sino_via` | Recorded route/type supporting a Sino classification, when available. |
| `native_strict` | Whether the concept meets the strict native-origin evidence rule. |
| `reading_source` | Reading source used for a verified Han string, when applicable. |
| `n_sources` | Number of configured independent attestation sources for the selected Vietnamese form. |
| `in_vi_gloss` | Whether the Vietnamese entry contains the English lemma in an English gloss. |
| `in_wikidata` | Whether the Vietnamese–English pair is attested by Wikidata labels or aliases. |
| `external_attested` | Whether the pair is attested by MUSE or Wikidata. |
| `bt_route` | Back-translation pass route: strict, rt_contain, fwd_only, or none. |
| `bt_pass_strict_v1` | Whether the original strict exact-match round-trip rule passed. |
| `zh_nllb_agree` | Whether the provisional Mandarin form agrees with NLLB beam output. |
| `fr_nllb_agree` | Whether the provisional French form agrees with NLLB beam output. |
| `id_nllb_agree` | Whether the provisional Indonesian form agrees with NLLB beam output. |
| `ru_nllb_agree` | Whether the selected Russian form agrees with NLLB beam output. |
| `min_surface_dist` | Minimum normalized Vietnamese-to-en/fr/id surface edit distance. |
| `n_senses_vi` | Vietnamese sense count summed over homograph entries for the canonical form. |
| `zipf_vi` | wordfreq Zipf frequency for the canonical vi form. |
| `zipf_en` | wordfreq Zipf frequency for the canonical en form. |
| `zipf_zh` | wordfreq Zipf frequency for the canonical zh form. |
| `zipf_fr` | wordfreq Zipf frequency for the canonical fr form. |
| `zipf_id` | wordfreq Zipf frequency for the canonical id form. |
| `n_syllables` | Whitespace-separated syllable count of the Vietnamese canonical form. |
| `n_chars_vi` | Unicode codepoint count, including spaces, in the canonical vi form. |
| `n_chars_en` | Unicode codepoint count, including spaces, in the canonical en form. |
| `n_chars_zh` | Unicode codepoint count, including spaces, in the canonical zh form. |
| `n_chars_fr` | Unicode codepoint count, including spaces, in the canonical fr form. |
| `n_chars_id` | Unicode codepoint count, including spaces, in the canonical id form. |
| `concreteness_match` | Brysbaert matching route (exact/head/none); concreteness values are omitted. |
| `ntok_gemma_vi` | gemma tokenizer count for vi, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_gemma_en` | gemma tokenizer count for en, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_gemma_zh` | gemma tokenizer count for zh, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_gemma_fr` | gemma tokenizer count for fr, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_gemma_id` | gemma tokenizer count for id, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_qwen_vi` | qwen tokenizer count for vi, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_qwen_en` | qwen tokenizer count for en, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_qwen_zh` | qwen tokenizer count for zh, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_qwen_fr` | qwen tokenizer count for fr, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_qwen_id` | qwen tokenizer count for id, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_llama_vi` | llama tokenizer count for vi, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_llama_en` | llama tokenizer count for en, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_llama_zh` | llama tokenizer count for zh, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_llama_fr` | llama tokenizer count for fr, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `ntok_llama_id` | llama tokenizer count for id, measured in context as len(tok('x: '+form))−len(tok('x: ')). |
| `single_token_vi_gemma` | Whether the Vietnamese form is one token under the Gemma tokenizer. |
| `single_token_vi_qwen` | Whether the Vietnamese form is one token under the Qwen tokenizer. |
| `single_token_vi_llama` | Whether the Vietnamese form is one token under the Llama tokenizer. |
| `m1_eligible_gemma` | Gemma M1 eligibility flag for main-test or M1-extension concepts. |
| `m1_eligible_qwen` | Qwen M1 eligibility flag for main-test or M1-extension concepts. |
| `m1_eligible_llama` | Llama M1 eligibility flag for main-test or M1-extension concepts. |
| `m1_extension` | True only for concepts in the separate M1-only extension set. |
| `cloze_available` | Deprecated frozen v1.2 indicator; retained for provenance and not used by any analysis. |
| `fewshot_set` | Fixed few-shot set assignment (1–6), or empty when not selected. |

## Splits

| Split | Rows | Notes |
|---|---:|---|
| test | 1,658 | Main test 1,535 plus M1-only extension 123; `m1_extension` distinguishes the extension. |
| directions | 132 | Prompt-matched direction concepts. |
| fewshot_reservoir | 47 | Reservoir concepts; selected prompt demonstrations are identified by `fewshot_set`. |

Partitions are disjoint by concept ID, English lemma, and Vietnamese spelling-equivalence key. The test partition contains the main test set and the separately flagged M1-only extension; the extension is never part of the main test set.

## Configurations, splits, and prompt records

The two prompt formats are repetition and translation; the package also includes directions data.


| Hugging Face config | Split names | File | Rows per split |
|---|---|---|---:|
| `concepts` | `train` | `concepts.jsonl` | 1,837 |
| `repetition` | `diac_set1`–`diac_set6` | `prompts/repetition_diac_set*.jsonl` | 1,658 |
| `repetition` | `nodiac_set1`–`nodiac_set6` | `prompts/repetition_nodiac_set*.jsonl` | 1,236 |
| `translation` | `diac_set1`–`diac_set6` | `prompts/translation_diac_set*.jsonl` | 1,647 |
| `translation` | `nodiac_set1`–`nodiac_set6` | `prompts/translation_nodiac_set*.jsonl` | 1,226 |
| `directions` | `matched` | `prompts/directions_matched.jsonl` | 1,320 |

The JSONL `split` field is the frozen partition carried by each record (`test`,
`directions`, `fit`, or `validate`); it is distinct from the Hugging Face
dataset split names in the table. Concept rows also use `fewshot_reservoir` for
the reservoir partition.

Every `concepts.jsonl` field is stored as a string. Prompt JSONL records use
JSON strings, booleans, and integers. Repetition and translation records have
`concept_id`, `split`, `m1_extension`, `format`, `condition`, `fewshot_set`,
`prompt`, and `target_vi`, `target_en`, `target_zh`, `target_fr`, `target_id`;
target strings retain the leading space used by the prompt pipeline. Direction
records have `concept_id`, `format`, `lang`, `prompt`, `split`, and `target`.

## Files and checksums

SHA-256 checksums for the concept table and all released prompt data files:

| File | SHA-256 |
|---|---|
| `concepts.jsonl` | `f3d009bbec7d7e54738a889a536ba06d6fcbe023f301043f9988e11e97677fc7` |
| `prompts/directions_matched.jsonl` | `dd0e5fc8ce9366a1df7b38f8555f50e418706045bd4d36fb96acaeec5812dd83` |
| `prompts/repetition_diac_set1.jsonl` | `e9804c7b44e096a2d410d9551b8fffbeacd92f08c3f0823526d6d59d64810a08` |
| `prompts/repetition_diac_set2.jsonl` | `2d5eff20242cc5156a3ffee0dde8bf992bb96e51e8678cc1f65c2390c20ba46d` |
| `prompts/repetition_diac_set3.jsonl` | `8abfaa9ed700adf5dcb1051f5f2c44c382fee95e7980ecb7d62d849d689bd4fb` |
| `prompts/repetition_diac_set4.jsonl` | `af991f4d3bf9522c131e97a94935775caa946e1bf54d8f8dc26137c02341a10e` |
| `prompts/repetition_diac_set5.jsonl` | `1d8d2d4e22e3c3f78eaa8c8541c3f82184b855aeaef8c677c4012d4e1348710f` |
| `prompts/repetition_diac_set6.jsonl` | `a5b9ad871b8c1db495e27ab8075c01f6af527511b0360419a388801f8e89caaa` |
| `prompts/repetition_nodiac_set1.jsonl` | `9afe85547c87bca7c2b1747908cdb768f612a5690ddfb5fd24cd9df7c9f9cab2` |
| `prompts/repetition_nodiac_set2.jsonl` | `46925adfb5396f26bff09152f0d57229e1937b3cd231ddda15b64b34bae7170d` |
| `prompts/repetition_nodiac_set3.jsonl` | `0e29dcf67457f4cd8e923bd6fd95ef2b6123082d24d9fa8318a8cd3db5dd46b7` |
| `prompts/repetition_nodiac_set4.jsonl` | `f1cb68f03b010c5554f003ad9071742fb97492740f71f8e76761414987e6b3d8` |
| `prompts/repetition_nodiac_set5.jsonl` | `cfdc90ee88044b5510302bd614c92afa0eaa89c318869bd85a8807f30c7ddc61` |
| `prompts/repetition_nodiac_set6.jsonl` | `3a960ca83ec0240f486bce23b4509606d7c4f087cd658ed85a7530f4638a7753` |
| `prompts/translation_diac_set1.jsonl` | `fb894bb19e43d57c5d173f17bb9fd26492891ccfccfc01340e90743bc949ef77` |
| `prompts/translation_diac_set2.jsonl` | `54f1e8fa653d1e70413283594a9ed2033723988caec7a401c46cca06ac69ff2e` |
| `prompts/translation_diac_set3.jsonl` | `c76e6e8f05f7abc9170e424a26948815c3e9fcd26a3922dce10857e29af734a5` |
| `prompts/translation_diac_set4.jsonl` | `5d16938956de4c19173a5b2a1027076a0ae7dca7c3b7dd588e8f1c4ac046165b` |
| `prompts/translation_diac_set5.jsonl` | `8b7058df84cec38c7cf9e84241403a8ee3eebf55f06ecb81ff9f0613b905b8fc` |
| `prompts/translation_diac_set6.jsonl` | `d02b10b7d176f11693fe0f8ebb2efbadcd8c48115ded444fb5deb9b0fa2d34b4` |
| `prompts/translation_nodiac_set1.jsonl` | `fff9e0eaafae9604204368835b599e0ca0da204134feb726074d88472c71c6d6` |
| `prompts/translation_nodiac_set2.jsonl` | `7c3c3d83b00213d4bd4d768cce782f4dcc9c2b047f3e2b8150d9198f0b2e335e` |
| `prompts/translation_nodiac_set3.jsonl` | `9eceb035ca246bed95f76a70a0f8a851334f708af4ef71d364092253c18d4dbe` |
| `prompts/translation_nodiac_set4.jsonl` | `165faaabef0b5b48f19496d96363a849cb574dc8c765a3d10fcc75ae6381237f` |
| `prompts/translation_nodiac_set5.jsonl` | `cb6f46b7d226addcd7e3cd7d7f1529d6d5282d076b9c1080df86c4a594118ed8` |
| `prompts/translation_nodiac_set6.jsonl` | `2f49370f49225704aaf08d2a8affbf4d81c309c0d1941725a4da31e008fe3392` |

## Quality metrics

Primary Signal A/B Cohen's κ = **0.594470** (95% CI [0.557133, 0.631475]; n=1659); under the preregistered rule, H3 is **exploratory**. 149/577 (25.82%) of verified Han strings used Wiktionary character entries for readings, so Signal B is partly dependent on Wiktionary.

Attestation coverage among the main-test concepts:

| Source flag | Main test coverage |
|---|---|
| `in_vi_gloss` | 1,419/1,535 (92.44%) |
| `in_muse` | 150/1,535 (9.77%) |
| `in_wikidata` | 1,006/1,535 (65.54%) |
| `external_attested` | 1,086/1,535 (70.75%) |

MUSE was used only to compute an attestation flag; the flag column and dictionary pairs are not distributed. Back-translation routes in the main test set:

| Back-translation route | Main test concepts |
|---|---:|
| `strict` | 1,111/1,535 (72.38%) |
| `rt_contain` | 293/1,535 (19.09%) |
| `fwd_only` | 131/1,535 (8.53%) |
| `none` | 0/1,535 (0.00%) |

Concreteness match-route counts across all package concepts (scores omitted):

| Match route | Concepts |
|---|---:|
| `exact` | 1,653/1,837 (89.98%) |
| `head` | 135/1,837 (7.35%) |
| `none` | 49/1,837 (2.67%) |

## Limitations

The main test set is 83.4% nouns, and its concepts have one Vietnamese sense each; the separate M1 extension has exactly two senses per concept. Etymology classification is automatic (primary κ = 0.594); Signal B partly reuses Wiktionary character-entry readings. Vietnamese Zipf frequency is not directly comparable with Zipf scores for other languages. The preregistered pre-model imbalance was d = +0.89 (Vietnamese frequency), +0.83 (Chinese frequency), −0.83 (concreteness), −0.62 (Vietnamese tokens).

## Not included, and how to add it

- `in_muse` is omitted because MUSE is CC BY-NC; restoration instructions and source details are available in the [preparation repository at `prereg-v1.2`](https://github.com/tarudesu/viLens-concepts-preparation/tree/prereg-v1.2).
- Brysbaert concreteness values are omitted until redistribution terms are confirmed; restoration instructions are available in the [preparation repository at `prereg-v1.2`](https://github.com/tarudesu/viLens-concepts-preparation/tree/prereg-v1.2). The `concreteness_match` route is retained.
- `directions_flores.jsonl` and all FLORES sentence text are omitted because FLORES+ is gated; reconstruction instructions are available in the [preparation repository at `prereg-v1.2`](https://github.com/tarudesu/viLens-concepts-preparation/tree/prereg-v1.2).

## Licensing and attribution

This package is offered under **CC BY-SA 4.0**. Attribution: Wiktionary/Wiktextract (kaikki.org) and wordfreq; Wikidata structured data is CC0; Unihan is distributed under the Unicode License; CC-CEDICT is CC BY-SA. NLLB-200 was used only for validation flags and model weights are not redistributed. MUSE pairs, Brysbaert values, and FLORES sentence text are not included. See [`LICENSE`](LICENSE) for the legal-code reference; source provenance is documented in the [preparation repository at `prereg-v1.2`](https://github.com/tarudesu/viLens-concepts-preparation/tree/prereg-v1.2).

## Known issues

- Cloze was withdrawn in v1.3 after its audit and rebuild exposed malformed and trivially passing examples; the v1.2 files remain at the [HF `v1.2` tag](https://huggingface.co/datasets/tarudesu/viLens-concepts/tree/v1.2) for provenance only.
- `cloze_available` is a frozen v1.2 column and is deprecated; do not use it in any analysis.
- The v1.2 `prompts` config cannot be loaded with `load_dataset`; read its prompt files directly at that tag.
- `vi_nodiac` is lowercased for gấu Bắc Cực (`1626f004dcd7`) and đậu Lima (`f918ffa42beb`).
- Nodiac demonstrations include collapsed forms: thứ tư, tiệc, and sắt.
- Questionable targets are retained as frozen data: chết đuối (`151a3fe224ea`), rưỡi (`17781bd98b3c`), tủy xương (`922cd1167021`), hội nghị thượng đỉnh (`c9584b841817`), and cảnh sát chính tả (`c505c130104c`).

## Changelog

- **v1.3:** Withdraws cloze before any study-LLM run; removes the cloze configuration and files from the main release. Repetition, translation, directions, and the frozen concept table are unchanged.

## Citation

To be updated on publication:

```bibtex
@article{vilens_concepts_2026,
  author = {Anonymous},
  title = {The Latent Language of Vietnamese Prompts: Where and When Do LLMs Translate Back?},
  year = {2026},
  note = {To be updated on publication}
}
```
