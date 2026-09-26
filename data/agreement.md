# Dataset agreement and release audit

Protocol: [PROTOCOL_AMENDMENTS.md](../PROTOCOL_AMENDMENTS.md)  
Source provenance: [SOURCES.md](../build/SOURCES.md)

## Dropflow (latest run per step)

| step | split | stage | unit | n_in | n_out | n_out_by_pos |
|---|---|---|---|---|---|---|
| 03_pool | all | raw_groups_pos_filtered | entries | 1492836 | 1181943 | {"adj": 184321, "noun": 806315, "verb": 191307} |
| 03_pool | all | five_way | groups | 92378 | 6194 | {"adj": 454, "noun": 5298, "verb": 442} |
| 03_pool | all | filter6_missing_vi_entry | concepts | 6194 | 4426 | {"adj": 397, "noun": 3622, "verb": 407} |
| 04_attest | all | filter1_source_agreement | concepts | 4426 | 3945 | {"adj": 303, "noun": 3277, "verb": 365} |
| 05_split | fewshot_reservoir | split | concepts | 3945 | 80 | {"adj": 5, "noun": 68, "verb": 7} |
| 05_split | directions | split | concepts | 3945 | 222 | {"adj": 18, "noun": 182, "verb": 22} |
| 05_split | test | split | concepts | 3945 | 3643 | {"adj": 280, "noun": 3027, "verb": 336} |
| 06 | fewshot_reservoir | canonical_missing_lang | concepts | 80 | 80 | {"adj": 5, "noun": 68, "verb": 7} |
| 06 | fewshot_reservoir | filter5_proper_noun | concepts | 80 | 78 | {"adj": 5, "noun": 66, "verb": 7} |
| 06 | fewshot_reservoir | filter3_polysemy | concepts | 78 | 59 | {"adj": 3, "noun": 52, "verb": 4} |
| 06 | fewshot_reservoir | filter4_surface | concepts | 59 | 59 | {"adj": 3, "noun": 52, "verb": 4} |
| 06 | fewshot_reservoir | filter4b_loanword | concepts | 59 | 58 | {"adj": 3, "noun": 51, "verb": 4} |
| 06 | fewshot_reservoir | filter4c_hyphen | concepts | 58 | 58 | {"adj": 3, "noun": 51, "verb": 4} |
| 06 | fewshot_reservoir | dedup_vi_form | concepts | 58 | 56 | {"adj": 3, "noun": 50, "verb": 3} |
| 06 | fewshot_reservoir | dedup_en_lemma | concepts | 56 | 55 | {"adj": 3, "noun": 49, "verb": 3} |
| 06 | directions | canonical_missing_lang | concepts | 222 | 222 | {"adj": 18, "noun": 182, "verb": 22} |
| 06 | directions | filter5_proper_noun | concepts | 222 | 217 | {"adj": 17, "noun": 178, "verb": 22} |
| 06 | directions | filter3_polysemy | concepts | 217 | 155 | {"adj": 13, "noun": 131, "verb": 11} |
| 06 | directions | filter4_surface | concepts | 155 | 147 | {"adj": 13, "noun": 123, "verb": 11} |
| 06 | directions | filter4b_loanword | concepts | 147 | 145 | {"adj": 13, "noun": 121, "verb": 11} |
| 06 | directions | filter4c_hyphen | concepts | 145 | 145 | {"adj": 13, "noun": 121, "verb": 11} |
| 06 | directions | dedup_vi_form | concepts | 145 | 136 | {"adj": 12, "noun": 114, "verb": 10} |
| 06 | directions | dedup_en_lemma | concepts | 136 | 135 | {"adj": 12, "noun": 113, "verb": 10} |
| 06 | test | canonical_missing_lang | concepts | 3643 | 3642 | {"adj": 280, "noun": 3026, "verb": 336} |
| 06 | test | filter5_proper_noun | concepts | 3642 | 3590 | {"adj": 267, "noun": 2987, "verb": 336} |
| 06 | test | filter3_polysemy | concepts | 3590 | 2736 | {"adj": 182, "noun": 2325, "verb": 229} |
| 06 | test | filter4_surface | concepts | 2736 | 2620 | {"adj": 179, "noun": 2214, "verb": 227} |
| 06 | test | filter4b_loanword | concepts | 2620 | 2558 | {"adj": 177, "noun": 2154, "verb": 227} |
| 06 | test | filter4c_hyphen | concepts | 2558 | 2557 | {"adj": 177, "noun": 2153, "verb": 227} |
| 06 | test | dedup_vi_form | concepts | 2557 | 2360 | {"adj": 158, "noun": 2005, "verb": 197} |
| 06 | test | dedup_en_lemma | concepts | 2360 | 2298 | {"adj": 149, "noun": 1966, "verb": 183} |
| 07 | fewshot_reservoir | filter2_backtranslation | concepts | 55 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 07 | all | filter3_polysemy_fewshot_reservoir_post_promotion | concepts | 47 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 07 | all | filter4_surface_fewshot_reservoir_post_promotion | concepts | 47 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 07 | all | filter4b_loanword_fewshot_reservoir_post_promotion | concepts | 47 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 07 | all | filter4c_hyphen_fewshot_reservoir_post_promotion | concepts | 47 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 07 | all | dedup_vi_form_fewshot_reservoir_post_promotion | concepts | 47 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 07 | all | dedup_en_lemma_fewshot_reservoir_post_promotion | concepts | 47 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 07 | directions | filter2_backtranslation | concepts | 135 | 135 | {"adj": 12, "noun": 113, "verb": 10} |
| 07 | all | filter3_polysemy_directions_post_promotion | concepts | 135 | 133 | {"adj": 12, "noun": 111, "verb": 10} |
| 07 | all | filter4_surface_directions_post_promotion | concepts | 133 | 133 | {"adj": 12, "noun": 111, "verb": 10} |
| 07 | all | filter4b_loanword_directions_post_promotion | concepts | 133 | 133 | {"adj": 12, "noun": 111, "verb": 10} |
| 07 | all | filter4c_hyphen_directions_post_promotion | concepts | 133 | 133 | {"adj": 12, "noun": 111, "verb": 10} |
| 07 | all | dedup_vi_form_directions_post_promotion | concepts | 133 | 132 | {"adj": 12, "noun": 110, "verb": 10} |
| 07 | all | dedup_en_lemma_directions_post_promotion | concepts | 132 | 132 | {"adj": 12, "noun": 110, "verb": 10} |
| 07 | test | filter2_backtranslation | concepts | 2298 | 1569 | {"adj": 114, "noun": 1310, "verb": 145} |
| 07 | all | filter3_polysemy_test_post_promotion | concepts | 1569 | 1543 | {"adj": 113, "noun": 1288, "verb": 142} |
| 07 | all | filter4_surface_test_post_promotion | concepts | 1543 | 1543 | {"adj": 113, "noun": 1288, "verb": 142} |
| 07 | all | filter4b_loanword_test_post_promotion | concepts | 1543 | 1541 | {"adj": 113, "noun": 1286, "verb": 142} |
| 07 | all | filter4c_hyphen_test_post_promotion | concepts | 1541 | 1541 | {"adj": 113, "noun": 1286, "verb": 142} |
| 07 | all | dedup_vi_form_test_post_promotion | concepts | 1541 | 1535 | {"adj": 113, "noun": 1280, "verb": 142} |
| 07 | all | dedup_en_lemma_test_post_promotion | concepts | 1535 | 1535 | {"adj": 113, "noun": 1280, "verb": 142} |
| 08 | fewshot_reservoir | etymology_classified | concepts | 47 | 47 | {"adj": 3, "noun": 41, "verb": 3} |
| 08 | directions | etymology_classified | concepts | 132 | 132 | {"adj": 12, "noun": 110, "verb": 10} |
| 08 | test | etymology_classified | concepts | 1535 | 1535 | {"adj": 113, "noun": 1280, "verb": 142} |
| 11b | all | extension_candidates | concepts | 3643 | 181 | {"adj": 21, "noun": 127, "verb": 33} |
| 11b | all | filter2_backtranslation | concepts | 181 | 130 | {"adj": 19, "noun": 80, "verb": 31} |
| 11b | all | post_promotion_filters | concepts | 130 | 125 | {"adj": 17, "noun": 78, "verb": 30} |
| 11b | all | disjoint_main_set | concepts | 125 | 123 | {"adj": 16, "noun": 77, "verb": 30} |
| 11b | all | extension_enriched_08_11 | concepts | 123 | 123 | {"adj": 16, "noun": 77, "verb": 30} |
| 12 | test | translation_ru_available | concepts | 1658 | 1647 | {"adj": 128, "noun": 1347, "verb": 172} |
| 12 | test | cloze_sentence_available | concepts | 1658 | 337 | {"adj": 44, "noun": 242, "verb": 51} |
| 12 | main_test | cloze_sentence_available | concepts | 1535 | 265 | {"adj": 35, "noun": 200, "verb": 30} |
| 12 | test | cloze_nodiac_noncollapsed | concepts | 337 | 193 | {"adj": 23, "noun": 156, "verb": 14} |

## Attestation

Shares are computed on the released rows within each split; flags are sourced from the selected VI candidate.

| split | flag | share |
|---|---|---|
| fewshot_reservoir | in_vi_gloss | 38/47 (0.8085) |
| fewshot_reservoir | in_muse | 9/47 (0.1915) |
| fewshot_reservoir | in_wikidata | 44/47 (0.9362) |
| fewshot_reservoir | external_attested | 47/47 (1.0000) |
| directions | in_vi_gloss | 120/132 (0.9091) |
| directions | in_muse | 10/132 (0.0758) |
| directions | in_wikidata | 84/132 (0.6364) |
| directions | external_attested | 88/132 (0.6667) |
| test | in_vi_gloss | 1419/1535 (0.9244) |
| test | in_muse | 150/1535 (0.0977) |
| test | in_wikidata | 1006/1535 (0.6554) |
| test | external_attested | 1086/1535 (0.7075) |

MUSE unique VI-side forms containing a space: 0/73874 (0.0000).

## Etymology agreement and H3

Signal A × primary Signal B:

| A | B=sino | B=nonsino |
|---|---|---|
| sino | 537 | 333 |
| nonsino | 9 | 780 |
| other_loan | 0 | 2 |
| conflict | 28 | 25 |

Cohen's κ (A ∈ {sino, nonsino}); seeded percentile bootstrap:

| B variant | n | κ | 95% CI |
|---|---|---|---|
| primary | 1659 | 0.594470 | [0.557133, 0.631475] |
| strict | 1659 | 0.540098 | [0.501878, 0.577138] |
| relaxed | 1659 | 0.584800 | [0.546988, 0.621144] |

Primary-verified reading_source = wiktionary_char for 149/577 (0.2582).

sino_via counts: `{"calque": 51, "chinese": 109, "japanese_kango": 53, "template": 862}`.
TEST native_strict concepts: 51.

Power (two-sided standardized MDE, 80% target):

| comparison | n1 | n2 | d at α=.05 | d at Holm α |
|---|---|---|---|---|
| sino_vs_nonsino | 481 | 706 | 0.165636 | 0.205738 |
| sino_vs_native_strict | 481 | 51 | 0.412574 | 0.512463 |

H3 status by configured rule: **exploratory** (primary κ lower bound=0.557133; Holm MDE=0.205738).

## Strata

| split | stratum | concepts |
|---|---|---|
| fewshot_reservoir | sino | 16 |
| fewshot_reservoir | nonsino | 17 |
| fewshot_reservoir | ambiguous | 14 |
| fewshot_reservoir | other_loan | 0 |
| directions | sino | 40 |
| directions | nonsino | 57 |
| directions | ambiguous | 34 |
| directions | other_loan | 1 |
| test | sino | 481 |
| test | nonsino | 706 |
| test | ambiguous | 347 |
| test | other_loan | 1 |

TEST stratum by POS × syllable bucket:

| POS | syllables | stratum | concepts |
|---|---|---|---|
| adj | 1 | ambiguous | 11 |
| adj | 1 | nonsino | 17 |
| adj | 1 | sino | 5 |
| adj | 2 | ambiguous | 13 |
| adj | 2 | nonsino | 23 |
| adj | 2 | sino | 37 |
| adj | 3+ | ambiguous | 5 |
| adj | 3+ | nonsino | 2 |
| noun | 1 | ambiguous | 30 |
| noun | 1 | nonsino | 60 |
| noun | 1 | other_loan | 1 |
| noun | 1 | sino | 26 |
| noun | 2 | ambiguous | 208 |
| noun | 2 | nonsino | 358 |
| noun | 2 | sino | 336 |
| noun | 3+ | ambiguous | 53 |
| noun | 3+ | nonsino | 176 |
| noun | 3+ | sino | 32 |
| verb | 1 | ambiguous | 12 |
| verb | 1 | nonsino | 34 |
| verb | 1 | sino | 11 |
| verb | 2 | ambiguous | 15 |
| verb | 2 | nonsino | 36 |
| verb | 2 | sino | 33 |
| verb | 3+ | sino | 1 |

## Diacritic collapse

| split | stratum | collapsed |
|---|---|---|
| fewshot_reservoir | sino | 4/16 (0.2500) |
| fewshot_reservoir | nonsino | 4/17 (0.2353) |
| fewshot_reservoir | ambiguous | 4/14 (0.2857) |
| directions | sino | 6/40 (0.1500) |
| directions | nonsino | 11/57 (0.1930) |
| directions | ambiguous | 7/34 (0.2059) |
| directions | other_loan | 1/1 (1.0000) |
| test | sino | 85/481 (0.1767) |
| test | nonsino | 150/706 (0.2125) |
| test | ambiguous | 68/347 (0.1960) |
| test | other_loan | 1/1 (1.0000) |

## E1 tokenization and M1

Token counts and single-token coverage are computed on the TEST rows.

| model | lang | ntok mean | median | max |
|---|---|---|---|---|
| gemma | vi | 2.399 | 2.000 | 6 |
| gemma | en | 1.209 | 1.000 | 6 |
| gemma | zh | 2.401 | 2.000 | 5 |
| gemma | fr | 1.931 | 2.000 | 7 |
| gemma | id | 2.116 | 2.000 | 7 |
| qwen | vi | 2.594 | 2.000 | 7 |
| qwen | en | 1.334 | 1.000 | 4 |
| qwen | zh | 3.397 | 3.000 | 6 |
| qwen | fr | 2.483 | 2.000 | 9 |
| qwen | id | 2.958 | 3.000 | 12 |
| llama | vi | 2.515 | 2.000 | 7 |
| llama | en | 1.334 | 1.000 | 4 |
| llama | zh | 2.866 | 3.000 | 7 |
| llama | fr | 2.511 | 2.000 | 9 |
| llama | id | 2.914 | 3.000 | 11 |

Single-token VI coverage by stratum:

| model | stratum | single-token VI |
|---|---|---|
| gemma | sino | 26/481 (0.0541) |
| gemma | nonsino | 50/706 (0.0708) |
| gemma | ambiguous | 31/347 (0.0893) |
| gemma | other_loan | 1/1 (1.0000) |
| qwen | sino | 21/481 (0.0437) |
| qwen | nonsino | 46/706 (0.0652) |
| qwen | ambiguous | 29/347 (0.0836) |
| qwen | other_loan | 0/1 (0.0000) |
| llama | sino | 23/481 (0.0478) |
| llama | nonsino | 47/706 (0.0666) |
| llama | ambiguous | 26/347 (0.0749) |
| llama | other_loan | 0/1 (0.0000) |

Single-token VI coverage by syllable bucket:

| model | syllable bucket | single-token VI |
|---|---|---|
| gemma | 1 | 108/207 (0.5217) |
| gemma | 2 | 0/1059 (0.0000) |
| gemma | 3+ | 0/269 (0.0000) |
| qwen | 1 | 96/207 (0.4638) |
| qwen | 2 | 0/1059 (0.0000) |
| qwen | 3+ | 0/269 (0.0000) |
| llama | 1 | 96/207 (0.4638) |
| llama | 2 | 0/1059 (0.0000) |
| llama | 3+ | 0/269 (0.0000) |

M1 extension concepts: 123

| model | M1 eligible (main test + extension) |
|---|---|
| gemma | 186 |
| qwen | 166 |
| llama | 162 |

## FLORES fertility

| model | lang | tokens/character | tokens/syllable |
|---|---|---|---|
| gemma | vi | 0.2683285832822252 | 1.2103824247355575 |
| gemma | en | 0.20546201294272767 | NA |
| gemma | zh | 0.6761006289308176 | NA |
| gemma | fr | 0.24220048465471522 | NA |
| gemma | id | 0.21919925799969084 | NA |
| qwen | vi | 0.2860348497420542 | 1.2902522375915377 |
| qwen | en | 0.20930391161359743 | NA |
| qwen | zh | 0.6405151683314836 | NA |
| qwen | fr | 0.27686852154937264 | NA |
| qwen | id | 0.2987675487289029 | NA |
| llama | vi | 0.2728020491359717 | 1.2305614320585843 |
| llama | en | 0.20589394237909764 | NA |
| llama | zh | 0.8048464668886423 | NA |
| llama | fr | 0.27577107042718124 | NA |
| llama | id | 0.29355387231411345 | NA |

FLORES direction-prompt JSONL SHA-256 (text not reproduced here): `e768392ccbee4ddbd20fd127fa97103f64f19c54e333df285e12bb67067ea25e`.

## Cloze and few-shot

Recorded cloze status: **supplementary**.

Step-12 main-test cloze coverage: 265/1535.

Main-test cloze coverage by stratum:

| stratum | available | main test |
|---|---|---|
| ambiguous | 64 | 347 |
| nonsino | 84 | 706 |
| other_loan | 0 | 1 |
| sino | 117 | 481 |

Selected few-shot concepts (set | en | vi | ru | stratum):

```text
1 | research | nghiên cứu | исследование | sino
1 | violence | bạo lực | насилие | sino
1 | fourth | thứ tư | четвёртый | nonsino
1 | battle | trận đánh | сражение | nonsino
1 | milk tea | trà sữa | чай с молоком | nonsino
2 | commander-in-chief | tổng tư lệnh | главнокомандующий | sino
2 | text | văn bản | текст | sino
2 | candidate | ứng cử viên | кандидат | ambiguous
2 | foreign language | ngoại ngữ | иностранный язык | sino
2 | truck | xe tải | грузовик | nonsino
3 | generation | thế hệ | поколение | sino
3 | divorce | ly hôn | развод | ambiguous
3 | therapy | điều trị | терапия | ambiguous
3 | fine arts | mỹ thuật | изобразительное искусство | ambiguous
3 | group | nhóm | группа | nonsino
4 | feeling | cảm giác | ощущение | sino
4 | director | giám đốc | директор | sino
4 | air strike | không kích | воздушный удар | ambiguous
4 | war crime | tội ác chiến tranh | военное преступление | nonsino
4 | natural gas | khí đốt | природный газ | nonsino
5 | sixth | thứ sáu | шестой | nonsino
5 | yellow | vàng | жёлтый | ambiguous
5 | civil war | nội chiến | гражданская война | sino
5 | complex number | số phức | комплексное число | ambiguous
5 | wet | ướt | мокрый | nonsino
6 | budget | ngân sách | бюджет | ambiguous
6 | iron | sắt | железо | nonsino
6 | short story | truyện ngắn | рассказ | nonsino
6 | basketball | bóng rổ | баскетбол | nonsino
6 | party | tiệc | вечеринка | ambiguous
```

## Sources

See the full file-level table in [SOURCES.md](../build/SOURCES.md).
