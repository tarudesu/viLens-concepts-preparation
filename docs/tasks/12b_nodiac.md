TASK 12b: fix the nodiac condition; re-run step 12 (cache only).

1. nodiac = strip diacritics (common.strip_diacritics; đ→d; preserve case) from
   ALL Vietnamese text in the prompt: few-shot vi forms, the test item, cloze
   sentences and cloze answers, and the label "Tiếng Việt" → "Tieng Viet". Also
   target_vi = the stripped form. Russian text and the en/zh/fr/id targets are unchanged.
   Exclude concepts whose TEST vi form is collapsed (as before).
2. Report the cloze coverage for the MAIN test set only (m1_extension == False),
   and apply the 300 threshold to that number. Record cloze_status (primary |
   supplementary) in config and in the step log.
3. Tests: in a nodiac translation prompt there are no Vietnamese diacritics
   anywhere; the Russian text is identical to diac; target_vi is stripped;
   repetition and cloze likewise.
4. PROTOCOL_AMENDMENTS.md (v1.2): "E9 nodiac condition: all Vietnamese text in the
   prompt and the Vietnamese target are diacritic-stripped (simulating undiacritized
   input); items whose stripped test form collides with another word are excluded.
   The cloze primary/supplementary status is determined on the main test set only."
   REPORT BACK: one rendered nodiac prompt per format; cloze coverage on the main test
   set by stratum and the resulting status; record counts per file.
