# TASK 14-prep-b (small):

1. Step 12 writes data/interim/12_fewshot.json with: the 30 selected concept_ids by
   set, the cloze demonstration concept_ids, cloze_status and the main-test cloze
   coverage. Re-run step 12 from cache and confirm all prompt files are byte-identical
   to before. Step 14 reads this JSON instead of parsing logs. Apply the same rule
   everywhere: step 14 and agreement.md must read structured outputs
   (parquet/json/dropflow.jsonl), never log text.
2. Confirm that the FLORES file names 10b/13b expect
   (data/raw/flores/{dev,devtest}_{lang}.parquet) match exactly what
   01_download.py --only flores writes; align them if not.
3. In the REAL working repo (not a fresh clone), run each target once and report
   the message:
   make concreteness      → expected: the Brysbaert-missing message
   make fertility         → expected: the FLORES-missing message
   make directions-flores → expected: the FLORES-missing message
   make release           → expected: a refusal because concreteness is pending
   Commit as "step 14-prep-b: structured release inputs".
