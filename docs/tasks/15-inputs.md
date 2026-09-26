Both manual inputs are now in place:
- data/raw/brysbaert/13428_2013_403_MOESM1_ESM.xlsx (the original Brysbaert et al.
  2014 supplementary file, unmodified);
- FLORES+ gated access has been accepted on Hugging Face (token in keys/hf.txt).

STEP A: register the inputs
1. Run `01_download.py --only brysbaert`: hash the xlsx file and record it in
   SOURCES.md (source: Brysbaert, Warriner & Kuperman 2014, BRM,
   doi 10.3758/s13428-013-0403-5, supplementary material ESM1; license: "as
   distributed with the article; check terms before redistributing values").
2. Run `01_download.py --only flores`. Confirm all 10 files exist:
   data/raw/flores/{dev,devtest}_{vie_Latn,eng_Latn,cmn_Hans,fra_Latn,ind_Latn}.parquet,
   and that SOURCES.md records the dataset commit hash. If access is still denied,
   STOP and report the exact error (do not use any other FLORES copy).
3. Brysbaert schema check before step 09b: read the xlsx with pandas (add openpyxl
   if it is not installed; commit uv.lock). Print the column names, the row count
   and the first 5 rows. Required columns: "Word" and "Conc.M"; if a "Bigram"
   column exists, it marks two-word expressions (use those rows for exact matches
   on two-word English lemmas). If "Word" or "Conc.M" is missing, STOP and report
   the actual columns.

STEP B: resume TASK 15 from the beginning, exactly as specified in docs/tasks/15.md,
with one correction: the Makefile target is `directions-flores` (the spec's
   `direction_flores` was a typo).
Order: make concreteness → make fertility → make directions-flores → make release,
then the fresh-clone reproducibility check, the integrity tests, the final amendment,
PREREG.md and the `prereg-v1.2` tag (do not push the tag).

REPORT BACK: everything listed in TASK 15, plus: the Brysbaert columns and row count;
the concreteness match rates (exact / head / none) by stratum and split.
