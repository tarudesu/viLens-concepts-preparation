Decisions:

1. Step 09 without concreteness (for now):

   - Run TASK 09 but SKIP concreteness; add the columns concreteness = NA and
     concreteness_match = "pending". Everything else as specified.
   - Brysbaert: the user will place the original file in data/raw/brysbaert/ by
     hand. When it exists, a later TASK 09b will run 01_download.py --only
     brysbaert (hash only) and fill the concreteness columns. Step 14 must
     REFUSE to build concepts.tsv while concreteness_match == "pending".

2. Step 11: proceed. Reference lexicon = all Vietnamese Wiktextract headwords ∪
   the full wordfreq Vietnamese list (10,622 words; that is the whole list).
   Record both sizes in the log.

3. Pipeline order is now 08 → 09 → 10 → 11 (re-run 10a on 09's output so the
   columns flow through; the results must be identical).

4. NEW diagnostic, report only (no data changes): the M1 extension estimate.
   From 05/06/07 intermediates, count TEST concepts that:

   - have a 1-syllable canonical vi,
   - were dropped ONLY by filter3_polysemy with n_senses_vi == 2
     (i.e. would pass every other step-06 filter),
   - and are a single token in each model (use the 10a counting rule).
     Report per model: the count before back-translation, and the count that
     would also pass filter 2 (run NLLB for these candidates only; the cache is
     fine), plus the stratum split (sino/nonsino/ambiguous) of those passing.
     Also report the main-set single-token count + that number = the projected M1 set.

5. Append to PROTOCOL_AMENDMENTS.md (v1.2):
   "E1 tokenizer audit (no model weights loaded): single-token Vietnamese coverage
   is 7.0% (Gemma), 6.3% (Qwen), 6.3% (Llama); all single-token items are
   monosyllabic. M1 adequacy (≥150) is not met by the main set. Planned remedy (main
   dataset unchanged): an M1-only extension set of monosyllabic concepts with
   exactly 2 senses that pass all other filters, used solely for the M1-vs-M2
   comparison (C1)."

Commit each step separately. Report: step 09 summary (without concreteness),
step 11 report, and the M1 extension diagnostic.
