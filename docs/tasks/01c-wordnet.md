Approved: fetch WordNet via the download step, not step 08.

- Add a source "wordnet" to data/build/01_download.py: download the NLTK corpus
  "wordnet" (and "omw-1.4" only if nltk's WordNet reader requires it) into
  data/raw/nltk_data/ with nltk.download(..., download_dir=...). Run only this
  source: `01_download.py --only wordnet`.
- Record it in SOURCES.md like the other sources (files, SHA-256, NLTK version,
  WordNet version as reported by wordnet.get_version(), license: WordNet License).
- Step 08 must load WordNet ONLY from data/raw/nltk_data (set nltk.data.path
  explicitly; no network). If the corpus is missing, step 08 stops with a clear message.
- Commit this as "step 01c: add WordNet source", then continue TASK 08b from
  point 2 and stop at Checkpoint C.
