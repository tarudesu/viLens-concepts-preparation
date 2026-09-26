# viLens data pipeline

Use the existing `sb` environment. Activate it, then install the locked
dependencies into that environment and run the pipeline through `uv`:

```sh
uv sync --active --locked --inexact
uv run --active --no-sync python data/build/00_scaffold.py --config configs/data.yaml
uv run --active --no-sync pytest
```

`--inexact` preserves unrelated packages already installed in the shared `sb`
environment. Make targets also invoke scripts through `uv run --active --no-sync`.

`make step-00` runs `data/build/00_scaffold.py --config configs/data.yaml`.
Make automatically exposes `step-NN` for each `NN_<name>.py` script added to
the configured build directory; `all` runs available steps in the configured
dependency order. Override `CONFIG` or `PYTHON` when needed. All configured
relative paths are relative to the repository working directory.

For an offline rebuild after sources have been registered, run:

```sh
make -o step-01 all
```

Step 01 performs live source checks and requires network access (and the
appropriate access tokens). The `-o step-01` option tells Make to treat that
target as already up to date and skip it. All later steps rebuild offline from
`data/raw/` and the cached Wikidata and NLLB responses; required raw inputs and
caches must already be present.

Raw downloads live in `data/raw/`; intermediate Parquet files and logs live in
`data/interim/`; prompts live in `data/prompts/`; final outputs live in `data/`.
The raw and interim contents are ignored by Git apart from directory markers.
The Chinese code field and model revisions remain explicit TODOs until their
sources are supplied. Each consuming step must reject unresolved required TODOs,
missing languages, and unexpected schemas with a diagnostic.

Shared helpers in `data/build/common.py` safely load YAML, normalize nested
strings and keys to NFC, stream JSONL one record at a time, append filter-stage
counts, and log to stdout plus a per-step file. Duplicate/NFC-colliding keys,
blank or malformed JSONL records, non-standard JSON numbers, and inconsistent
drop counts fail explicitly. Repeated runs replace the timestamp-free step log;
data outputs must be byte-identical. The requested append-only dropflow is an
audit history: reruns append identical stage records, so the history grows.
Task 00 has no filter stages and writes no dropflow records.

Each later step must take `--config`, read data only from raw/interim, and keep
all paths, thresholds, sizes, and seeds in the config. Network access is limited
to steps 01 and 04, plus tokenizer downloads in step 10. Random operations must
use a local `numpy.random.default_rng(config["seed"])` without changing global
random state. Add dependencies with `uv`, commit `uv.lock`, and test every pure
function, including normalization, edit distance, diacritic stripping, and
matching rules as those functions are introduced.
