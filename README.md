# babel_tower

Sentence-level alignment of an English book and its Hungarian translation. It turns two EPUBs
into a parallel corpus of matched sentence groups. Most groups are 1-1 pairs, but 1-2, 2-1,
2-2, 1-3 and 3-1 groups capture places where the translator split or merged sentences.

## Pipeline

| Module | Role |
|---|---|
| `epub.py` | EPUB → paragraphs in reading order |
| `splitter.py` | Paragraphs → sentences, using rules for English and Hungarian (dialogue dashes, abbreviations, initials) |
| `embed.py` | Embeds windows of consecutive sentences with [LaBSE](https://huggingface.co/sentence-transformers/LaBSE), cached on disk |
| `align.py` | Sentences → aligned groups, using a dynamic-programming aligner over those embeddings |
| `records.py` | The `Paragraph` / `Sentence` / `Pair` / `Unaligned` records; each is one JSONL line |
| `report.py`, `review.py` | Reports for checking each stage by eye; sampling pairs and diffing two runs |
| `gold.py` | Hand-checked reference alignments, and scoring a run against them |
| `provenance.py` | Input checksums, code commit and environment for `run.json` |
| `cli.py` | The `babel-tower` command |

The aligner embeds every window of 1–3 consecutive sentences on each side. It then finds the
cheapest monotonic path through the full alignment grid:

```text
cost(group) = (1 − cosine) · (n + m) + group_penalty · (n + m − 2)
cost(skip)  = skip_cost per skipped sentence
```

`--skip-cost` sets the cosine below which sentences are left unaligned rather than forced
together. `--group-penalty` stops large windows, which naturally score higher cosines, from
swallowing what should be separate 1-1 pairs. Embeddings are cached, so reruns with different
parameters are fast.

## Usage

Requires Python 3.14, [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/) (the
EPUBs in the `data` submodule are stored in LFS).

```sh
git submodule update --init && (cd data && git lfs install && git lfs pull)
uv sync
uv run babel-tower run data/eng/book.epub data/hun/book.epub data/runs/book_skip06 --skip-cost 0.6
uv run babel-tower sample data/runs/book_skip06/pairs.jsonl --lo 0.5 --hi 0.6
uv run babel-tower compare data/runs/book_skip06/pairs.jsonl data/runs/book_gp02/pairs.jsonl
```

`babel-tower run` writes everything for one run into the output directory:

| File | Contents |
|---|---|
| `{en,hu}.paragraphs.jsonl` | extracted paragraphs |
| `{en,hu}.sentences.jsonl` | split sentences (the ids that pairs refer to) |
| `pairs.jsonl` | aligned groups |
| `pairs.unaligned.jsonl` | sentences left out of every group |
| `run.json` | input checksums, settings, code commit, library versions, device, timings, summary stats |

Embedding uses CUDA, then MPS, then CPU, whichever is available first (override with `--device`). Embeddings are cached in
`data/cache/`, so reruns with other `--skip-cost` / `--group-penalty` values skip the model entirely.
Each step can also run on its own: `babel-tower extract`, `split` and `align` (see `babel-tower --help`).

Each output line is one aligned group:

```json
{"pair": 0, "type": "1-1", "sim": 0.89, "en_ids": [12], "hu_ids": [14], "en": "Good morning.", "hu": "Jó reggelt."}
```

Sentences left out of every group are written to `pairs.unaligned.jsonl`.

## Measuring quality

A gold file is a hand-corrected alignment of a few stretches of the book. `gold-template`
pre-fills one from a run; fix it by moving lines between groups (the format is explained at the
top of the file). Then score any run against it:

```sh
uv run babel-tower gold-template data/runs/book_skip06 data/gold/book.gold.txt   # then correct it by hand
uv run babel-tower eval data/gold/book.gold.txt data/runs/book_skip06
```

`eval` reports precision, recall and F1 for three things, counting only sentences inside the gold windows:

- **links**: (English sentence, Hungarian sentence) pairs that translate each other, so a group that is
  nearly right still earns partial credit.
- **groups**: aligned groups that match a gold group exactly.
- **skipped**: sentences correctly left without a counterpart.

It then prints the gold groups that the run got wrong, next to what the run did instead. Gold
files refer to sentence ids, so they only fit runs made with the same sentence split: `eval`
checks the texts, and refuses if they differ.

## Training corpus

`babel-tower corpus` turns the pairs of one or more runs into `train.jsonl` and `dev.jsonl`
(one `{"en", "hu", "sim", "type", "source", "pair"}` object per line) plus a `manifest.json`:

```sh
uv run babel-tower corpus data/corpus/v1 data/runs/book_a data/runs/book_b --min-sim 0.5
```

- **dev** is a few contiguous blocks of each book (5% by default), so dev sentences do not sit
  next to the train sentences that share their context.
- **filters**: similarity below `--min-sim` (0.5: below that, pairs are mostly loose or wrong),
  length ratio above `--max-length-ratio`, no letters, exact duplicates.
- **benchmark guard**: a pair is dropped if any of its sentences also appears in a benchmark
  (compared case- and punctuation-insensitively).
- the manifest records what was dropped and why, word and word-type counts per language, and
  the `run.json` of every source.

## Translation benchmarks

The aligned pairs are training data for an English ↔ Hungarian translator. Benchmarks are kept
separate from the books, so models trained on them can be tested fairly:

| Benchmark | Contents |
|---|---|
| `literary` | 200 fiction-style sentences in 8 sections (narration, dialogue, idioms, ...) written for this project: [`benchmarks/literary.txt`](benchmarks/literary.txt) |
| `flores-dev`, `flores-devtest` | [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), the standard general-domain test set. Gated: accept its terms on Hugging Face and log in with `hf auth login`. It is downloaded to the Hugging Face cache, never into this repo. |

```sh
uv run babel-tower bench 'Helsinki-NLP/opus-mt-{src}-{tgt}'   # both directions; {src}/{tgt} filled in
uv run babel-tower results                                      # table of every result so far
```

Scores are chrF++ (the main metric: character n-grams give partial credit for a word with
the right stem and the wrong suffix, which matters for Hungarian) and BLEU, from sacrebleu.
chrF++ comes with a 95% bootstrap confidence interval (the `±95%` column of `results`): two
models whose intervals overlap are not reliably different. Each result in `data/results/` keeps every translation, per-section scores, sacrebleu
signatures, generation settings and the code version. Scores are not directly comparable across
directions, since EN→HU is scored on Hungarian text and HU→EN on English.

## Development

```sh
uv run pytest                     # tests (synthetic text only)
uv run ruff check src tests       # lint
uv run ruff format src tests      # format
uv run mypy                       # type-check (strict)
```
