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

## Development

```sh
uv run ruff check src      # lint
uv run ruff format src     # format
uv run mypy                # type-check (strict)
```
