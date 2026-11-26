# babel_tower

Sentence-level alignment of an English book and its Hungarian translation. It turns two EPUBs
into a parallel corpus of matched sentence groups. Most groups are 1-1 pairs, but 1-2, 2-1,
2-2, 1-3 and 3-1 groups capture places where the translator split or merged sentences.

## Pipeline

| Module | Role |
|---|---|
| `extract_epub.py` | EPUB → paragraphs in reading order (JSONL) |
| `splitter.py` | Paragraphs → sentences, using rules for English and Hungarian (dialogue dashes, abbreviations, initials) |
| `align.py` | Sentences → aligned groups, using [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings and a dynamic-programming aligner |
| `inspect_pairs.py` | Samples pairs by similarity or group type, or diffs two alignment runs |

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

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python src/babel_tower/extract_epub.py en.epub cache/en.paragraphs.jsonl
uv run python src/babel_tower/splitter.py cache/en.paragraphs.jsonl cache/en.sentences.jsonl --lang en
# …same two steps for the Hungarian EPUB with --lang hu…
uv run python src/babel_tower/align.py cache/en.sentences.jsonl cache/hu.sentences.jsonl out/pairs.jsonl
uv run python src/babel_tower/inspect_pairs.py sample out/pairs.jsonl --lo 0.5 --hi 0.6
```

Each output line is one aligned group:

```json
{"pair": 0, "type": "1-1", "sim": 0.89, "en_ids": [12], "hu_ids": [14], "en": "Good morning.", "hu": "Jó reggelt."}
```

Sentences left out of every group are written to `pairs.unaligned.jsonl`.
