"""babel-tower: align an English book with its Hungarian translation, sentence by sentence.

    babel-tower run en.epub hu.epub data/runs/book/ --skip-cost 0.6

`run` chains every step and writes all intermediate files plus run.json (input checksums,
settings, code commit, library versions, device, timings, summary stats) to one directory.
The steps can also be run one at a time: extract, split, align. sample and compare are for
looking at the output.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from babel_tower import jsonl, provenance, report, review
from babel_tower.align import MAX_GROUP, AlignConfig, Alignment, align
from babel_tower.embed import DEFAULT_MODEL, Embedder
from babel_tower.epub import LfsPointerError, extract_paragraphs
from babel_tower.records import LANGS, Lang, Pair, Paragraph, Sentence
from babel_tower.splitter import split_paragraphs


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        args.command(args)
    except LfsPointerError as e:
        sys.exit(f"error: {e}")


def cmd_extract(args: argparse.Namespace) -> None:
    rows = extract_paragraphs(args.epub)
    jsonl.write(args.out, rows)
    report.paragraphs(rows)


def cmd_split(args: argparse.Namespace) -> None:
    rows = split_paragraphs(jsonl.read(args.paragraphs, Paragraph), args.lang)
    jsonl.write(args.out, rows)
    report.sentences(rows, verbose=True)


def cmd_align(args: argparse.Namespace) -> None:
    result = align(
        jsonl.read(args.en, Sentence), jsonl.read(args.hu, Sentence), _config(args), _embedder(args), args.cache
    )
    _write_alignment(args.out, result)
    report.alignment(result)


def cmd_run(args: argparse.Namespace) -> None:
    out: Path = args.out
    timings: dict[str, float] = {}
    books: dict[Lang, list[Sentence]] = {}
    for lang, epub in zip(LANGS, (args.en, args.hu), strict=True):
        print(f"=== {lang}: {epub}")
        t = time.perf_counter()
        paragraphs = extract_paragraphs(epub)
        jsonl.write(out / f"{lang}.paragraphs.jsonl", paragraphs)
        timings[f"extract_{lang}_s"] = round(time.perf_counter() - t, 3)
        report.paragraphs(paragraphs)

        t = time.perf_counter()
        books[lang] = split_paragraphs(paragraphs, lang)
        jsonl.write(out / f"{lang}.sentences.jsonl", books[lang])
        timings[f"split_{lang}_s"] = round(time.perf_counter() - t, 3)
        report.sentences(books[lang], verbose=args.verbose)

    print("=== align")
    config, embedder = _config(args), _embedder(args)
    result = align(books["en"], books["hu"], config, embedder, args.cache)
    timings |= {"embed_s": result.embed_s, "dp_s": result.dp_s}
    _write_alignment(out / "pairs.jsonl", result)

    run: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "inputs": {lang: provenance.file_info(p) for lang, p in zip(LANGS, (args.en, args.hu), strict=True)},
        "settings": {**asdict(config), "model": args.model, "max_group": MAX_GROUP},
        "code": provenance.code_info(),
        "env": provenance.env_info(embedder.device),
        "embedding": {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(embedder.stats).items()},
        "timings": timings,
        "stats": report.alignment_stats(result),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report.alignment(result)
    s = embedder.stats
    print(
        f"=== wrote {out}/  (device {s.device}, embedding cache hits {s.cache_hits}/{s.cache_hits + s.cache_misses}, "
        f"encode {s.encode_s:.1f}s, dp {result.dp_s}s)"
    )


def cmd_sample(args: argparse.Namespace) -> None:
    review.sample(jsonl.read(args.pairs, Pair), lo=args.lo, hi=args.hi, group_type=args.type, n=args.n, seed=args.seed)


def cmd_compare(args: argparse.Namespace) -> None:
    review.compare(jsonl.read(args.a, Pair), jsonl.read(args.b, Pair), args.n, args.seed)


def _write_alignment(pairs_path: Path, result: Alignment) -> None:
    jsonl.write(pairs_path, result.pairs)
    jsonl.write(pairs_path.with_suffix(".unaligned.jsonl"), result.unaligned)


def _config(args: argparse.Namespace) -> AlignConfig:
    return AlignConfig(skip_cost=args.skip_cost, group_penalty=args.group_penalty, keep_verse=args.keep_verse)


def _embedder(args: argparse.Namespace) -> Embedder:
    return Embedder(args.model, args.device, args.batch_size)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="babel-tower", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(required=True, metavar="COMMAND")

    def command(name: str, fn: Any, help: str) -> argparse.ArgumentParser:  # noqa: A002
        p = sub.add_parser(name, help=help, description=help)
        p.set_defaults(command=fn)
        return p

    p = command("run", cmd_run, "EPUBs -> paragraphs -> sentences -> aligned pairs, into one directory")
    p.add_argument("en", type=Path, help="English EPUB")
    p.add_argument("hu", type=Path, help="Hungarian EPUB")
    p.add_argument("out", type=Path, help="output directory")
    p.add_argument("-v", "--verbose", action="store_true", help="print sample sentences in the splitter report")
    _add_align_options(p)

    p = command("extract", cmd_extract, "EPUB -> paragraphs JSONL")
    p.add_argument("epub", type=Path)
    p.add_argument("out", type=Path)

    p = command("split", cmd_split, "paragraphs JSONL -> sentences JSONL")
    p.add_argument("paragraphs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--lang", choices=LANGS, required=True)

    p = command("align", cmd_align, "two sentences JSONL files -> pairs JSONL (+ .unaligned.jsonl)")
    p.add_argument("en", type=Path, help="English sentences JSONL")
    p.add_argument("hu", type=Path, help="Hungarian sentences JSONL")
    p.add_argument("out", type=Path, help="output pairs JSONL")
    _add_align_options(p)

    p = command("sample", cmd_sample, "show random pairs in a similarity range or of one group type")
    p.add_argument("pairs", type=Path)
    p.add_argument("--lo", type=float, default=0.0)
    p.add_argument("--hi", type=float, default=1.01)
    p.add_argument("--type", help="group type, e.g. 2-2")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p = command("compare", cmd_compare, "show the pairs that differ between two runs")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return parser


def _add_align_options(p: argparse.ArgumentParser) -> None:
    defaults = AlignConfig()
    p.add_argument(
        "--skip-cost", type=float, default=defaults.skip_cost,
        help="cost per unaligned segment; pairs with cosine below about this are left out (default %(default)s)",
    )  # fmt: skip
    p.add_argument(
        "--group-penalty", type=float, default=defaults.group_penalty,
        help="extra cost per segment beyond a 1-1 pair; raise if 2-2 / 1-3 groups dominate (default %(default)s)",
    )  # fmt: skip
    p.add_argument("--keep-verse", action="store_true", help="align verse blocks too")
    p.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model (default %(default)s)")
    p.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cache", type=Path, default=Path("data/cache"), help="embedding cache directory")
