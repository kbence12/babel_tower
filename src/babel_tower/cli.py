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
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from babel_tower import bench, corpus, gold, jsonl, provenance, report, review
from babel_tower.align import MAX_GROUP, AlignConfig, Alignment, align
from babel_tower.embed import DEFAULT_MODEL, Embedder
from babel_tower.epub import LfsPointerError, extract_paragraphs
from babel_tower.records import LANGS, Lang, Pair, Paragraph, Sentence, Unaligned
from babel_tower.splitter import split_paragraphs

UNALIGNED = "pairs.unaligned.jsonl"  # next to pairs.jsonl in a run directory


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        args.command(args)
    except (LfsPointerError, gold.GoldFormatError, bench.BenchmarkUnavailableError) as e:
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


def cmd_gold_template(args: argparse.Namespace) -> None:
    run_dir: Path = args.run
    if args.out.exists() and not args.force:
        sys.exit(f"error: {args.out} exists; it may hold corrections. Pass --force to overwrite it.")
    groups = gold.path_groups(jsonl.read(run_dir / "pairs.jsonl", Pair), jsonl.read(run_dir / UNALIGNED, Unaligned))
    windows = gold.pick_windows(groups, args.windows, args.size, args.seed)
    gold.write_template(args.out, windows, _sides(run_dir))
    n_en = sum(len(g.en) for w in windows for g in w)
    n_hu = sum(len(g.hu) for w in windows for g in w)
    print(f"wrote {args.out}: {len(windows)} windows, {n_en} EN and {n_hu} HU sentences to check")


def cmd_eval(args: argparse.Namespace) -> None:
    run_dir: Path = args.run
    windows = gold.parse(args.gold)
    if problems := gold.check_texts(windows, _sides(run_dir)):
        print("\n".join(problems[:10]))
        sys.exit(
            f"error: {len(problems)} gold sentences differ from {run_dir}: "
            "the gold file was made from a run with a different sentence split"
        )
    groups = gold.path_groups(jsonl.read(run_dir / "pairs.jsonl", Pair), jsonl.read(run_dir / UNALIGNED, Unaligned))
    if missing := gold.missing_lines(windows, groups):
        listed = ", ".join(f"window {w} {lang.upper()} {i}" for w, lang, i in missing)
        print(
            f"warning: {len(missing)} sentences inside gold windows are missing from the file and not scored: {listed}"
        )
        print("         (to mark a sentence as having no counterpart, keep its line as a group of its own)")
    report.evaluation(gold.evaluate(windows, groups), windows, _sides(run_dir), show=args.show)


def cmd_bench(args: argparse.Namespace) -> None:
    directions = bench.DIRECTIONS if args.direction == "both" else (args.direction,)
    benchmarks = [bench.load(name) for name in args.benchmarks]
    for direction in directions:
        src, tgt = bench.languages(direction)
        model = args.model.format(src=src, tgt=tgt)
        label = (args.label or model.rstrip("/").rsplit("/", 1)[-1]).format(src=src, tgt=tgt)
        translator = bench.Translator(model, args.device, args.batch_size, args.beams)
        for b in benchmarks:
            print(f"=== {label}: {b.name} {direction} ({len(b.en)} sentences)")
            result = {
                "label": label,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                **bench.evaluate(b, direction, translator),
                "code": provenance.code_info(),
                "env": provenance.env_info(translator.device),
                "timings": {"load_s": round(translator.load_s, 2), "translate_s": round(translator.translate_s, 2)},
            }
            path = args.out / f"{label}.{b.name}.{direction}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            report.bench_result(result)


def cmd_corpus(args: argparse.Namespace) -> None:
    out: Path = args.out
    config = corpus.CorpusConfig(args.min_sim, args.max_length_ratio, args.dev_fraction, args.dev_blocks, args.seed)
    sources = [
        corpus.Source.of(
            run.name,
            jsonl.read(run / "pairs.jsonl", Pair),
            *(jsonl.read(run / f"{lang}.sentences.jsonl", Sentence) for lang in LANGS),
        )
        for run in args.runs
    ]
    result = corpus.build(sources, [bench.load(name) for name in args.benchmarks], config)
    jsonl.write(out / "train.jsonl", result.train)
    jsonl.write(out / "dev.jsonl", result.dev)
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "config": asdict(config),
        "sources": {
            s.name: {"path": str(run), "pairs": len(s.pairs), "run": _read_json(run / "run.json")}
            for run, s in zip(args.runs, sources, strict=True)
        },
        "benchmarks_checked": args.benchmarks,
        "dropped": dict(result.dropped.most_common()),
        "train": corpus.stats(result.train),
        "dev": corpus.stats(result.dev),
        "contaminated": result.contaminated,
        "code": provenance.code_info(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    report.corpus(manifest)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cmd_results(args: argparse.Namespace) -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.dir.glob("*.json"))]
    report.results_table(rows)


def _sides(run_dir: Path) -> gold.Sides:
    return gold.Sides.of(*(jsonl.read(run_dir / f"{lang}.sentences.jsonl", Sentence) for lang in LANGS))


def _write_alignment(pairs_path: Path, result: Alignment) -> None:
    jsonl.write(pairs_path, result.pairs)
    jsonl.write(pairs_path.with_suffix(".unaligned.jsonl"), result.unaligned)


def _config(args: argparse.Namespace) -> AlignConfig:
    return AlignConfig(skip_cost=args.skip_cost, group_penalty=args.group_penalty, keep_verse=args.keep_verse)


def _embedder(args: argparse.Namespace) -> Embedder:
    return Embedder(args.model, args.device, args.batch_size)


type Command = Callable[[str, Callable[[argparse.Namespace], None], str], argparse.ArgumentParser]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="babel-tower", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(required=True, metavar="COMMAND")

    def command(name: str, fn: Callable[[argparse.Namespace], None], help: str) -> argparse.ArgumentParser:  # noqa: A002
        p = sub.add_parser(name, help=help, description=help)
        p.set_defaults(command=fn)
        return p

    for add in (_add_pipeline_commands, _add_review_commands, _add_gold_commands, _add_bench_commands):
        add(command)
    return parser


def _add_pipeline_commands(command: Command) -> None:
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


def _add_review_commands(command: Command) -> None:
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


def _add_gold_commands(command: Command) -> None:
    p = command("gold-template", cmd_gold_template, "pre-fill a gold alignment file from a run, to correct by hand")
    p.add_argument("run", type=Path, help="run directory (from `babel-tower run`)")
    p.add_argument("out", type=Path, help="gold file to write")
    p.add_argument("--windows", type=int, default=6, help="stretches of the book to check (default %(default)s)")
    p.add_argument("--size", type=int, default=25, help="English sentences per window (default %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite an existing gold file")

    p = command("eval", cmd_eval, "score a run against a gold alignment file")
    p.add_argument("gold", type=Path)
    p.add_argument("run", type=Path, help="run directory (from `babel-tower run`)")
    p.add_argument("--show", type=int, default=10, help="gold groups the run got wrong to print (default %(default)s)")


def _add_bench_commands(command: Command) -> None:
    p = command("bench", cmd_bench, "translate benchmark sets with a model and score the output")
    p.add_argument(
        "model",
        help="Hugging Face model id or local directory; {src} and {tgt} are filled in per direction, "
        "e.g. Helsinki-NLP/opus-mt-{src}-{tgt}",
    )
    p.add_argument("--direction", choices=[*bench.DIRECTIONS, "both"], default="both")
    p.add_argument("--benchmarks", nargs="+", choices=bench.BENCHMARKS, default=["literary", "flores-devtest"])
    p.add_argument("--label", help="name for this model in result files (default: last part of the model id)")
    p.add_argument("--out", type=Path, default=Path("data/results"), help="results directory (default %(default)s)")
    p.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beams", type=int, default=4, help="beam search width (default %(default)s)")

    p = command("corpus", cmd_corpus, "aligned pairs from one or more runs -> train/dev corpus for translation")
    p.add_argument("out", type=Path, help="corpus directory to write")
    p.add_argument("runs", type=Path, nargs="+", help="run directories (from `babel-tower run`)")
    defaults = corpus.CorpusConfig()
    p.add_argument("--min-sim", type=float, default=defaults.min_sim, help="default %(default)s")
    p.add_argument("--max-length-ratio", type=float, default=defaults.max_length_ratio, help="default %(default)s")
    p.add_argument("--dev-fraction", type=float, default=defaults.dev_fraction, help="default %(default)s")
    p.add_argument("--dev-blocks", type=int, default=defaults.dev_blocks, help="per run (default %(default)s)")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument(
        "--benchmarks", nargs="+", choices=bench.BENCHMARKS, default=list(bench.BENCHMARKS),
        help="benchmarks whose sentences must not appear in the corpus (default: all)",
    )  # fmt: skip

    p = command("results", cmd_results, "table of all benchmark results in a directory")
    p.add_argument("dir", type=Path, nargs="?", default=Path("data/results"))


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
