"""`babel-tower finetune ...` commands."""

import argparse
import random
from pathlib import Path
from typing import TYPE_CHECKING

from babel_tower.bench import DIRECTIONS, Direction, languages
from babel_tower.finetune import data, train
from babel_tower.report import clip, heading

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-{src}-{tgt}"


def add_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(required=True, metavar="STEP")

    p = sub.add_parser("data", help="show how a corpus is tokenised and batched for a model")
    p.set_defaults(command=cmd_data)
    p.add_argument("corpus", type=Path, help="corpus directory (from `babel-tower corpus`)")
    p.add_argument("--direction", choices=DIRECTIONS, default="en-hu")
    p.add_argument("--model", default=DEFAULT_MODEL, help="default %(default)s; {src}/{tgt} are filled in")
    p.add_argument("--max-length", type=int, default=256, help="tokens per side (default %(default)s)")
    p.add_argument("--batch-size", type=int, default=16)

    defaults = train.TrainConfig()
    p = sub.add_parser("train", help="fine-tune a model on a corpus, evaluating after every epoch")
    p.set_defaults(command=cmd_train)
    p.add_argument("corpus", type=Path, help="corpus directory (from `babel-tower corpus`)")
    p.add_argument("out", type=Path, help="directory to save the fine-tuned model in")
    p.add_argument("--direction", choices=DIRECTIONS, default=defaults.direction)
    p.add_argument("--model", default=defaults.base_model, help="model to start from (default %(default)s)")
    p.add_argument("--lr", type=float, default=defaults.learning_rate, help="peak learning rate (default %(default)s)")
    p.add_argument("--epochs", type=int, default=defaults.epochs, help="default %(default)s")
    p.add_argument("--batch-size", type=int, default=defaults.batch_size, help="default %(default)s")
    p.add_argument(
        "--warmup", type=float, default=defaults.warmup_fraction, help="share of steps (default %(default)s)"
    )
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay, help="default %(default)s")
    p.add_argument("--freeze", choices=train.FREEZE_CHOICES, default=defaults.freeze, help="default %(default)s")
    p.add_argument(
        "--flores", type=int, default=defaults.flores_sentences,
        help="flores-dev sentences to score after each epoch, to watch for forgetting (default %(default)s)",
    )  # fmt: skip
    p.add_argument("--seed", type=int, default=defaults.seed)


def cmd_train(args: argparse.Namespace) -> None:
    config = train.TrainConfig(
        direction=args.direction,
        base_model=args.model,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_fraction=args.warmup,
        weight_decay=args.weight_decay,
        freeze=args.freeze,
        flores_sentences=args.flores,
        seed=args.seed,
    )
    train.train(args.corpus, args.out, config)


def model_name(template: str, direction: Direction) -> str:
    src, tgt = languages(direction)
    return template.format(src=src, tgt=tgt)


def cmd_data(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name(args.model, args.direction))
    pad_id = tokenizer.pad_token_id
    src_lang, tgt_lang = languages(args.direction)

    for split in ("train", "dev"):
        pairs = data.load_pairs(args.corpus, split)
        sources, targets = data.source_and_target(pairs, args.direction)
        examples = data.tokenize(tokenizer, sources, targets, args.max_length)
        report = data.length_report(examples, tokenizer, args.max_length)
        print(f"=== {split}: {len(examples)} pairs, {args.direction}")
        for side, lang in (("source", src_lang), ("target", tgt_lang)):
            s = report[side]
            words = sum(len(t.split()) for t in (sources if side == "source" else targets))
            print(
                f"  {side} ({lang}): {s['tokens']} tokens for {words} words ({s['tokens'] / words:.2f} per word)   "
                f"median {s['median']}  p95 {s['p95']}  max {s['max']}   "
                f"truncated {s['truncated']}  <unk> {s['unknown']}"
            )
        if split == "train":
            _show_tokens(tokenizer, sources, targets)
            _show_padding(examples, args.batch_size, pad_id)


def _show_tokens(tokenizer: PreTrainedTokenizerBase, sources: list[str], targets: list[str]) -> None:
    heading("how 3 random pairs are tokenised")
    for i in random.Random(1).sample(range(len(sources)), 3):
        enc = tokenizer(sources[i], text_target=targets[i])
        src_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        tgt_tokens = tokenizer.convert_ids_to_tokens(enc["labels"])
        print(f"  SRC {clip(sources[i], 90)}\n      {' '.join(src_tokens)}")
        print(f"  TGT {clip(targets[i], 90)}\n      {' '.join(tgt_tokens)}\n")


def _show_padding(examples: list[data.Example], batch_size: int, pad_id: int) -> None:
    """Why `batches` sorts by length: count padding with and without sorting."""

    def padding_share(groups: list[list[data.Example]]) -> float:
        real = sum(len(e.input_ids) for g in groups for e in g)
        total = sum(max(len(e.input_ids) for e in g) * len(g) for g in groups)
        return 1 - real / total

    shuffled = examples[:]
    random.Random(0).shuffle(shuffled)
    by_length = sorted(examples, key=lambda e: len(e.input_ids))

    def chunks(xs: list[data.Example]) -> list[list[data.Example]]:
        return [xs[k : k + batch_size] for k in range(0, len(xs), batch_size)]

    first = next(data.batches(examples, batch_size, pad_id))
    heading(f"batching, {batch_size} pairs per batch")
    print("  one batch: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in first.items()))
    print(f"  source tokens that are padding: random batches {padding_share(chunks(shuffled)):.0%}, "
          f"length-sorted batches {padding_share(chunks(by_length)):.0%}")  # fmt: skip
