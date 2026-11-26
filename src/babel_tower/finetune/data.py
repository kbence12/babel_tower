"""From corpus pairs to batches a sequence-to-sequence model can train on.

A translation model never sees text, only token ids. Three things happen on the way:

1. Tokenising. OPUS-MT uses SentencePiece: a fixed vocabulary (~62k entries for en-hu) of
   whole words and word pieces, learned from the model's original training data. Frequent
   words are one token ("▁kutya"); rare or long ones are split ("▁házak" + "ban"). "▁" marks
   the start of a word. Every sentence ends with the end-of-sentence token </s>.

2. Source and target. The source sentence becomes `input_ids`, which the encoder reads. The
   target becomes `labels`: what the decoder must produce, one token at a time. During training
   the decoder is shown the correct previous tokens ("teacher forcing"). The model builds that
   input itself by shifting the labels one position right, so we only provide labels.

3. Batching. Sentences differ in length, but a batch is a rectangular tensor, so shorter
   sentences are padded. Source padding is masked out with `attention_mask` (0 = ignore). Label
   padding is set to -100, the index PyTorch's cross-entropy loss skips, so the model is not
   trained to predict padding.

Sorting by length before batching (as `batches` does) keeps padding, and wasted GPU work, low.
"""

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from babel_tower import jsonl, punct
from babel_tower.bench import Direction, languages
from babel_tower.corpus import CorpusPair

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100  # labels with this value do not count towards the loss


@dataclass(frozen=True, slots=True)
class Example:
    """One tokenised training pair."""

    input_ids: list[int]  # source tokens, ending in </s>
    labels: list[int]  # target tokens, ending in </s>


def load_pairs(corpus_dir: Path, split: str) -> list[CorpusPair]:
    return jsonl.read(corpus_dir / f"{split}.jsonl", CorpusPair)


def source_and_target(pairs: Sequence[CorpusPair], direction: Direction) -> tuple[list[str], list[str]]:
    """Source and target texts for one direction, with punctuation the vocabulary lacks mapped
    to characters it has (punct.py); otherwise every dialogue dash would be trained as <unk>."""
    src, tgt = languages(direction)
    side = {"en": [punct.to_model(p.en) for p in pairs], "hu": [punct.to_model(p.hu) for p in pairs]}
    return side[src], side[tgt]


def tokenize(
    tokenizer: PreTrainedTokenizerBase, sources: Sequence[str], targets: Sequence[str], max_length: int
) -> list[Example]:
    """Tokenise both sides. Anything beyond max_length tokens is cut off (see `length_report`)."""
    enc = tokenizer(list(sources), text_target=list(targets), truncation=True, max_length=max_length)
    return [Example(i, lab) for i, lab in zip(enc["input_ids"], enc["labels"], strict=True)]


def collate(examples: Sequence[Example], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a list of examples into one batch of rectangular tensors."""
    import torch

    width_in = max(len(e.input_ids) for e in examples)
    width_out = max(len(e.labels) for e in examples)
    input_ids = [e.input_ids + [pad_id] * (width_in - len(e.input_ids)) for e in examples]
    attention_mask = [[1] * len(e.input_ids) + [0] * (width_in - len(e.input_ids)) for e in examples]
    labels = [e.labels + [IGNORE_INDEX] * (width_out - len(e.labels)) for e in examples]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def batches(
    examples: Sequence[Example], batch_size: int, pad_id: int, rng: random.Random | None = None
) -> Iterator[dict[str, torch.Tensor]]:
    """Group examples of similar length into batches.

    With an rng (training), the batches come out in random order, so each epoch sees them
    differently; the grouping by length stays. Without one (evaluation), order is fixed.
    """
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].input_ids))
    groups = [order[k : k + batch_size] for k in range(0, len(order), batch_size)]
    if rng is not None:
        rng.shuffle(groups)
    for group in groups:
        yield collate([examples[i] for i in group], pad_id)


def length_report(examples: Sequence[Example], tokenizer: PreTrainedTokenizerBase, max_length: int) -> dict[str, Any]:
    """Token counts per side, how many examples hit max_length, and how often <unk> appears."""

    def summary(seqs: list[list[int]]) -> dict[str, Any]:
        lengths = sorted(len(s) for s in seqs)
        n = len(lengths)
        return {
            "tokens": sum(lengths),
            "median": lengths[n // 2],
            "p95": lengths[min(n - 1, int(0.95 * n))],
            "max": lengths[-1],
            "truncated": sum(length >= max_length for length in lengths),
            "unknown": sum(s.count(tokenizer.unk_token_id) for s in seqs),
        }

    return {"source": summary([e.input_ids for e in examples]), "target": summary([e.labels for e in examples])}
