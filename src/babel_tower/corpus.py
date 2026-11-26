"""Aligned pairs from one or more runs -> a parallel corpus for training a translator.

1. Split each run's pairs into train and dev by position: dev is a few contiguous blocks
   spread over the book, so dev sentences never sit right next to the train sentences that
   share their context.
2. Filter both splits: similarity below min_sim (likely misalignments or very free
   translations), implausible length ratios, segments without letters.
3. Drop exact duplicates, and any pair containing a sentence that is also in a benchmark,
   so that benchmark scores stay honest.
"""

import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from babel_tower.bench import Benchmark
from babel_tower.records import LANGS, Lang, Pair, Sentence


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    min_sim: float = 0.5
    max_length_ratio: float = 3.0  # longer side / shorter side, in characters
    dev_fraction: float = 0.05
    dev_blocks: int = 4  # per source
    seed: int = 0


@dataclass(frozen=True, slots=True)
class CorpusPair:
    """One training example; source and pair point back to the run it came from."""

    en: str
    hu: str
    sim: float
    type: str
    source: str
    pair: int


@dataclass(frozen=True, slots=True)
class Source:
    """One run's output: its pairs and the sentences they were made of."""

    name: str
    pairs: list[Pair]
    sentences: dict[Lang, dict[int, str]]

    @classmethod
    def of(cls, name: str, pairs: list[Pair], en: Iterable[Sentence], hu: Iterable[Sentence]) -> Source:
        return cls(name, pairs, {"en": {s.id: s.text for s in en}, "hu": {s.id: s.text for s in hu}})


@dataclass(slots=True)
class Corpus:
    train: list[CorpusPair] = field(default_factory=list)
    dev: list[CorpusPair] = field(default_factory=list)
    dropped: Counter[str] = field(default_factory=Counter)  # reason -> pairs dropped
    contaminated: list[dict[str, Any]] = field(default_factory=list)


def build(sources: Sequence[Source], benchmarks: Sequence[Benchmark], config: CorpusConfig) -> Corpus:
    benchmark_sentences = {lang: {normalize(s) for b in benchmarks for s in b.side(lang)} for lang in LANGS}
    corpus = Corpus()
    seen: set[tuple[str, str]] = set()
    rng = random.Random(config.seed)
    for source in sources:
        dev_positions = _dev_positions(len(source.pairs), config, rng)
        for k, p in enumerate(source.pairs):
            if reason := _rejection(p, config):
                corpus.dropped[reason] += 1
                continue
            if hits := _benchmark_hits(source, p, benchmark_sentences):
                corpus.dropped["sentence also in a benchmark"] += 1
                corpus.contaminated.append({"source": source.name, "pair": p.pair, "sentences": hits})
                continue
            if (p.en, p.hu) in seen:
                corpus.dropped["duplicate"] += 1
                continue
            seen.add((p.en, p.hu))
            split = corpus.dev if k in dev_positions else corpus.train
            split.append(CorpusPair(p.en, p.hu, p.sim, p.type, source.name, p.pair))
    return corpus


def normalize(text: str) -> str:
    """Lowercase letters and digits only, for matching sentences regardless of punctuation."""
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def stats(pairs: Sequence[CorpusPair]) -> dict[str, Any]:
    out: dict[str, Any] = {"pairs": len(pairs)}
    for lang in LANGS:
        words = [w for p in pairs for w in normalize(p.en if lang == "en" else p.hu).split()]
        out[lang] = {"words": len(words), "word_types": len(set(words))}
    return out


def _rejection(p: Pair, config: CorpusConfig) -> str | None:
    if p.sim < config.min_sim:
        return f"similarity below {config.min_sim}"
    if not normalize(p.en) or not normalize(p.hu):
        return "no letters or digits"
    shorter, longer = sorted((len(p.en), len(p.hu)))
    if longer > config.max_length_ratio * shorter:
        return f"length ratio above {config.max_length_ratio}"
    return None


def _benchmark_hits(source: Source, p: Pair, benchmark: dict[Lang, set[str]]) -> list[str]:
    """Sentences of the pair that also occur, after normalising, in a benchmark."""
    return [
        text
        for lang, ids in zip(LANGS, (p.en_ids, p.hu_ids), strict=True)
        for text in (source.sentences[lang][i] for i in ids)
        if normalize(text) in benchmark[lang]
    ]


def _dev_positions(n: int, config: CorpusConfig, rng: random.Random) -> set[int]:
    """dev_blocks contiguous blocks covering about dev_fraction of n, one per equal slice of the book."""
    if not n or config.dev_fraction <= 0:
        return set()
    block = max(1, round(n * config.dev_fraction / config.dev_blocks))
    slice_len = n / config.dev_blocks
    positions: set[int] = set()
    for k in range(config.dev_blocks):
        lo, hi = int(k * slice_len), int((k + 1) * slice_len)
        start = rng.randrange(lo, max(lo + 1, hi - block + 1))
        positions.update(range(start, min(start + block, hi)))
    return positions
