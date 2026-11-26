"""Sentences -> aligned groups, with window embeddings and a dynamic-programming aligner.

  1. Drop verse blocks (unless keep_verse) and segments with no letters or digits.
  2. Embed every window of 1..3 consecutive segments on each side. Quote marks and dialogue
     dashes are removed from what the embedder sees, because they differ by language.
  3. Find the cheapest monotonic path through the (English x Hungarian) grid. Moves are the
     groups in MOVES, or skipping one segment on either side:
        cost(group) = (1 - cosine) * (n + m) + group_penalty * (n + m - 2)
        cost(skip)  = skip_cost per skipped segment
     So skip_cost 0.5 means roughly "pairs with cosine below ~0.5 are better left unaligned".
     Bigger windows score higher cosines, which would merge consecutive 1-1 pairs into 2-2
     groups; the group penalty nudges the aligner towards the finest valid split.

The DP is exact over the full grid, computed row by row with numpy: fine for a book, but
for much larger inputs it should be restricted to a band around the diagonal.
"""

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np

from babel_tower.embed import Embedder, Windows, embed_windows
from babel_tower.records import Pair, Sentence, Unaligned

MOVES: Final = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
MAX_GROUP: Final = max(max(move) for move in MOVES)
SKIP_EN: Final = len(MOVES)  # back-pointer codes after the move indices
SKIP_HU: Final = len(MOVES) + 1
NOT_SEEN_BY_EMBEDDER = re.compile(r"[“”„«»\"–—]")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class AlignConfig:
    skip_cost: float = 0.5
    group_penalty: float = 0.1
    keep_verse: bool = False


class Step(NamedTuple):
    """One move of the path: n_en English segments from en_start against n_hu Hungarian ones.
    A skip has n_en == 0 or n_hu == 0."""

    en_start: int
    n_en: int
    hu_start: int
    n_hu: int


@dataclass(frozen=True, slots=True)
class Alignment:
    pairs: list[Pair]
    unaligned: list[Unaligned]
    n_en: int  # segments that took part in the alignment
    n_hu: int
    embed_s: float
    dp_s: float


def embed_view(text: str) -> str:
    """What the embedder sees: quote marks and dialogue dashes replaced by spaces."""
    return WHITESPACE.sub(" ", NOT_SEEN_BY_EMBEDDER.sub(" ", text)).strip()


def alignable(sentences: Sequence[Sentence], keep_verse: bool) -> list[Sentence]:
    return [s for s in sentences if (keep_verse or not s.verse) and any(ch.isalnum() for ch in embed_view(s.text))]


def align(
    en_sentences: Sequence[Sentence],
    hu_sentences: Sequence[Sentence],
    config: AlignConfig,
    embedder: Embedder,
    cache_dir: Path,
) -> Alignment:
    en = alignable(en_sentences, config.keep_verse)
    hu = alignable(hu_sentences, config.keep_verse)

    t = time.perf_counter()
    win_en = embed_windows([embed_view(s.text) for s in en], MAX_GROUP, "en", cache_dir, embedder)
    win_hu = embed_windows([embed_view(s.text) for s in hu], MAX_GROUP, "hu", cache_dir, embedder)
    embed_s = time.perf_counter() - t

    t = time.perf_counter()
    steps = dp_align(win_en, win_hu, len(en), len(hu), skip_cost=config.skip_cost, group_penalty=config.group_penalty)
    dp_s = time.perf_counter() - t

    pairs: list[Pair] = []
    unaligned: list[Unaligned] = []
    for step in steps:
        en_group = en[step.en_start : step.en_start + step.n_en]
        hu_group = hu[step.hu_start : step.hu_start + step.n_hu]
        if en_group and hu_group:
            sim = float(win_en[step.n_en][step.en_start] @ win_hu[step.n_hu][step.hu_start])
            pairs.append(Pair(
                pair=len(pairs),
                type=f"{step.n_en}-{step.n_hu}",
                sim=round(sim, 4),
                en_ids=[s.id for s in en_group],
                hu_ids=[s.id for s in hu_group],
                en=" ".join(s.text for s in en_group),
                hu=" ".join(s.text for s in hu_group),
            ))  # fmt: skip
        else:
            unaligned += [Unaligned("en", s.id, s.text) for s in en_group]
            unaligned += [Unaligned("hu", s.id, s.text) for s in hu_group]
    return Alignment(pairs, unaligned, len(en), len(hu), round(embed_s, 3), round(dp_s, 3))


def dp_align(
    win_en: Windows, win_hu: Windows, n_en: int, n_hu: int, *, skip_cost: float, group_penalty: float
) -> list[Step]:
    """Cheapest monotonic alignment of n_en English and n_hu Hungarian segments.

    win_en[n][i] / win_hu[m][j] is the unit-norm embedding of the window of n (m) segments
    starting at i (j). cost[i, j] is the cheapest way to align the first i English and the
    first j Hungarian segments; each row is computed from the rows above it in one vector pass.
    """
    cost = np.empty((n_en + 1, n_hu + 1), dtype=np.float64)
    back = np.empty((n_en + 1, n_hu + 1), dtype=np.int8)
    skips = np.arange(n_hu + 1, dtype=np.float64) * skip_cost
    cost[0] = skips
    back[0] = SKIP_HU

    for i in range(1, n_en + 1):
        best = cost[i - 1] + skip_cost  # skip English segment i-1
        move = np.full(n_hu + 1, SKIP_EN, dtype=np.int8)
        for k, (n, m) in enumerate(MOVES):
            if i < n or n_hu < m:
                continue
            sims = win_hu[m] @ win_en[n][i - n]  # one cosine per Hungarian window start
            candidate = np.full(n_hu + 1, np.inf)
            candidate[m:] = cost[i - n][: n_hu + 1 - m] + (1.0 - sims) * (n + m) + group_penalty * (n + m - 2)
            better = candidate < best
            best = np.where(better, candidate, best)
            move = np.where(better, k, move).astype(np.int8)
        # skipping Hungarian segments is a running minimum along the row:
        #   cost[i, j] = min over k <= j of best[k] + (j - k) * skip_cost
        scan = np.minimum.accumulate(best - skips) + skips
        move = np.where(scan < best - 1e-9, SKIP_HU, move).astype(np.int8)
        cost[i] = scan
        back[i] = move

    return _trace_back(back, n_en, n_hu)


def _trace_back(back: np.ndarray, i: int, j: int) -> list[Step]:
    steps = []
    while i > 0 or j > 0:
        code = int(back[i, j])
        if code == SKIP_EN:
            n, m = 1, 0
        elif code == SKIP_HU:
            n, m = 0, 1
        else:
            n, m = MOVES[code]
        i, j = i - n, j - m
        steps.append(Step(i, n, j, m))
    steps.reverse()
    return steps
