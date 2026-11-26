"""Look at alignment output without reading all of it."""

import random
from collections import Counter
from collections.abc import Sequence

from babel_tower.records import Pair
from babel_tower.report import heading, show_pair


def sample(pairs: Sequence[Pair], *, lo: float, hi: float, group_type: str | None, n: int, seed: int) -> None:
    """Show n random pairs with lo <= sim < hi, optionally of one group type such as "2-2"."""
    matching = [p for p in pairs if lo <= p.sim < hi and group_type in (None, p.type)]
    print(f"{len(matching)} pairs match")
    for p in random.Random(seed).sample(matching, min(n, len(matching))):
        show_pair(p)


def compare(a: Sequence[Pair], b: Sequence[Pair], n: int, seed: int) -> None:
    """Show the pairs that only one of two runs produced."""

    def key(p: Pair) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(p.en_ids), tuple(p.hu_ids)

    keys_a, keys_b = {key(p) for p in a}, {key(p) for p in b}
    only_a = [p for p in a if key(p) not in keys_b]
    only_b = [p for p in b if key(p) not in keys_a]
    print(f"A: {len(a)} pairs   B: {len(b)} pairs   identical in both: {len(a) - len(only_a)}")
    rng = random.Random(seed)
    for name, only in (("only in A", only_a), ("only in B", only_b)):
        mean = sum(p.sim for p in only) / len(only) if only else 0.0
        types = dict(sorted(Counter(p.type for p in only).items()))
        heading(f"{name}: {len(only)} pairs, mean sim {mean:.2f}, types {types}")
        for p in rng.sample(only, min(n, len(only))):
            show_pair(p)
