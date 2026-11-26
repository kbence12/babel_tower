"""Human-readable reports for each pipeline stage, meant for checking a run by eye."""

import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from babel_tower.align import Alignment
from babel_tower.records import LANGS, Pair, Paragraph, Sentence

WIDTH = 100
ENDS_SENTENCE = re.compile(r"[.!?…\"”’»)]$")


def clip(text: str, width: int = WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def show_pair(p: Pair) -> None:
    print(f"  [{p.type} sim={p.sim:.2f}]")
    print(f"     EN {clip(p.en)}")
    print(f"     HU {clip(p.hu)}")


def heading(title: str) -> None:
    print(f"--- {title} ---")


def paragraphs(rows: Sequence[Paragraph]) -> None:
    blob = "".join(r.text for r in rows)
    good = sum(blob.count(c) for c in "őűŐŰ")
    bad = sum(blob.count(c) for c in "õûÕÛ")  # typical Latin-1/Latin-2 mangling of ő/ű
    print(
        f"blocks: {len(rows)}  words: {sum(len(r.text.split()) for r in rows)}  "
        f"headings: {sum(r.is_heading for r in rows)}  multi-line (verse?): {sum(r.is_verse for r in rows)}"
    )
    print(f"ő/ű: {good}   õ/û: {bad}   (a high second number on Hungarian text means an encoding problem)")
    for title, part in (("first 5", rows[:5]), ("last 5", rows[-5:])):
        heading(title)
        for r in part:
            print(f"[{r.tag}] {r.text[:WIDTH]}")


def sentences(rows: Sequence[Sentence], verbose: bool = False) -> None:
    n_words = sorted(len(s.text.split()) for s in rows)
    if not n_words:
        print("segments: 0")
        return

    def pct(p: float) -> int:
        return n_words[min(len(n_words) - 1, int(p * len(n_words)))]

    body = [s for s in rows if not s.is_heading and not s.verse]
    no_end = [s for s in body if not ENDS_SENTENCE.search(s.text)]
    short = [s for s in body if len(s.text.split()) <= 2]
    print(f"segments: {len(rows)}   words/segment: p5={pct(0.05)} median={pct(0.5)} p95={pct(0.95)} max={n_words[-1]}")
    print(
        f"> 60 words (long, often genuine run-ons): {sum(n > 60 for n in n_words)}   "
        f"<= 2 words: {sum(n <= 2 for n in n_words)}   "
        f"no end punctuation (excl. headings/verse): {len(no_end)}   verse/unsplit: {sum(s.verse for s in rows)}"
    )
    if not verbose:
        return

    rng = random.Random(0)
    heading("15 random segments")
    for s in rng.sample(rows, min(15, len(rows))):
        print(f"{s.id:5d} {s.text[:WIDTH]}")
    heading("5 longest")
    for s in sorted(rows, key=lambda s: -len(s.text.split()))[:5]:
        print(f"{s.id:5d} ({len(s.text.split())}w) {s.text[:WIDTH]}")
    heading("5 shortest non-heading")
    for s in sorted((s for s in rows if not s.is_heading), key=lambda s: len(s.text))[:5]:
        print(f"{s.id:5d} {s.text[:WIDTH]}")
    heading(f"{min(12, len(short))} random short segments, with the end of the segment before")
    for s in rng.sample(short, min(12, len(short))):
        before = rows[s.id - 1].text[-50:] if s.id else ""  # ids are list positions
        print(f"{s.id:5d} {s.text!r:24} <- ...{before}")
    heading(f"{min(8, len(no_end))} segments without end punctuation")
    for s in rng.sample(no_end, min(8, len(no_end))):
        print(f"{s.id:5d} ...{s.text[-70:]}")


@dataclass(frozen=True, slots=True)
class SimQuantiles:
    p10: float
    p25: float
    median: float
    p75: float
    mean: float
    below_06: int
    below_07: int

    @classmethod
    def of(cls, pairs: Sequence[Pair]) -> SimQuantiles | None:
        if not pairs:
            return None
        sims = np.array([p.sim for p in pairs])
        p10, p25, median, p75 = (round(float(x), 4) for x in np.percentile(sims, [10, 25, 50, 75]))
        return cls(
            p10, p25, median, p75, round(float(sims.mean()), 4), int((sims < 0.6).sum()), int((sims < 0.7).sum())
        )


def alignment_stats(a: Alignment) -> dict[str, Any]:
    """Headline numbers for a run, as written to run.json."""
    stats: dict[str, Any] = {
        "segments": {"en": a.n_en, "hu": a.n_hu},
        "pairs": len(a.pairs),
        "types": dict(sorted(Counter(p.type for p in a.pairs).items())),
        "coverage": {
            "en": round(sum(len(p.en_ids) for p in a.pairs) / max(a.n_en, 1), 4),
            "hu": round(sum(len(p.hu_ids) for p in a.pairs) / max(a.n_hu, 1), 4),
        },
        "unaligned": {lang: sum(u.side == lang for u in a.unaligned) for lang in LANGS},
    }
    if q := SimQuantiles.of(a.pairs):
        stats["sim"] = {
            "p10": q.p10, "p25": q.p25, "median": q.median, "p75": q.p75, "mean": q.mean,
            "below_0.6": q.below_06, "below_0.7": q.below_07,
        }  # fmt: skip
    return stats


def alignment(a: Alignment) -> None:
    stats = alignment_stats(a)
    cov, un = stats["coverage"], stats["unaligned"]
    print(f"segments aligned: EN {a.n_en}  HU {a.n_hu}   pairs: {len(a.pairs)}   types: {stats['types']}")
    print(f"coverage: EN {cov['en']:.1%}  HU {cov['hu']:.1%}   unaligned: EN {un['en']}  HU {un['hu']}")
    if q := SimQuantiles.of(a.pairs):
        print(
            f"cosine: p10={q.p10:.2f} p25={q.p25:.2f} median={q.median:.2f} p75={q.p75:.2f} mean={q.mean:.2f}   "
            f"below 0.6: {q.below_06}   below 0.7: {q.below_07}"
        )

    rng = random.Random(0)
    by_sim = sorted(a.pairs, key=lambda p: p.sim)
    mid = by_sim[int(len(by_sim) * 0.3) : int(len(by_sim) * 0.7)]
    for title, part in (
        ("first 3 pairs", a.pairs[:3]),
        ("last 3 pairs", a.pairs[-3:]),
        ("6 lowest-similarity pairs", by_sim[:6]),
        ("6 random mid-range pairs", rng.sample(mid, min(6, len(mid)))),
    ):
        heading(title)
        for p in part:
            show_pair(p)
    heading("4 random unaligned segments per side")
    for lang in LANGS:
        side = [u for u in a.unaligned if u.side == lang]
        for u in rng.sample(side, min(4, len(side))):
            print(f"  [{lang}] {clip(u.text)}")
