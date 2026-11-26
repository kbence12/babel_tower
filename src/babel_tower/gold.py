"""Hand-checked reference alignments ("gold"), and scoring a run against them.

A gold file holds a few windows of consecutive sentences, each split into groups:

    === window 1 | EN 12-13 | HU 14-15
    EN 12 | Good morning.
    HU 14 | Jó reggelt.
    ---
    EN 13 | Lovely day.
    ---
    HU 15 | Mondta.

Lines between two "---" (or "===") lines form one group: its EN sentences are a translation of
its HU sentences. A group with lines from only one side means those sentences have no
counterpart. `write_template` pre-fills a file from an existing run; correcting it means
moving lines between groups, adding "---" to split a group, or deleting one to merge.
"""

import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from babel_tower.records import LANGS, Lang, Pair, Sentence, Unaligned

HEADER = """\
# babel-tower gold alignment
#
# Lines between two "---" (or "===") lines are one group: its EN sentences translate its HU
# sentences. A group with lines from only one side means those sentences have no counterpart.
#
# This file was pre-filled from an automatic run. Correct it by moving lines between groups,
# adding "---" to split a group, or deleting a "---" to merge two. Keep every line, and keep the
# EN lines and the HU lines each in their original order. Lines starting with "#" are comments.
"""
WINDOW = re.compile(r"=== window (\d+)((?: \| (?:EN|HU) \d+-\d+)*)")
EXTENT = re.compile(r"(EN|HU) (\d+)-(\d+)")
LINE = re.compile(r"(EN|HU) (\d+) \| ?(.*)")


class GoldFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Group:
    en: tuple[int, ...]
    hu: tuple[int, ...]

    @property
    def is_pair(self) -> bool:
        return bool(self.en and self.hu)

    def links(self) -> set[tuple[int, int]]:
        return {(e, h) for e in self.en for h in self.hu}

    def side(self, lang: Lang) -> tuple[int, ...]:
        return self.en if lang == "en" else self.hu

    def sentences(self) -> set[tuple[Lang, int]]:
        en: set[tuple[Lang, int]] = {("en", i) for i in self.en}
        return en | {("hu", i) for i in self.hu}


@dataclass(slots=True)
class Window:
    number: int
    # first and last sentence id per side, as the template wrote them; a sentence in this range
    # but not in the file was deleted by mistake (deleting a line does not mean "no counterpart")
    extent: dict[Lang, tuple[int, int]] = field(default_factory=dict)
    groups: list[Group] = field(default_factory=list)
    texts: dict[tuple[Lang, int], str] = field(default_factory=dict)

    def ids(self, lang: Lang) -> set[int]:
        return {i for g in self.groups for i in g.side(lang)}


def path_groups(pairs: Sequence[Pair], unaligned: Iterable[Unaligned]) -> list[Group]:
    """The aligner's path as one ordered list of groups, unaligned sentences as one-sided groups."""
    pending: dict[Lang, list[int]] = {"en": [], "hu": []}
    for u in unaligned:
        pending[u.side].append(u.id)
    queue = {lang: sorted(ids, reverse=True) for lang, ids in pending.items()}  # pop() takes the smallest

    def skipped_before(lang: Lang, limit: float) -> list[Group]:
        out = []
        while queue[lang] and queue[lang][-1] < limit:
            i = queue[lang].pop()
            out.append(Group((i,), ()) if lang == "en" else Group((), (i,)))
        return out

    groups: list[Group] = []
    for p in pairs:
        groups += skipped_before("en", p.en_ids[0]) + skipped_before("hu", p.hu_ids[0])
        groups.append(Group(tuple(p.en_ids), tuple(p.hu_ids)))
    return groups + skipped_before("en", float("inf")) + skipped_before("hu", float("inf"))


def pick_windows(groups: Sequence[Group], n_windows: int, size: int, seed: int) -> list[list[Group]]:
    """n_windows runs of consecutive groups with about `size` English sentences each, one from
    each of n_windows equal slices of the book, at a random offset inside its slice."""
    rng = random.Random(seed)
    slice_len = len(groups) / n_windows
    windows = []
    for k in range(n_windows):
        lo, hi = int(k * slice_len), int((k + 1) * slice_len)
        start = rng.randrange(lo, max(lo + 1, hi - size))
        window: list[Group] = []
        n_en = 0
        for g in groups[start:hi]:
            if n_en >= size:
                break
            window.append(g)
            n_en += len(g.en)
        windows.append(window)
    return windows


@dataclass(frozen=True, slots=True)
class Sides:
    """Sentence texts by id, for both languages."""

    en: Mapping[int, str]
    hu: Mapping[int, str]

    @classmethod
    def of(cls, en: Iterable[Sentence], hu: Iterable[Sentence]) -> Self:
        return cls({s.id: s.text for s in en}, {s.id: s.text for s in hu})

    def text(self, lang: Lang, i: int) -> str:
        return (self.en if lang == "en" else self.hu)[i]


def write_template(path: Path, windows: Sequence[Sequence[Group]], sides: Sides) -> None:
    lines = [HEADER]
    for number, window in enumerate(windows, 1):
        extents = "".join(
            f" | {lang.upper()} {min(ids)}-{max(ids)}"
            for lang in LANGS
            if (ids := [i for g in window for i in g.side(lang)])
        )
        lines.append(f"=== window {number}{extents}")
        for k, g in enumerate(window):
            if k:
                lines.append("---")
            lines += [f"EN {i} | {sides.text('en', i)}" for i in g.en]
            lines += [f"HU {i} | {sides.text('hu', i)}" for i in g.hu]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse(path: Path) -> list[Window]:
    windows: list[Window] = []
    en: list[int] = []
    hu: list[int] = []

    def close_group() -> None:
        if en or hu:
            windows[-1].groups.append(Group(tuple(en), tuple(hu)))
        en.clear()
        hu.clear()

    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if m := WINDOW.fullmatch(line):
            if windows:
                close_group()
            extent: dict[Lang, tuple[int, int]] = {
                ("en" if side == "EN" else "hu"): (int(lo), int(hi)) for side, lo, hi in EXTENT.findall(m.group(2))
            }
            windows.append(Window(int(m.group(1)), extent=extent))
        elif not windows:
            raise GoldFormatError(f"{path}:{n}: expected '=== window N' before any group")
        elif line == "---":
            close_group()
        elif m := LINE.fullmatch(line):
            lang: Lang = "en" if m.group(1) == "EN" else "hu"
            i = int(m.group(2))
            (en if lang == "en" else hu).append(i)
            windows[-1].texts[lang, i] = m.group(3)
        else:
            raise GoldFormatError(f"{path}:{n}: cannot read line: {raw!r}")
    if windows:
        close_group()
    for w in windows:
        _validate(path, w)
    return windows


def _validate(path: Path, w: Window) -> None:
    for lang in ("en", "hu"):
        ids = [i for g in w.groups for i in (g.en if lang == "en" else g.hu)]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise GoldFormatError(f"{path}: window {w.number}: {lang.upper()} {dupes} appear more than once")
        if ids != sorted(ids):
            raise GoldFormatError(f"{path}: window {w.number}: {lang.upper()} lines are out of order")


def check_texts(windows: Iterable[Window], sides: Sides) -> list[str]:
    """Gold sentences whose text differs from the run's: the run used a different sentence split."""
    problems = []
    for w in windows:
        for (lang, i), text in w.texts.items():
            actual = (sides.en if lang == "en" else sides.hu).get(i)
            if actual != text:
                problems.append(f"window {w.number} {lang.upper()} {i}: gold {text!r} != run {actual!r}")
    return problems


@dataclass(frozen=True, slots=True)
class Score:
    """Precision / recall over some kind of item (links, groups or skipped sentences)."""

    correct: int
    predicted: int
    gold: int

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 1.0

    @property
    def recall(self) -> float:
        return self.correct / self.gold if self.gold else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @classmethod
    def of[T](cls, predicted: set[T], gold: set[T]) -> Self:
        return cls(len(predicted & gold), len(predicted), len(gold))


@dataclass(frozen=True, slots=True)
class Disagreement:
    """A stretch of a window where gold and run group the same sentences differently."""

    window: int
    gold: list[Group]
    run: list[Group]


@dataclass(frozen=True, slots=True)
class Evaluation:
    links: Score  # (en id, hu id) pairs that are translations of each other
    groups: Score  # aligned groups that match a gold group exactly
    skipped: Score  # sentences left without a counterpart
    disagreements: list[Disagreement]


def evaluate(windows: Iterable[Window], predicted: Sequence[Group]) -> Evaluation:
    """Score predicted groups against gold, counting only sentences inside the gold windows.

    A predicted link with one end outside a window counts as wrong: gold says where that
    in-window sentence belongs, and it is not there.
    """
    pred_links: set[tuple[int, int]] = set()
    gold_links: set[tuple[int, int]] = set()
    pred_groups: set[tuple[int, Group]] = set()
    gold_groups: set[tuple[int, Group]] = set()
    pred_skips: set[tuple[Lang, int]] = set()
    gold_skips: set[tuple[Lang, int]] = set()
    disagreements = []

    for w in windows:
        en_ids, hu_ids = w.ids("en"), w.ids("hu")
        run = [g for g in predicted if en_ids.intersection(g.en) or hu_ids.intersection(g.hu)]
        gold = _singleton_skips(w.groups)
        for g in run:
            if g.is_pair:
                pred_groups.add((w.number, g))
                pred_links |= {(e, h) for e, h in g.links() if e in en_ids or h in hu_ids}
            else:
                pred_skips |= g.sentences()
        for g in gold:
            if g.is_pair:
                gold_groups.add((w.number, g))
                gold_links |= g.links()
            else:
                gold_skips |= g.sentences()
        disagreements += _disagreements(w.number, gold, run)

    return Evaluation(
        links=Score.of(pred_links, gold_links),
        groups=Score.of(pred_groups, gold_groups),
        skipped=Score.of(pred_skips, gold_skips),
        disagreements=disagreements,
    )


def missing_lines(windows: Iterable[Window], predicted: Sequence[Group]) -> list[tuple[int, Lang, int]]:
    """Sentences the run aligned that fall inside a gold window but are not in it (deleted lines)."""
    known = {lang: {i for g in predicted for i in g.side(lang)} for lang in LANGS}
    missing = []
    for w in windows:
        for lang in LANGS:
            ids = w.ids(lang)
            lo, hi = w.extent.get(lang) or ((min(ids), max(ids)) if ids else (0, -1))
            missing += [(w.number, lang, i) for i in range(lo, hi + 1) if i in known[lang] - ids]
    return missing


def _singleton_skips(groups: Iterable[Group]) -> list[Group]:
    """A one-sided gold group of several lines means each of them has no counterpart."""
    out: list[Group] = []
    for g in groups:
        if g.is_pair:
            out.append(g)
        else:
            out += [Group((i,), ()) for i in g.en] + [Group((), (i,)) for i in g.hu]
    return out


def _disagreements(window: int, gold: list[Group], run: list[Group]) -> list[Disagreement]:
    """Gold and run groups that do not match, clustered into stretches that share sentences.

    Both lists are in path order, and so are the groups inside each disagreement.
    """
    clusters: list[Disagreement] = []
    for g in gold + run:
        if (g in gold) == (g in run):
            continue  # matched
        new = Disagreement(window, [g] if g in gold else [], [g] if g in run else [])
        for c in [c for c in clusters if _sentences(c) & g.sentences()]:
            clusters.remove(c)
            new.gold.extend(c.gold)
            new.run.extend(c.run)
        clusters.append(new)
    for c in clusters:
        c.gold.sort(key=gold.index)
        c.run.sort(key=run.index)
    return sorted(clusters, key=lambda c: gold.index(c.gold[0]) if c.gold else len(gold))


def _sentences(d: Disagreement) -> set[tuple[Lang, int]]:
    return set().union(*(g.sentences() for g in d.gold + d.run))
