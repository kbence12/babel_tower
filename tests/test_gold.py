from pathlib import Path

import pytest

from babel_tower import jsonl
from babel_tower.gold import (
    GoldFormatError,
    Group,
    Sides,
    check_texts,
    evaluate,
    missing_lines,
    parse,
    path_groups,
    pick_windows,
    write_template,
)
from babel_tower.records import Pair, Unaligned

SIDES = Sides(en={i: f"en {i}." for i in range(10)}, hu={i: f"hu {i}." for i in range(10)})


def pair(k: int, en: list[int], hu: list[int]) -> Pair:
    return Pair(k, f"{len(en)}-{len(hu)}", 0.8, en, hu, "", "")


def test_path_groups_puts_skips_between_the_pairs_around_them() -> None:
    pairs = [pair(0, [0], [0]), pair(1, [2], [3, 4]), pair(2, [3], [5])]
    unaligned = [Unaligned("hu", 1, ""), Unaligned("en", 1, ""), Unaligned("hu", 2, ""), Unaligned("en", 4, "")]
    assert path_groups(pairs, unaligned) == [
        Group((0,), (0,)), Group((1,), ()), Group((), (1,)), Group((), (2,)),
        Group((2,), (3, 4)), Group((3,), (5,)), Group((4,), ()),
    ]  # fmt: skip


def test_pick_windows_spreads_over_the_book() -> None:
    groups = [Group((i,), (i,)) for i in range(100)]
    windows = pick_windows(groups, n_windows=4, size=5, seed=0)
    assert [len(w) for w in windows] == [5, 5, 5, 5]
    assert [w[0].en[0] // 25 for w in windows] == [0, 1, 2, 3]


def test_template_round_trips(tmp_path: Path) -> None:
    windows = [[Group((0,), (0, 1)), Group((1,), ())], [Group((), (5,)), Group((5, 6), (6,))]]
    path = tmp_path / "gold.txt"
    write_template(path, windows, SIDES)
    parsed = parse(path)
    assert [w.groups for w in parsed] == windows
    assert check_texts(parsed, SIDES) == []
    assert check_texts(parsed, Sides(en={**SIDES.en, 1: "changed"}, hu=SIDES.hu)) != []


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("EN 1 | x\n", "before any group"),
        ("=== window 1\nEN 1 | x\nEN 1 | x\n", "more than once"),
        ("=== window 1\nHU 2 | x\n---\nHU 1 | x\n", "out of order"),
        ("=== window 1\nthis is not a line\n", "cannot read"),
    ],
)
def test_parse_rejects_broken_files(tmp_path: Path, body: str, error: str) -> None:
    path = tmp_path / "gold.txt"
    path.write_text(body)
    with pytest.raises(GoldFormatError, match=error):
        parse(path)


def test_evaluate(tmp_path: Path) -> None:
    path = tmp_path / "gold.txt"
    path.write_text(
        "# comment\n=== window 1\n"
        "EN 0 | a\nHU 0 | a\n---\nEN 1 | b\nHU 1 | b\nHU 2 | c\n---\nEN 2 | d\n---\nEN 3 | e\nHU 3 | e\n"
    )
    # right: 0-0 and 3-3. Wrong: 1 was paired with 1 only, 2 was skipped instead of with 1,
    # and EN 2 was paired with HU 2 instead of being left out.
    predicted = [Group((0,), (0,)), Group((1,), (1,)), Group((2,), (2,)), Group((3,), (3,)), Group((9,), (9,))]
    ev = evaluate(parse(path), predicted)
    assert (ev.links.correct, ev.links.predicted, ev.links.gold) == (3, 4, 4)
    assert (ev.groups.correct, ev.groups.predicted, ev.groups.gold) == (2, 4, 3)
    assert (ev.skipped.correct, ev.skipped.predicted, ev.skipped.gold) == (0, 0, 1)
    assert ev.skipped.precision == 1.0
    assert [(d.gold, d.run) for d in ev.disagreements] == [
        ([Group((1,), (1, 2)), Group((2,), ())], [Group((1,), (1,)), Group((2,), (2,))]),
    ]


def test_missing_lines_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "gold.txt"
    path.write_text("=== window 1\nEN 0 | a\nHU 0 | a\n---\nEN 2 | c\nHU 2 | c\n")
    predicted = [Group((0,), (0,)), Group((1,), (1,)), Group((2,), (2,))]
    assert missing_lines(parse(path), predicted) == [(1, "en", 1), (1, "hu", 1)]


def test_multi_line_skip_group_means_each_line_has_no_counterpart(tmp_path: Path) -> None:
    path = tmp_path / "gold.txt"
    path.write_text("=== window 1\nEN 0 | a\nHU 0 | a\n---\nHU 1 | b\nHU 2 | c\n")
    ev = evaluate(parse(path), [Group((0,), (0,)), Group((), (1,)), Group((), (2,))])
    assert ev.disagreements == []
    assert ev.skipped.f1 == 1.0


def test_jsonl_round_trip(tmp_path: Path) -> None:
    rows = [pair(0, [1, 2], [3]), pair(1, [4], [5])]
    jsonl.write(tmp_path / "pairs.jsonl", rows)
    assert jsonl.read(tmp_path / "pairs.jsonl", Pair) == rows


def test_line_deleted_at_the_window_edge_is_found_through_the_header_extent(tmp_path: Path) -> None:
    path = tmp_path / "gold.txt"
    write_template(path, [[Group((0,), (0,)), Group((1,), (1,))]], SIDES)
    assert path.read_text().count("=== window 1 | EN 0-1 | HU 0-1") == 1
    path.write_text(path.read_text().replace("HU 1 | hu 1.\n", ""))
    assert missing_lines(parse(path), [Group((0,), (0,)), Group((1,), (1,))]) == [(1, "hu", 1)]
