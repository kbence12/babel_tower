from babel_tower.bench import Benchmark
from babel_tower.corpus import CorpusConfig, Source, _dev_positions, build, normalize
from babel_tower.records import Pair, Sentence


def sentences(texts: list[str]) -> list[Sentence]:
    return [Sentence(id=i, para=i, pos=0, doc=0, tag="p", verse=False, text=t) for i, t in enumerate(texts)]


def source(rows: list[tuple[str, str, float]]) -> Source:
    pairs = [Pair(k, "1-1", sim, [k], [k], en, hu) for k, (en, hu, sim) in enumerate(rows)]
    return Source.of("book", pairs, sentences([r[0] for r in rows]), sentences([r[1] for r in rows]))


def test_build_filters_dedupes_and_removes_benchmark_sentences() -> None:
    rows = [
        ("It rained all day.", "Egész nap esett.", 0.8),
        ("It rained all day.", "Egész nap esett.", 0.8),  # duplicate
        ("Hello.", "Valami egészen más, nagyon hosszú mondat, ami nem fordítása.", 0.9),  # length ratio
        ("Yes.", "Nem.", 0.3),  # low similarity
        ("Not yet!", "Még nem!", 0.9),  # in the benchmark after normalising
        ("...", "!", 0.9),
    ]
    bench = Benchmark("b", ["not yet"], ["Valami."])
    result = build([source(rows)], [bench], CorpusConfig(min_sim=0.5, dev_fraction=0))
    assert [(p.en, p.hu) for p in result.train] == [("It rained all day.", "Egész nap esett.")]
    assert result.dropped == {
        "duplicate": 1, "length ratio above 3.0": 1, "similarity below 0.5": 1,
        "sentence also in a benchmark": 1, "no letters or digits": 1,
    }  # fmt: skip
    assert result.contaminated == [{"source": "book", "pair": 4, "sentences": ["Not yet!"]}]


def test_dev_is_contiguous_blocks_spread_over_the_book() -> None:
    import random

    positions = sorted(_dev_positions(1000, CorpusConfig(dev_fraction=0.1, dev_blocks=4), random.Random(0)))
    assert len(positions) == 100
    blocks = [p for i, p in enumerate(positions) if i == 0 or positions[i - 1] != p - 1]
    assert len(blocks) == 4
    assert [b // 250 for b in blocks] == [0, 1, 2, 3]


def test_normalize_ignores_case_and_punctuation() -> None:
    assert normalize("– „Igen!” – mondta Ödön.") == normalize("igen mondta ödön") == "igen mondta ödön"
