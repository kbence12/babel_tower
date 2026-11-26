import numpy as np
import pytest

from babel_tower.align import MAX_GROUP, Step, alignable, dp_align, embed_view
from babel_tower.embed import Vectors, Windows
from babel_tower.records import Sentence

DIM = 64


def unit(v: np.ndarray) -> Vectors:
    out: Vectors = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def windows(vectors: list[Vectors]) -> Windows:
    """Window embeddings the way a sentence model roughly behaves: the normalised sum of its parts."""
    return {
        n: np.array([unit(np.sum(vectors[i : i + n], axis=0)) for i in range(len(vectors) - n + 1)]).reshape(-1, DIM)
        for n in range(1, MAX_GROUP + 1)
    }


@pytest.fixture
def concepts() -> list[Vectors]:
    rng = np.random.default_rng(0)
    return [unit(rng.normal(size=DIM)) for _ in range(6)]


def align(en: list[Vectors], hu: list[Vectors]) -> list[Step]:
    return dp_align(windows(en), windows(hu), len(en), len(hu), skip_cost=0.5, group_penalty=0.1)


def test_identical_sides_align_one_to_one(concepts: list[Vectors]) -> None:
    assert align(concepts, concepts) == [Step(i, 1, i, 1) for i in range(len(concepts))]


def test_sentence_split_by_the_translator_becomes_a_1_2_group(concepts: list[Vectors]) -> None:
    a, b, c = concepts[:3]
    orthogonal = unit(concepts[3] - (concepts[3] @ b) * b)
    halves = [unit(b + orthogonal), unit(b - orthogonal)]  # each ~0.7 like b, together exactly b
    assert align([a, b, c], [a, *halves, c]) == [Step(0, 1, 0, 1), Step(1, 1, 1, 2), Step(2, 1, 3, 1)]


def test_unrelated_sentence_is_skipped(concepts: list[Vectors]) -> None:
    a, b, c, extra = concepts[:4]
    assert align([a, b, c], [a, extra, b, c]) == [
        Step(0, 1, 0, 1),
        Step(1, 0, 1, 1),
        Step(1, 1, 2, 1),
        Step(2, 1, 3, 1),
    ]


def test_empty_side_is_all_skips(concepts: list[Vectors]) -> None:
    assert align(concepts[:2], []) == [Step(0, 1, 0, 0), Step(1, 1, 0, 0)]


def test_embed_view_drops_quotes_and_dialogue_dashes() -> None:
    assert embed_view("– „Igen” – mondta.") == "Igen mondta."
    assert embed_view('"Yes," he said — quietly.') == "Yes, he said quietly."


def test_alignable_drops_verse_and_punctuation_only_segments() -> None:
    def sentence(i: int, text: str, verse: bool = False) -> Sentence:
        return Sentence(id=i, para=i, pos=0, doc=0, tag="p", verse=verse, text=text)

    rows = [sentence(0, "Hello."), sentence(1, "* * *"), sentence(2, "A song", verse=True), sentence(3, "– …")]
    assert [s.id for s in alignable(rows, keep_verse=False)] == [0]
    assert [s.id for s in alignable(rows, keep_verse=True)] == [0, 2]
