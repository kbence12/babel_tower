from pathlib import Path

import pytest

from babel_tower.bench import LITERARY, Scores, load_literary


def test_literary_benchmark_is_well_formed() -> None:
    b = load_literary(LITERARY)
    assert len(b.en) == len(b.hu) == len(b.sections) == 200
    assert len(set(b.en)) == len(b.en), "duplicate English sentences"
    assert all(b.sections)


def test_literary_parser_rejects_unpaired_lines(tmp_path: Path) -> None:
    path = tmp_path / "set.txt"
    path.write_text("## s\nEN: One.\nEN: Two.\nHU: Kettő.\n")
    with pytest.raises(ValueError, match="expected an HU: line"):
        load_literary(path)
    path.write_text("EN: One.\n")
    with pytest.raises(ValueError, match="no HU line"):
        load_literary(path)


def test_scores_reward_near_misses_more_under_chrf() -> None:
    perfect = Scores.of(["A kutya a házban alszik."], ["A kutya a házban alszik."])
    near = Scores.of(["A kutya a házakban alszik."], ["A kutya a házban alszik."])
    assert perfect.chrf_pp == perfect.bleu == 100.0
    assert near.chrf_pp > 75 > near.bleu
    assert "nw:2" in near.signatures["chrf++"]
