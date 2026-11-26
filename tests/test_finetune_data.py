import random

from babel_tower.finetune.data import IGNORE_INDEX, Example, batches, collate

PAD = 99


def test_collate_pads_inputs_with_pad_and_labels_with_ignore_index() -> None:
    batch = collate([Example([1, 2, 3], [7]), Example([4], [8, 9])], PAD)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, PAD, PAD]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[7, IGNORE_INDEX], [8, 9]]


def test_batches_group_by_length_and_cover_every_example_once() -> None:
    examples = [Example(list(range(n)), [n]) for n in (5, 1, 4, 2, 3, 6)]
    seen = []
    for batch in batches(examples, batch_size=2, pad_id=PAD, rng=random.Random(0)):
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        assert max(lengths) - min(lengths) <= 1  # neighbours in length share a batch
        seen += batch["labels"][:, 0].tolist()
    assert sorted(seen) == [1, 2, 3, 4, 5, 6]
