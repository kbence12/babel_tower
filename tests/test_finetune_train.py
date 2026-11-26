from babel_tower.finetune.train import _warmup_then_linear_decay


def test_learning_rate_warms_up_then_decays_to_zero() -> None:
    factor = _warmup_then_linear_decay(warmup=10, total=110)
    assert factor(0) == 0.1
    assert factor(9) == 1.0
    assert factor(60) == 0.5
    assert factor(110) == 0.0
