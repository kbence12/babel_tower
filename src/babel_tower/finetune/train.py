"""The fine-tuning loop, written out by hand.

One training step, for a batch of (source, target) pairs:

  1. forward   The model reads the source and, for every target position, predicts a
               probability for each of the ~62k vocabulary tokens, given the correct tokens
               before it (teacher forcing). The loss is the cross-entropy: the average of
               -log(probability given to the correct token), over all real target tokens.
  2. backward  loss.backward() computes, for every weight, the gradient: which direction
               (and how strongly) changing that weight would change the loss.
  3. clip      If the gradients are unusually large (an odd batch), scale them down to a
               maximum overall size, so one batch cannot throw the model far off.
  4. step      The optimiser (AdamW) moves every trainable weight a small step against its
               gradient. How big that step is is the learning rate.
  5. schedule  The learning rate warms up from 0 over the first steps (early gradients from a
               model meeting new data are noisy), then decays linearly to 0 by the end.

Fine-tuning needs a much lower learning rate than training from scratch (here 2e-5 against
about 5e-4): the model already translates well, and big steps would overwrite what it knows
with whatever our 1k pairs happen to say. That forgetting is what the flores-dev check after
each epoch watches for, while the dev set (held-out book text) shows whether the book style is
being learned. Dev loss going up while train loss goes down means overfitting: the model is
memorising the training pairs rather than learning from them.

About the model (Marian / OPUS-MT, 77M weights): one embedding table (62,522 tokens x 512
numbers, 32M weights) is shared by the encoder input, the decoder input and the output layer.
So "freeze the encoder" here means freezing the encoder's 6 transformer layers only; freezing
the shared table would also freeze the decoder's input and output.
"""

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from babel_tower import bench, punct
from babel_tower.bench import Direction, languages
from babel_tower.embed import pick_device
from babel_tower.finetune import data

type Freeze = Literal["none", "encoder-layers", "embeddings"]
FREEZE_CHOICES: tuple[Freeze, ...] = ("none", "encoder-layers", "embeddings")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    direction: Direction = "en-hu"
    base_model: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    learning_rate: float = 2e-5
    epochs: int = 5
    batch_size: int = 16
    warmup_fraction: float = 0.1  # share of all steps spent warming the learning rate up
    weight_decay: float = 0.01  # pulls weights slightly towards 0 each step; a mild regulariser
    max_grad_norm: float = 1.0
    max_length: int = 256
    freeze: Freeze = "none"
    flores_sentences: int = 200  # flores-dev sentences translated after each epoch (0 = skip)
    seed: int = 0

    @property
    def model_name(self) -> str:
        src, tgt = languages(self.direction)
        return self.base_model.format(src=src, tgt=tgt)


def train(
    corpus_dir: Path, out_dir: Path, config: TrainConfig, log: Callable[[str], None] = print
) -> list[dict[str, Any]]:
    """Fine-tune, evaluating before the first epoch and after each one. Returns the history."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(config.seed)
    torch.manual_seed(config.seed)  # same seed, same data order and dropout: reruns are comparable
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(device)
    _freeze(model, config.freeze)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"{config.model_name} on {device}: {trainable / 1e6:.1f}M of {total / 1e6:.1f}M weights trainable")

    splits = {
        split: data.tokenize(
            tokenizer, *data.source_and_target(data.load_pairs(corpus_dir, split), config.direction), config.max_length
        )
        for split in ("train", "dev")
    }
    dev_texts = data.source_and_target(data.load_pairs(corpus_dir, "dev"), config.direction)
    flores = bench.load("flores-dev") if config.flores_sentences else None

    steps_per_epoch = math.ceil(len(splits["train"]) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(_parameter_groups(model, config.weight_decay), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _warmup_then_linear_decay(round(config.warmup_fraction * total_steps), total_steps)
    )
    log(f"{len(splits['train'])} train / {len(splits['dev'])} dev pairs, {steps_per_epoch} steps per epoch")

    history: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    for epoch in range(config.epochs + 1):
        t = time.perf_counter()
        train_loss = None
        if epoch > 0:  # epoch 0 only evaluates: the starting point before any training
            model.train()  # switches dropout on
            losses = []
            for step, batch in enumerate(data.batches(splits["train"], config.batch_size, tokenizer.pad_token_id, rng)):
                loss = model(**{k: v.to(device) for k, v in batch.items()}).loss  # 1. forward
                loss.backward()  # 2. backward
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)  # 3. clip
                optimizer.step()  # 4. step
                scheduler.step()  # 5. schedule
                optimizer.zero_grad()  # gradients add up by default, so reset them for the next batch
                losses.append(loss.item())
                if step % 20 == 0:
                    log(f"  epoch {epoch} step {step:3d}/{steps_per_epoch}  loss {loss.item():.3f}  "
                        f"lr {scheduler.get_last_lr()[0]:.2e}")  # fmt: skip
            train_loss = sum(losses) / len(losses)

        scores = _evaluate(model, tokenizer, dev=splits["dev"], dev_texts=dev_texts, flores=flores, config=config)
        record = {"epoch": epoch, "train_loss": train_loss, **scores}
        record["seconds"] = round(time.perf_counter() - t, 1)
        history.append(record)
        log(_epoch_line(record))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=1) + "\n")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    log(f"saved to {out_dir}/ (try: babel-tower bench {out_dir} --direction {config.direction} --fix-punctuation)")
    return history


def _evaluate(
    model: Any,
    tokenizer: Any,
    *,
    dev: list[data.Example],
    dev_texts: tuple[list[str], list[str]],
    flores: bench.Benchmark | None,
    config: TrainConfig,
) -> dict[str, Any]:
    import torch

    device = str(model.device)
    model.eval()  # dropout off: evaluation must be deterministic
    # Dev loss, averaged over target tokens (not over batches, which hold different numbers of tokens).
    loss_sum, tokens = 0.0, 0
    with torch.inference_mode():
        for batch in data.batches(dev, config.batch_size, tokenizer.pad_token_id):
            n = int((batch["labels"] != data.IGNORE_INDEX).sum())
            loss_sum += model(**{k: v.to(device) for k, v in batch.items()}).loss.item() * n
            tokens += n
    dev_loss = loss_sum / tokens
    out: dict[str, Any] = {
        "dev_loss": round(dev_loss, 4),
        # perplexity: as if the model were choosing uniformly between this many tokens at each step
        "dev_perplexity": round(math.exp(dev_loss), 2),
    }

    translator = bench.Translator.from_loaded(config.model_name, tokenizer, model, device)
    _, tgt = languages(config.direction)
    out["dev_chrf++"] = _chrf(translator, *dev_texts, hungarian=tgt == "hu")
    if flores is not None:
        src, _ = languages(config.direction)
        k = config.flores_sentences
        out["flores_dev_chrf++"] = _chrf(translator, flores.side(src)[:k], flores.side(tgt)[:k], hungarian=tgt == "hu")
    return out


def _chrf(translator: bench.Translator, sources: list[str], references: list[str], hungarian: bool) -> float:
    hypotheses = translator.translate([punct.to_model(s) for s in sources])
    if hungarian:
        hypotheses = [punct.hungarian_output(h) for h in hypotheses]
    return bench.Scores.of(hypotheses, references).chrf_pp


def _freeze(model: Any, what: Freeze) -> None:
    """Stop some weights from training (requires_grad = False): no gradient, no update."""
    if what == "encoder-layers":
        for p in model.model.encoder.layers.parameters():
            p.requires_grad = False
    elif what == "embeddings":  # the shared token table, which is also the output layer
        model.model.shared.weight.requires_grad = False


def _parameter_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay for weight matrices only. Biases and LayerNorm scales set offsets and
    scales rather than store knowledge, and pulling them towards 0 only hurts."""
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 or "layer_norm" in name else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _warmup_then_linear_decay(warmup: int, total: int) -> Callable[[int], float]:
    """Learning-rate multiplier per step: 0 -> 1 over `warmup` steps, then 1 -> 0 at `total`."""

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    return factor


def _epoch_line(r: dict[str, Any]) -> str:
    train = f"train loss {r['train_loss']:.3f}" if r["train_loss"] is not None else "before training "
    flores = f"  flores-dev chrF++ {r['flores_dev_chrf++']:.1f}" if "flores_dev_chrf++" in r else ""
    return (
        f"epoch {r['epoch']}: {train}  dev loss {r['dev_loss']:.3f} (ppl {r['dev_perplexity']:.1f})  "
        f"dev chrF++ {r['dev_chrf++']:.1f}{flores}  [{r['seconds']}s]"
    )
