"""Translation benchmarks: load a test set, translate it with a model, score the output.

Benchmarks never contain book text, so they can test models trained on the books:
  literary        200 fiction-style sentences written for this project (benchmarks/literary.txt)
  flores-dev      FLORES+ dev, 997 general-domain sentences (gated on Hugging Face: accept its terms)
  flores-devtest  FLORES+ devtest, 1012 sentences; the standard test split

Scores are chrF++ (character and word n-gram F-score, the main metric) and BLEU, via sacrebleu,
with sacrebleu's signature recorded so that numbers stay comparable.
"""

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from babel_tower.embed import pick_device
from babel_tower.provenance import REPO_ROOT
from babel_tower.records import Lang

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

type Direction = Literal["en-hu", "hu-en"]
DIRECTIONS: tuple[Direction, Direction] = ("en-hu", "hu-en")
BENCHMARKS = ("literary", "flores-dev", "flores-devtest")
LITERARY = REPO_ROOT / "benchmarks" / "literary.txt"
FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_CODES: dict[Lang, str] = {"en": "eng_Latn", "hu": "hun_Latn"}


class BenchmarkUnavailableError(RuntimeError):
    pass


def languages(direction: Direction) -> tuple[Lang, Lang]:
    return ("en", "hu") if direction == "en-hu" else ("hu", "en")


@dataclass(frozen=True, slots=True)
class Benchmark:
    name: str
    en: list[str]
    hu: list[str]
    sections: list[str] = field(default_factory=list)  # per sentence, if the set has sections

    def __post_init__(self) -> None:
        if len(self.en) != len(self.hu):
            raise ValueError(f"{self.name}: {len(self.en)} EN but {len(self.hu)} HU sentences")

    def side(self, lang: Lang) -> list[str]:
        return self.en if lang == "en" else self.hu


def load(name: str) -> Benchmark:
    if name == "literary":
        return load_literary(LITERARY)
    if name in ("flores-dev", "flores-devtest"):
        return _load_flores(name.removeprefix("flores-"))
    raise ValueError(f"unknown benchmark {name!r}; known: {', '.join(BENCHMARKS)}")


def load_literary(path: Path) -> Benchmark:
    en, hu, sections = [], [], []
    section = ""
    expect: Lang = "en"
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"(EN|HU): (.+)", line)
        lang: Lang | None = None if m is None else ("en" if m.group(1) == "EN" else "hu")
        if lang != expect:
            raise ValueError(f"{path}:{n}: expected an {expect.upper()}: line, got {raw!r}")
        assert m is not None
        if lang == "en":
            en.append(m.group(2))
            sections.append(section)
        else:
            hu.append(m.group(2))
        expect = "hu" if expect == "en" else "en"
    if expect != "en":
        raise ValueError(f"{path}: the last EN line has no HU line")
    return Benchmark("literary", en, hu, sections)


def _load_flores(split: str) -> Benchmark:
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    sides: dict[Lang, list[str]] = {}
    for lang, code in FLORES_CODES.items():
        try:
            path = hf_hub_download(FLORES_REPO, f"{split}/{code}.jsonl", repo_type="dataset")
        except GatedRepoError as e:
            raise BenchmarkUnavailableError(
                f"FLORES+ is gated: accept its terms at https://huggingface.co/datasets/{FLORES_REPO} "
                "while logged in to the account that `hf auth whoami` shows"
            ) from e
        with Path(path).open(encoding="utf-8") as f:
            rows = sorted((json.loads(line) for line in f), key=lambda r: r["id"])
        sides[lang] = [r["text"] for r in rows]
    return Benchmark(f"flores-{split}", sides["en"], sides["hu"])


class Translator:
    """A Hugging Face sequence-to-sequence model (e.g. Marian / OPUS-MT), loaded on first use."""

    def __init__(
        self, model: str, device: str | None = None, batch_size: int = 32, num_beams: int = 4, max_new_tokens: int = 256
    ) -> None:
        self.model_name = model
        self.device = device or pick_device()
        self.batch_size, self.num_beams, self.max_new_tokens = batch_size, num_beams, max_new_tokens
        self.load_s = self.translate_s = 0.0
        # the model is Any: transformers' annotations make mypy reject model.generate(...)
        self._loaded: tuple[PreTrainedTokenizerBase, Any] | None = None

    def settings(self) -> dict[str, Any]:
        return {"model": self.model_name, "device": self.device, "batch_size": self.batch_size,
                "num_beams": self.num_beams, "max_new_tokens": self.max_new_tokens}  # fmt: skip

    def translate(self, texts: Sequence[str]) -> list[str]:
        import torch

        tokenizer, model = self._load()
        t = time.perf_counter()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths batch well
        out: list[str] = [""] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True)
            with torch.inference_mode():
                generated = model.generate(
                    **batch.to(self.device),
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    max_length=None,  # the model config's max_length would conflict with max_new_tokens
                )
            for i, text in zip(idx, tokenizer.batch_decode(generated, skip_special_tokens=True), strict=True):
                out[i] = text
        self.translate_s += time.perf_counter() - t
        return out

    def _load(self) -> tuple[PreTrainedTokenizerBase, Any]:
        if self._loaded is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            t = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()
            self._loaded = tokenizer, model
            self.load_s += time.perf_counter() - t
        return self._loaded


BOOTSTRAP_SAMPLES = 1000


@dataclass(frozen=True, slots=True)
class Scores:
    chrf_pp: float
    bleu: float
    signatures: dict[str, str]
    chrf_pp_ci: float | None = None  # half-width of the 95% bootstrap confidence interval

    @classmethod
    def of(cls, hypotheses: Sequence[str], references: Sequence[str], bootstrap: bool = False) -> Scores:
        chrf, bleu = CHRF(word_order=2), BLEU()
        refs = [list(references)]
        chrf_pp = chrf.corpus_score(hypotheses, refs).score
        ci = None
        if bootstrap and len(hypotheses) > 1:
            # resample the test set with replacement; how much the score moves says how far to trust it
            ci = round(float(chrf.corpus_score(hypotheses, refs, n_bootstrap=BOOTSTRAP_SAMPLES)._ci), 2)
        return cls(
            chrf_pp=round(chrf_pp, 2),
            bleu=round(bleu.corpus_score(hypotheses, refs).score, 2),
            signatures={"chrf++": str(chrf.get_signature()), "bleu": str(bleu.get_signature())},
            chrf_pp_ci=ci,
        )


def evaluate(benchmark: Benchmark, direction: Direction, translator: Translator) -> dict[str, Any]:
    """Translate a benchmark in one direction and score it; returns the full result record."""
    src_lang, tgt_lang = languages(direction)
    sources, references = benchmark.side(src_lang), benchmark.side(tgt_lang)
    hypotheses = translator.translate(sources)
    result: dict[str, Any] = {
        "benchmark": benchmark.name,
        "direction": direction,
        "sentences": len(sources),
        "scores": _scores_dict(Scores.of(hypotheses, references, bootstrap=True)),
        "translator": translator.settings(),
    }
    if benchmark.sections:
        result["by_section"] = {
            section: Scores.of(
                [h for h, s in zip(hypotheses, benchmark.sections, strict=True) if s == section],
                [r for r, s in zip(references, benchmark.sections, strict=True) if s == section],
            ).chrf_pp
            for section in dict.fromkeys(benchmark.sections)
        }
    result["translations"] = [
        {"source": s, "reference": r, "hypothesis": h} for s, r, h in zip(sources, references, hypotheses, strict=True)
    ]
    return result


def _scores_dict(s: Scores) -> dict[str, Any]:
    return {"chrf++": s.chrf_pp, "chrf++_ci95": s.chrf_pp_ci, "bleu": s.bleu, "signatures": s.signatures}
