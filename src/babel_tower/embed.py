"""Sentence embeddings for every window of consecutive sentences, cached on disk."""

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

type Vectors = npt.NDArray[np.float32]  # one unit-norm row per text
type Windows = dict[int, Vectors]  # window size n -> one row per window start

DEFAULT_MODEL = "sentence-transformers/LaBSE"


@dataclass
class EmbedStats:
    device: str
    batch_size: int
    load_s: float = 0.0
    encode_s: float = 0.0
    texts_encoded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gpu_peak_mem_mb: int | None = None


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Wraps a sentence-transformers model. The model is loaded on the first encode, so a run
    whose embeddings are all cached never loads it."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or pick_device()
        self.stats = EmbedStats(device=self.device, batch_size=batch_size)
        self._model: SentenceTransformer | None = None

    def encode(self, texts: list[str]) -> Vectors:
        model = self._load()
        t = time.perf_counter()
        emb = model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        ).astype(np.float32)
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()
            self.stats.gpu_peak_mem_mb = round(torch.cuda.max_memory_allocated() / 2**20)
        self.stats.encode_s += time.perf_counter() - t
        self.stats.texts_encoded += len(texts)
        return emb

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            t = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.stats.load_s += time.perf_counter() - t
        return self._model


def embed_windows(texts: Sequence[str], max_size: int, tag: str, cache_dir: Path, embedder: Embedder) -> Windows:
    """Embed every window of 1..max_size consecutive texts, joined with spaces.

    Each window size is cached as one .npy file, keyed on the exact texts and the model name.
    """
    out: Windows = {}
    for n in range(1, max_size + 1):
        windows = [" ".join(texts[i : i + n]) for i in range(len(texts) - n + 1)]
        if not windows:
            out[n] = np.zeros((0, 1), dtype=np.float32)
            continue
        path = cache_dir / f"{tag}_n{n}_{_cache_key(windows, embedder.model_name)}.npy"
        if path.exists():
            embedder.stats.cache_hits += 1
            out[n] = np.load(path)
            continue
        embedder.stats.cache_misses += 1
        out[n] = embedder.encode(windows)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, out[n])
    return out


def _cache_key(windows: list[str], model_name: str) -> str:
    data = ("\n".join(windows) + model_name).encode("utf-8")
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()[:12]
