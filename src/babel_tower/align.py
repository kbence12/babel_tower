#!/usr/bin/env python3
"""Align two sentence files (English, Hungarian) with LaBSE embeddings and a small DP aligner.

Usage:
    pip install sentence-transformers numpy
    python align.py data/interim/pooh_en.sentences.jsonl data/interim/pooh_hu.sentences.jsonl \
        data/aligned/pooh.pairs.jsonl

How it works:
  1. Drop verse blocks and segments with no letters/digits (--keep-verse keeps verse).
  2. Embed every window of 1..3 consecutive segments on each side with LaBSE. Quote marks and
     dialogue dashes are removed from what the embedder sees, because they differ by language.
     Embeddings are cached on disk, so reruns with different settings are fast.
  3. Dynamic programming over (English position, Hungarian position) finds the cheapest
     monotonic alignment. Moves: 1-1, 1-2, 2-1, 2-2, 1-3, 3-1, or skip one segment on either side.
        cost(aligned group) = (1 - cosine) * (n + m) + --group-penalty * (n + m - 2)
        cost(skip)          = --skip-cost per skipped segment
     So --skip-cost 0.5 means roughly "pairs with cosine below ~0.5 are better left unaligned".
     The group penalty exists because bigger windows score a higher cosine, which would otherwise
     merge consecutive 1-1 pairs into 2-2 groups; it nudges the aligner towards the finest valid split.
  4. Writes the aligned pairs, the unaligned segments, and prints a summary to check by eye.

The DP is exact over the full (N x M) grid, computed row by row with numpy. That is fine for a
single book or a few hundred; for much larger inputs restrict it to a band around the diagonal.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

MOVES = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
K = 3  # largest group size on either side
SKIP_EN, SKIP_HU = 100, 101
DROP_CHARS = re.compile(r"[“”„«»\"–—]")


def embed_view(text):
    """What the embedder sees: quote marks and dialogue dashes replaced by spaces."""
    return re.sub(r"\s+", " ", DROP_CHARS.sub(" ", text)).strip()


def load_segments(path, keep_verse):
    segs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["verse"] and not keep_verse:
                continue
            o["view"] = embed_view(o["text"])
            if not any(ch.isalnum() for ch in o["view"]):
                continue
            segs.append(o)
    return segs


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_windows(views, tag, cache_dir, model_name):
    """Embed every window of 1..K consecutive segments. Returns {n: array[N-n+1, dim]}."""
    out, model = {}, None
    for n in range(1, K + 1):
        texts = [" ".join(views[i : i + n]) for i in range(len(views) - n + 1)]
        if not texts:
            out[n] = np.zeros((0, 1), dtype=np.float32)
            continue
        key = hashlib.sha1(("\n".join(texts) + model_name).encode("utf-8")).hexdigest()[:12]
        path = cache_dir / f"{tag}_n{n}_{key}.npy"
        if path.exists():
            out[n] = np.load(path)
            continue
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=pick_device())
        emb = model.encode(
            texts, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        out[n] = emb
    return out


def dp_align(Een, Ehu, N, M, c, group_penalty=0.0):
    """Cheapest monotonic alignment.

    Een[n][i] / Ehu[m][j]: unit-norm embedding of the window of n (m) segments starting at i (j).
    Returns a list of steps (en_start, n, hu_start, m); n == 0 or m == 0 means a skipped segment.
    """
    D = np.empty((N + 1, M + 1), dtype=np.float64)
    back = np.empty((N + 1, M + 1), dtype=np.int8)
    j = np.arange(M + 1, dtype=np.float64)
    D[0] = j * c
    back[0] = SKIP_HU

    for i in range(1, N + 1):
        base = D[i - 1] + c  # skip English segment i-1
        mv = np.full(M + 1, SKIP_EN, dtype=np.int8)
        for k, (n, m) in enumerate(MOVES):
            if i < n or M < m:
                continue
            sims = Ehu[m] @ Een[n][i - n]  # one cosine per possible Hungarian window start
            cand = np.full(M + 1, np.inf)
            cand[m:] = D[i - n][: M + 1 - m] + (1.0 - sims) * (n + m) + group_penalty * (n + m - 2)
            better = cand < base
            base = np.where(better, cand, base)
            mv = np.where(better, k, mv).astype(np.int8)
        # skipping Hungarian segments is a running min along the row: D[i][j] = min_k base[k] + (j-k)*c
        scan = np.minimum.accumulate(base - j * c) + j * c
        mv = np.where(scan < base - 1e-9, SKIP_HU, mv).astype(np.int8)
        D[i] = scan
        back[i] = mv

    steps, i, jj = [], N, M
    while i > 0 or jj > 0:
        mv = int(back[i, jj])
        if mv == SKIP_EN:
            steps.append((i - 1, 1, jj, 0))
            i -= 1
        elif mv == SKIP_HU:
            steps.append((i, 0, jj - 1, 1))
            jj -= 1
        else:
            n, m = MOVES[mv]
            steps.append((i - n, n, jj - m, m))
            i -= n
            jj -= m
    steps.reverse()
    return steps


def clip(s, w=88):
    return s if len(s) <= w else s[: w - 1] + "…"


def show(p):
    print(f"  [{p['type']} sim={p['sim']:.2f}]")
    print(f"     EN {clip(p['en'])}")
    print(f"     HU {clip(p['hu'])}")


def summarize(pairs, unaligned, N, M):
    types = Counter(p["type"] for p in pairs)
    sims = np.array([p["sim"] for p in pairs])
    en_cov = sum(len(p["en_ids"]) for p in pairs) / max(N, 1)
    hu_cov = sum(len(p["hu_ids"]) for p in pairs) / max(M, 1)
    print(f"segments aligned: EN {N}  HU {M}   pairs: {len(pairs)}   types: {dict(sorted(types.items()))}")
    print(f"coverage: EN {en_cov:.1%}  HU {hu_cov:.1%}   unaligned: "
          f"EN {sum(u['side'] == 'en' for u in unaligned)}  HU {sum(u['side'] == 'hu' for u in unaligned)}")
    if len(sims):
        q = np.percentile(sims, [10, 25, 50, 75])
        print(f"cosine: p10={q[0]:.2f} p25={q[1]:.2f} median={q[2]:.2f} p75={q[3]:.2f} mean={sims.mean():.2f}   "
              f"below 0.6: {(sims < 0.6).sum()}   below 0.7: {(sims < 0.7).sum()}")

    random.seed(0)
    order = sorted(range(len(pairs)), key=lambda k: pairs[k]["sim"])
    print("--- first 3 pairs ---")
    for p in pairs[:3]:
        show(p)
    print("--- last 3 pairs ---")
    for p in pairs[-3:]:
        show(p)
    print("--- 6 lowest-similarity pairs ---")
    for k in order[:6]:
        show(pairs[k])
    mid = order[int(len(order) * 0.3) : int(len(order) * 0.7)]
    print("--- 6 random mid-range pairs ---")
    for k in random.sample(mid, min(6, len(mid))):
        show(pairs[k])
    print("--- 4 random unaligned segments per side ---")
    for side in ("en", "hu"):
        us = [u for u in unaligned if u["side"] == side]
        for u in random.sample(us, min(4, len(us))):
            print(f"  [{side}] {clip(u['text'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("en", type=Path, help="English sentences JSONL")
    ap.add_argument("hu", type=Path, help="Hungarian sentences JSONL")
    ap.add_argument("out", type=Path, help="output pairs JSONL")
    ap.add_argument("--skip-cost", type=float, default=0.5)
    ap.add_argument("--group-penalty", type=float, default=0.1,
                    help="extra cost per segment beyond a 1-1 pair; raise if 2-2 / 1-3 groups dominate")
    ap.add_argument("--keep-verse", action="store_true")
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = ap.parse_args()

    en = load_segments(args.en, args.keep_verse)
    hu = load_segments(args.hu, args.keep_verse)
    N, M = len(en), len(hu)
    print(f"loaded EN {N} / HU {M} segments")

    Een = embed_windows([s["view"] for s in en], "en", args.cache, args.model)
    Ehu = embed_windows([s["view"] for s in hu], "hu", args.cache, args.model)
    steps = dp_align(Een, Ehu, N, M, args.skip_cost, args.group_penalty)

    pairs, unaligned = [], []
    for a, n, b, m in steps:
        if n and m:
            pairs.append({
                "pair": len(pairs),
                "type": f"{n}-{m}",
                "sim": round(float(Een[n][a] @ Ehu[m][b]), 4),
                "en_ids": [en[a + t]["id"] for t in range(n)],
                "hu_ids": [hu[b + t]["id"] for t in range(m)],
                "en": " ".join(en[a + t]["text"] for t in range(n)),
                "hu": " ".join(hu[b + t]["text"] for t in range(m)),
            })
        else:
            for t in range(n):
                unaligned.append({"side": "en", "id": en[a + t]["id"], "text": en[a + t]["text"]})
            for t in range(m):
                unaligned.append({"side": "hu", "id": hu[b + t]["id"], "text": hu[b + t]["text"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    un_path = args.out.with_suffix(".unaligned.jsonl")
    with un_path.open("w", encoding="utf-8") as f:
        for u in unaligned:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    summarize(pairs, unaligned, N, M)


if __name__ == "__main__":
    main()