"""babel-tower: run the whole EPUB -> aligned pairs pipeline in one go.

    uv run babel-tower run en.epub hu.epub data/runs/book/ --skip-cost 0.6

Writes every intermediate file to the output directory, plus run.json recording the inputs
(with checksums), settings, code version, library versions, device and timings, so a result
can be traced back to exactly how it was made.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from babel_tower import align, extract_epub, splitter

PIPELINE_FILES = {
    "paragraphs": "{lang}.paragraphs.jsonl",
    "sentences": "{lang}.sentences.jsonl",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "."], cwd=root.parent.parent, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def env_info(device):
    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    info = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: ver(p) for p in ("babel-tower", "torch", "sentence-transformers",
                                             "transformers", "numpy", "ebooklib")}}
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        if device == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def cmd_run(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    sentences = {}
    for lang, path in (("en", a.en), ("hu", a.hu)):
        print(f"=== {lang}: extract {path}")
        t = time.perf_counter()
        paras = extract_epub.extract_paragraphs(path)
        align.write_jsonl(out / PIPELINE_FILES["paragraphs"].format(lang=lang), paras)
        extract_epub.report(paras)
        timings[f"extract_{lang}_s"] = round(time.perf_counter() - t, 3)

        print(f"=== {lang}: split")
        t = time.perf_counter()
        sents = splitter.split_rows(paras, lang)
        align.write_jsonl(out / PIPELINE_FILES["sentences"].format(lang=lang), sents)
        if a.verbose:
            splitter.report(sents)
        else:
            print(f"segments: {len(sents)}")
        timings[f"split_{lang}_s"] = round(time.perf_counter() - t, 3)
        sentences[lang] = sents

    print("=== align")
    embedder = align.Embedder(a.model, a.device, a.batch_size)
    pairs, unaligned, N, M, t_align = align.align(
        sentences["en"], sentences["hu"], a.skip_cost, a.group_penalty, a.keep_verse, a.cache, embedder)
    timings.update(t_align)
    pairs_path = out / "pairs.jsonl"
    align.write_jsonl(pairs_path, pairs)
    align.write_jsonl(align.unaligned_path(pairs_path), unaligned)

    run = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "inputs": {lang: {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                   for lang, p in (("en", a.en), ("hu", a.hu))},
        "settings": {"skip_cost": a.skip_cost, "group_penalty": a.group_penalty,
                     "keep_verse": a.keep_verse, "model": a.model, "max_group": align.K},
        "code": git_info(),
        "env": env_info(embedder.device),
        "embedding": {k: round(v, 3) if isinstance(v, float) else v for k, v in embedder.stats.items()},
        "timings": timings,
        "stats": align.stats(pairs, unaligned, N, M),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    align.summarize(pairs, unaligned, N, M)
    e = run["embedding"]
    print(f"=== wrote {out}/  (device {e['device']}, embedding cache hits {e['cache_hits']}/"
          f"{e['cache_hits'] + e['cache_misses']}, encode {e['encode_s']}s, dp {timings['dp_s']}s)")


def main():
    ap = argparse.ArgumentParser(prog="babel-tower", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="EPUBs -> paragraphs -> sentences -> aligned pairs")
    r.add_argument("en", type=Path, help="English EPUB")
    r.add_argument("hu", type=Path, help="Hungarian EPUB")
    r.add_argument("out", type=Path, help="output directory")
    r.add_argument("-v", "--verbose", action="store_true", help="print the full splitter report")
    align.add_args(r)
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
