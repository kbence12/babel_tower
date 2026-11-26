#!/usr/bin/env python3
"""Split paragraph JSONL (from extract_epub.py) into sentence JSONL.

Usage:
    python split_sentences.py data/interim/pooh_en.paragraphs.jsonl data/interim/pooh_en.sentences.jsonl --lang en
    python split_sentences.py data/interim/pooh_hu.paragraphs.jsonl data/interim/pooh_hu.sentences.jsonl --lang hu

Rule-based and dependency-free, so it is easy to read and debug. A boundary is a
.!? or … (plus any closing quotes) followed by whitespace, unless:
  - the next real character (skipping opening quotes, brackets and dashes) is lowercase.
    This is what keeps Hungarian speech tags together:  – Hogy vagy? – kérdezte Róbert.
  - the punctuation is a single "." after a known abbreviation (Mr., dr., pl., ...)
  - the punctuation is a single "." after an initial (A. A. Milne; roman numeral I.)

Headings and multi-line blocks (verse) are passed through unsplit and flagged.

Known weak spots (look for them in the report):
  - speech tag starting with a capitalised name after ! or ?:  "Oh, bother!" Pooh said.
  - abbreviations missing from ABBREV
  - English pronoun "I" at the end of a sentence is handled, but "I." as a roman numeral
    in English text is not (headings are not split, so this rarely matters)
"""
import argparse
import json
import random
import re
from pathlib import Path

# Abbreviations that (almost) never end a sentence. Extend as the report shows misses.
# Deliberately NOT included: stb, etc  (they often do end sentences).
ABBREV = {
    # English
    "mr", "mrs", "ms", "dr", "st", "mt", "sr", "jr", "prof", "gen", "col", "capt",
    "lt", "rev", "hon", "vs", "viz", "cf", "e.g", "i.e", "fig", "vol", "ch",
    # Hungarian
    "pl", "kb", "ún", "ill", "ifj", "id", "özv", "szül", "megh", "vö", "pld",
    "ált", "hrsz", "krt", "tkp", "ny", "sz", "kft",
}

CANDIDATE = re.compile(r"([.!?…]+)([\"'”’»)\]]*)(\s+)")
OPENERS = set("\"'“‘«„([–—-")


def is_boundary(text, m, lang, start=0):
    punct = m.group(1)
    before = text[: m.start()]
    after = text[m.end():]

    j = 0
    while j < len(after) and (after[j] in OPENERS or after[j].isspace()):
        j += 1
    if j >= len(after):
        return False
    nxt = after[j]
    if not nxt.isalnum() or nxt.islower():
        # lowercase continuation, or more punctuation (e.g. "... ?" or ". . ."): same sentence
        return False

    if punct == "." and re.fullmatch(r"\d{1,2}", text[start : m.start()].strip()):
        return False  # list marker such as "1. General remarks"

    if punct == ".":
        mt = re.search(r"([\w.]+)$", before)
        tok = mt.group(1) if mt else ""
        if tok.lower() in ABBREV:
            return False
        parts = tok.split(".")
        is_initial = bool(tok) and all(len(p) == 1 and p.isupper() for p in parts)
        if is_initial and not (lang == "en" and tok == "I"):
            return False
    return True


def split_sentences(text, lang):
    out, start = [], 0
    for m in CANDIDATE.finditer(text):
        if is_boundary(text, m, lang, start):
            out.append(text[start : m.end(2)].strip())
            start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    # safety net: a fragment with no letters or digits (e.g. '?"' or '...') belongs to the previous one
    merged = []
    for p in out:
        if merged and not any(ch.isalnum() for ch in p):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def report(out):
    n = len(out)
    lens = sorted(len(o["text"].split()) for o in out)
    pct = lambda p: lens[min(n - 1, int(p * n))]
    print(f"segments: {n}   words/segment: p5={pct(.05)} median={pct(.5)} p95={pct(.95)} max={lens[-1]}")
    long_ = sum(1 for l in lens if l > 60)
    short = sum(1 for l in lens if l <= 2)
    plain = [o for o in out if not o["tag"].startswith("h") and not o["verse"]]
    noend = [o for o in plain if not re.search(r"[.!?…\"”’»)]$", o["text"])]
    verse = sum(1 for o in out if o["verse"])
    print(f"> 60 words (long, often genuine run-ons): {long_}   <= 2 words: {short}   "
          f"no end punctuation (excl. headings/verse): {len(noend)}   verse/unsplit: {verse}")

    random.seed(0)
    print("--- 15 random segments ---")
    for o in random.sample(out, min(15, n)):
        print(f"{o['id']:5d} {o['text'][:110]}")
    print("--- 5 longest ---")
    for o in sorted(out, key=lambda o: -len(o["text"].split()))[:5]:
        print(f"{o['id']:5d} ({len(o['text'].split())}w) {o['text'][:110]}")
    print("--- 5 shortest non-heading ---")
    for o in sorted((o for o in out if not o["tag"].startswith("h")), key=lambda o: len(o["text"]))[:5]:
        print(f"{o['id']:5d} {o['text'][:110]}")

    # id == index in `out`, so out[k - 1] is the preceding segment
    short_segs = [o for o in plain if len(o["text"].split()) <= 2]
    print(f"--- {min(12, len(short_segs))} random short segments, with the end of the segment before ---")
    for o in random.sample(short_segs, min(12, len(short_segs))):
        k = o["id"]
        prev = out[k - 1]["text"][-50:] if k else ""
        print(f"{k:5d} {o['text']!r:24} <- ...{prev}")
    print(f"--- {min(8, len(noend))} segments without end punctuation ---")
    for o in random.sample(noend, min(8, len(noend))):
        print(f"{o['id']:5d} ...{o['text'][-70:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--lang", choices=["en", "hu"], required=True)
    args = ap.parse_args()

    with args.inp.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    out = []
    for r in rows:
        verse = r["n_lines"] > 1
        if r["tag"].startswith("h") or verse:
            parts = [r["text"]]
        else:
            parts = split_sentences(r["text"], args.lang)
        for k, t in enumerate(parts):
            out.append(
                {"id": len(out), "para": r["i"], "s": k, "doc": r["doc"],
                 "tag": r["tag"], "verse": verse, "text": t}
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    report(out)


if __name__ == "__main__":
    main()