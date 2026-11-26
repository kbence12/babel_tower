#!/usr/bin/env python3
"""Extract paragraphs from an EPUB, in reading order, into JSONL.

Usage (normally run through `babel-tower run`, which chains all steps):
    uv run python -m babel_tower.extract_epub book.epub cache/en.paragraphs.jsonl

One JSON object per block-level element:
    i        global paragraph index
    doc      index of the spine document (roughly a chapter, not guaranteed)
    file     file name inside the EPUB
    tag      p, h1..h6, blockquote or li
    n_lines  number of source lines (>1 usually means verse, split by <br>)
    text     cleaned text on a single line
"""
import argparse
import json
import re
import unicodedata
import warnings
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "li"]
BR = "\u2028"  # sentinel for <br>; plain newlines in the HTML source are just wrapping
WS = re.compile(r"\s+")
GUT_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.I)
GUT_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)


def clean(raw):
    """NFC-normalise, drop invisible characters, collapse whitespace.

    Returns (text on one line, number of non-empty source lines).
    NFC matters for Hungarian: it makes sure ő/ű are single code points.
    """
    s = unicodedata.normalize("NFC", raw)
    s = s.replace("\u00ad", "").replace("\u200b", "")  # soft hyphen, zero-width space
    lines = [WS.sub(" ", ln).strip() for ln in s.split(BR)]
    lines = [ln for ln in lines if ln]
    return " ".join(lines), len(lines)


def extract(path):
    with open(path, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            raise SystemExit(f"{path} is a Git LFS pointer, not the EPUB itself. "
                             "Install git-lfs, then run `git lfs install && git lfs pull` inside data/.")
    book =epub.read_epub(str(path), options={"ignore_ncx": True})
    rows = []
    for doc_idx, (idref, _linear) in enumerate(book.spine):
        item = book.get_item_with_id(idref)
        if item is None or isinstance(item, epub.EpubNav):
            continue
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "lxml")

        # Gutenberg licence blocks and page-number markers
        for el in soup.select("section.pg-boilerplate, [id^='pg-'], span.pagenum"):
            el.extract()
        # keep line structure of verse
        for br in soup.find_all("br"):
            br.replace_with(BR)

        for el in soup.find_all(BLOCK_TAGS):
            if el.find(BLOCK_TAGS):  # a container; its children are handled on their own
                continue
            text, n_lines = clean(el.get_text())
            if not text:
                continue
            rows.append(
                {
                    "doc": doc_idx,
                    "file": item.get_name(),
                    "tag": el.name,
                    "n_lines": n_lines,
                    "text": text,
                }
            )
    return rows


def trim_gutenberg(rows):
    """Cut everything outside the START/END markers, if the markers are present."""
    start = next((i for i, r in enumerate(rows) if GUT_START.search(r["text"])), None)
    if start is not None:
        rows = rows[start + 1 :]
    end = next((i for i, r in enumerate(rows) if GUT_END.search(r["text"])), None)
    if end is not None:
        rows = rows[:end]
    return rows


def extract_paragraphs(path):
    """EPUB -> list of paragraph rows with a global index `i`."""
    rows = trim_gutenberg(extract(path))
    for i, r in enumerate(rows):
        r["i"] = i
    return rows


def report(rows):
    words = sum(len(r["text"].split()) for r in rows)
    verse = sum(1 for r in rows if r["n_lines"] > 1)
    heads = sum(1 for r in rows if r["tag"].startswith("h"))
    blob = "".join(r["text"] for r in rows)
    good = sum(blob.count(c) for c in "őűŐŰ")
    bad = sum(blob.count(c) for c in "õûÕÛ")  # typical Latin-1/Latin-2 mangling of ő/ű
    print(f"blocks: {len(rows)}  words: {words}  headings: {heads}  multi-line (verse?): {verse}")
    print(f"ő/ű: {good}   õ/û: {bad}   (a high second number on Hungarian text means an encoding problem)")
    print("--- first 5 ---")
    for r in rows[:5]:
        print(f"[{r['tag']}] {r['text'][:100]}")
    print("--- last 5 ---")
    for r in rows[-5:]:
        print(f"[{r['tag']}] {r['text'][:100]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("epub", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    rows = extract_paragraphs(args.epub)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report(rows)


if __name__ == "__main__":
    main()