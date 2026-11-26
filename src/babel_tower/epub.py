"""EPUB -> paragraphs in reading order."""

import re
import unicodedata
import warnings
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from ebooklib import epub

from babel_tower.records import Paragraph

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "li"]
BR = "\u2028"  # stands in for <br>; plain newlines in the HTML source are just wrapping
INVISIBLE = str.maketrans("", "", "\u00ad\u200b")  # soft hyphen, zero-width space
WHITESPACE = re.compile(r"\s+")
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
# Gutenberg licence blocks and page-number markers
BOILERPLATE = "section.pg-boilerplate, [id^='pg-'], span.pagenum"
LFS_POINTER = b"version https://git-lfs"


class LfsPointerError(ValueError):
    """The file is a Git LFS pointer, i.e. the real content was never downloaded."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} is a Git LFS pointer, not the EPUB itself. "
            "Install git-lfs, then run `git lfs install && git lfs pull` inside data/."
        )


def extract_paragraphs(path: Path) -> list[Paragraph]:
    """All non-empty block elements of the book, without Gutenberg front and back matter."""
    with path.open("rb") as f:
        if f.read(len(LFS_POINTER)) == LFS_POINTER:
            raise LfsPointerError(path)
    blocks = trim_gutenberg(list(_iter_blocks(path)))
    return [replace(p, id=i) for i, p in enumerate(blocks)]


def clean(raw: str) -> tuple[str, int]:
    """NFC-normalise, drop invisible characters, collapse whitespace.

    Returns the text on one line and its number of non-empty source lines.
    NFC matters for Hungarian: it makes sure ő/ű are single code points.
    """
    text = unicodedata.normalize("NFC", raw).translate(INVISIBLE)
    lines = [line for line in (WHITESPACE.sub(" ", ln).strip() for ln in text.split(BR)) if line]
    return " ".join(lines), len(lines)


def trim_gutenberg(blocks: list[Paragraph]) -> list[Paragraph]:
    """Cut everything outside the START/END markers, if the markers are present."""
    start = next((i for i, b in enumerate(blocks) if GUTENBERG_START.search(b.text)), None)
    if start is not None:
        blocks = blocks[start + 1 :]
    end = next((i for i, b in enumerate(blocks) if GUTENBERG_END.search(b.text)), None)
    if end is not None:
        blocks = blocks[:end]
    return blocks


def _iter_blocks(path: Path) -> Iterator[Paragraph]:
    """Paragraphs in spine order, all with id -1 (numbered after trimming)."""
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    for doc_idx, (idref, _linear) in enumerate(book.spine):
        item = book.get_item_with_id(idref)
        if item is None or isinstance(item, epub.EpubNav) or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(item.get_content(), "lxml")
        for el in soup.select(BOILERPLATE):
            el.extract()
        for br in soup.find_all("br"):
            br.replace_with(BR)  # keep the line structure of verse

        for el in soup.find_all(BLOCK_TAGS):
            if not isinstance(el, Tag) or el.find(BLOCK_TAGS):
                continue  # a container: its children are handled on their own
            text, n_lines = clean(el.get_text())
            if text:
                yield Paragraph(id=-1, doc=doc_idx, file=item.get_name(), tag=el.name, n_lines=n_lines, text=text)
