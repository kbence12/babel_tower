"""The records that flow through the pipeline. Each one is a line of a JSONL file."""

from dataclasses import dataclass
from typing import Literal

type Lang = Literal["en", "hu"]
LANGS: tuple[Lang, Lang] = ("en", "hu")


@dataclass(frozen=True, slots=True)
class Paragraph:
    """One block-level element of an EPUB, in reading order."""

    id: int  # global index over the book
    doc: int  # index of the spine document (roughly a chapter, not guaranteed)
    file: str  # file name inside the EPUB
    tag: str  # p, h1..h6, blockquote or li
    n_lines: int  # non-empty source lines; > 1 usually means verse split by <br>
    text: str  # cleaned, on a single line

    @property
    def is_heading(self) -> bool:
        return self.tag.startswith("h")

    @property
    def is_verse(self) -> bool:
        return self.n_lines > 1


@dataclass(frozen=True, slots=True)
class Sentence:
    """One segment the aligner works on: a sentence, a heading or a whole verse block."""

    id: int  # global index over the book; pairs refer to this
    para: int  # Paragraph.id it came from
    pos: int  # position within that paragraph
    doc: int
    tag: str
    verse: bool
    text: str

    @property
    def is_heading(self) -> bool:
        return self.tag.startswith("h")


@dataclass(frozen=True, slots=True)
class Pair:
    """An aligned group of 1..3 English and 1..3 Hungarian sentences."""

    pair: int
    type: str  # "{len(en_ids)}-{len(hu_ids)}", e.g. "1-2"
    sim: float  # cosine between the two windows' embeddings
    en_ids: list[int]
    hu_ids: list[int]
    en: str
    hu: str


@dataclass(frozen=True, slots=True)
class Unaligned:
    """A sentence the aligner left out of every group."""

    side: Lang
    id: int
    text: str
