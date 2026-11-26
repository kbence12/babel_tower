"""Paragraphs -> sentences, with rules for English and Hungarian.

Rule-based and dependency-free, so it is easy to read and debug. A boundary is a .!? or …
(plus any closing quotes) followed by whitespace, unless:
  - the next real character (skipping opening quotes, brackets and dashes) is lowercase.
    This keeps Hungarian speech tags together:  – Hogy vagy? – kérdezte Róbert.
  - the punctuation is a single "." after a known abbreviation (Mr., dr., pl., ...)
  - the punctuation is a single "." after an initial (J. R. Smith; roman numeral I.)
  - the "." ends a list marker such as "1. General remarks"

Headings and multi-line blocks (verse) are passed through unsplit.

Known weak spots (look for them in the report):
  - a speech tag starting with a capitalised name after ! or ?:  "Oh, no!" Anna said.
  - abbreviations missing from ABBREVIATIONS
  - "I." as a roman numeral in English body text (headings are not split, so this is rare)
"""

import re
from collections.abc import Iterable

from babel_tower.records import Lang, Paragraph, Sentence

# Abbreviations that (almost) never end a sentence. Extend as the report shows misses.
# Deliberately NOT included: stb, etc (they often do end sentences).
ABBREVIATIONS = frozenset({
    # English
    "mr", "mrs", "ms", "dr", "st", "mt", "sr", "jr", "prof", "gen", "col", "capt",
    "lt", "rev", "hon", "vs", "viz", "cf", "e.g", "i.e", "fig", "vol", "ch",
    # Hungarian
    "pl", "kb", "ún", "ill", "ifj", "id", "özv", "szül", "megh", "vö", "pld",
    "ált", "hrsz", "krt", "tkp", "ny", "sz", "kft",
})  # fmt: skip

CANDIDATE = re.compile(r"([.!?…]+)([\"'”’»)\]]*)(\s+)")
OPENERS = frozenset("\"'“‘«„([–—-")
LIST_MARKER = re.compile(r"\d{1,2}")
LAST_TOKEN = re.compile(r"[\w.]+$")


def split_paragraphs(paragraphs: Iterable[Paragraph], lang: Lang) -> list[Sentence]:
    """Number every sentence of the book. Headings and verse stay whole."""
    out: list[Sentence] = []
    for p in paragraphs:
        parts = [p.text] if p.is_heading or p.is_verse else split_sentences(p.text, lang)
        first_id = len(out)
        out.extend(
            Sentence(id=first_id + k, para=p.id, pos=k, doc=p.doc, tag=p.tag, verse=p.is_verse, text=text)
            for k, text in enumerate(parts)
        )
    return out


def split_sentences(text: str, lang: Lang) -> list[str]:
    parts, start = [], 0
    for m in CANDIDATE.finditer(text):
        if _is_boundary(text, m, lang, start):
            parts.append(text[start : m.end(2)].strip())
            start = m.end()
    if tail := text[start:].strip():
        parts.append(tail)

    # a fragment with no letters or digits (e.g. '?"' or '...') belongs to the previous one
    merged: list[str] = []
    for part in parts:
        if merged and not _has_alnum(part):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def _is_boundary(text: str, m: re.Match[str], lang: Lang, start: int) -> bool:
    nxt = _next_real_char(text, m.end())
    if nxt is None or not nxt.isalnum() or nxt.islower():
        # end of text, a lowercase continuation, or more punctuation (". . ."): same sentence
        return False
    if m.group(1) != ".":
        return True
    if LIST_MARKER.fullmatch(text[start : m.start()].strip()):
        return False
    token_match = LAST_TOKEN.search(text[: m.start()])
    token = token_match.group() if token_match else ""
    if token.lower() in ABBREVIATIONS:
        return False
    return not _is_initial(token) or (lang == "en" and token == "I")


def _next_real_char(text: str, i: int) -> str | None:
    """The first character from i on that is not whitespace or an opening quote/bracket/dash."""
    while i < len(text) and (text[i] in OPENERS or text[i].isspace()):
        i += 1
    return text[i] if i < len(text) else None


def _is_initial(token: str) -> bool:
    """'A', 'J.R' or 'A.A' (the final '.' is not part of the token)."""
    return bool(token) and all(len(part) == 1 and part.isupper() for part in token.split("."))


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)
