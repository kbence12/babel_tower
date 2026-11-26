import pytest

from babel_tower.records import Lang, Paragraph
from babel_tower.splitter import split_paragraphs, split_sentences


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        ("en", "It rained. We stayed in.", ["It rained.", "We stayed in."]),
        ("en", '"Stop!" He ran off.', ['"Stop!"', "He ran off."]),
        ("en", '"Where?" she asked. "Here."', ['"Where?" she asked.', '"Here."']),
        ("en", "Wait... what was that?", ["Wait... what was that?"]),
        ("en", "Mr. Brown sat down. Dr. Green stood up.", ["Mr. Brown sat down.", "Dr. Green stood up."]),
        ("en", "J. R. Smith wrote it. Then he left.", ["J. R. Smith wrote it.", "Then he left."]),
        ("en", "Not me, said I. Then we went home.", ["Not me, said I.", "Then we went home."]),
        ("en", "1. General remarks come first. Then details.", ["1. General remarks come first.", "Then details."]),
        ("en", "One. (Two.) Three.", ["One.", "(Two.)", "Three."]),
        ("en", "No end punctuation", ["No end punctuation"]),
        ("hu", "– Hogy vagy? – kérdezte Anna. – Jól.", ["– Hogy vagy? – kérdezte Anna.", "– Jól."]),
        ("hu", "– Igen! – Nem!", ["– Igen!", "– Nem!"]),
        ("hu", "Ez pl. egy példa. Másik mondat.", ["Ez pl. egy példa.", "Másik mondat."]),
        ("hu", "Kb. tíz perc volt. Aztán eljött.", ["Kb. tíz perc volt.", "Aztán eljött."]),
        ("hu", "Ott volt I. Béla is.", ["Ott volt I. Béla is."]),  # roman numeral, not a pronoun
        ("hu", "„Gyere ide!” Elment.", ["„Gyere ide!”", "Elment."]),
    ],
)
def test_split_sentences(lang: Lang, text: str, expected: list[str]) -> None:
    assert split_sentences(text, lang) == expected


@pytest.mark.xfail(strict=True, reason="known weak spot: capitalised name after ! or ? starts a new sentence")
def test_speech_tag_with_capitalised_name() -> None:
    assert split_sentences('"Oh, no!" Anna said.', "en") == ['"Oh, no!" Anna said.']


def test_punctuation_only_fragment_joins_previous() -> None:
    assert split_sentences("He stopped. . . . Then went on.", "en") == ["He stopped. . . .", "Then went on."]


def test_split_paragraphs_numbers_sentences_and_keeps_headings_and_verse_whole() -> None:
    paragraphs = [
        Paragraph(id=0, doc=1, file="c1.xhtml", tag="h2", n_lines=1, text="Chapter One. The Start."),
        Paragraph(id=1, doc=1, file="c1.xhtml", tag="p", n_lines=1, text="First. Second."),
        Paragraph(id=2, doc=1, file="c1.xhtml", tag="p", n_lines=2, text="Roses are red. Violets are blue."),
        Paragraph(id=3, doc=2, file="c2.xhtml", tag="p", n_lines=1, text="Third."),
    ]
    sentences = split_paragraphs(paragraphs, "en")
    assert [(s.id, s.para, s.pos, s.verse, s.text) for s in sentences] == [
        (0, 0, 0, False, "Chapter One. The Start."),
        (1, 1, 0, False, "First."),
        (2, 1, 1, False, "Second."),
        (3, 2, 0, True, "Roses are red. Violets are blue."),
        (4, 3, 0, False, "Third."),
    ]
    assert sentences[0].is_heading
    assert sentences[4].doc == 2
