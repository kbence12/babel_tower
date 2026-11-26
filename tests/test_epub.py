from pathlib import Path

import pytest

from babel_tower.epub import BR, LfsPointerError, clean, extract_paragraphs, trim_gutenberg
from babel_tower.records import Paragraph


def test_clean_normalises_and_counts_lines() -> None:
    decomposed = "ő"  # o + combining double acute, NFC turns it into ő
    assert clean(f"  sz{decomposed}­lő\u200b\n  fürt {BR} második  sor {BR}{BR}") == ("szőlő fürt második sor", 2)


def test_trim_gutenberg_keeps_only_the_text_between_markers() -> None:
    def para(text: str) -> Paragraph:
        return Paragraph(id=-1, doc=0, file="f", tag="p", n_lines=1, text=text)

    rows = [para("Licence"), para("*** START OF THE PROJECT GUTENBERG EBOOK X ***"), para("Body"),
            para("*** END OF THIS PROJECT GUTENBERG EBOOK X ***"), para("More licence")]  # fmt: skip
    assert [p.text for p in trim_gutenberg(rows)] == ["Body"]
    assert trim_gutenberg(rows[2:3]) == rows[2:3]


def test_lfs_pointer_is_reported(tmp_path: Path) -> None:
    pointer = tmp_path / "book.epub"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(LfsPointerError, match="git lfs pull"):
        extract_paragraphs(pointer)
