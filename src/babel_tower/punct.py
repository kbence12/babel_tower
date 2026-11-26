"""Punctuation that OPUS-MT's vocabulary lacks, mapped to characters it has, and back.

OPUS-MT (Marian) tokenises with a fixed SentencePiece vocabulary that has no token for the en
dash "–" (Hungarian dialogue), the em dash "—" (English parenthetical dash) or « ». They become
<unk>, which the model can neither read nor write. So:

  to_model(text)     before tokenising, on both sides: dashes become "-", « » become '"'.
                     The model already writes a spaced "-" where a dialogue dash belongs.
  hungarian_output   after translating into Hungarian: a free-standing "-" (between spaces,
                     at the start of a line or before a comma) becomes the dialogue dash "–"
                     again. Hyphens inside or at the end of words ("három-hat", "Kéz- és")
                     are left alone.

English output is left as it is: whether a "-" there should be "—" cannot be told reliably.
"""

import re

TO_MODEL = str.maketrans({"–": "-", "—": "-", "«": '"', "»": '"'})
FREE_HYPHEN = re.compile(r"(^|\s)-(?=\s|,|$)")


def to_model(text: str) -> str:
    return text.translate(TO_MODEL)


def hungarian_output(text: str) -> str:
    return FREE_HYPHEN.sub(r"\1–", text)
