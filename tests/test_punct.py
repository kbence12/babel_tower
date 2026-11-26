import pytest

from babel_tower.punct import hungarian_output, to_model


def test_to_model_replaces_characters_missing_from_the_vocabulary() -> None:
    assert to_model("– Igen – mondta. «Na»") == '- Igen - mondta. "Na"'
    assert to_model("wait—what") == "wait-what"


@pytest.mark.parametrize(
    ("model_output", "restored"),
    [
        ("- Gyere ide - mondta Anna.", "– Gyere ide – mondta Anna."),
        ("- Nos - mondta lassan -, az attól függ.", "– Nos – mondta lassan –, az attól függ."),
        ("Három-hat hónap.", "Három-hat hónap."),  # hyphen inside a word
        ("Kéz- és lábtörést!", "Kéz- és lábtörést!"),  # hyphen ending a word
    ],
)
def test_hungarian_output_restores_only_free_standing_hyphens(model_output: str, restored: str) -> None:
    assert hungarian_output(model_output) == restored


def test_round_trip_of_dialogue() -> None:
    text = "– Hogy vagy? – kérdezte Anna."
    assert hungarian_output(to_model(text)) == text
