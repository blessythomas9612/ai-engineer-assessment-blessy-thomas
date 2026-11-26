import pytest

from app.names import canonical_name, name_key, normalize_dashes


@pytest.mark.parametrize("typed", ["spiderman", "spider man", "Spider-Man", "SPIDER-MAN", "Spider‑Man"])
def test_spellings_map_to_the_canonical_name(typed):
    assert canonical_name(typed) == "Spider-Man"


def test_spaced_names_map_too():
    assert canonical_name("ironman") == "Iron Man"
    assert canonical_name("wonderwoman") == "Wonder Woman"


def test_unknown_names_are_kept():
    assert canonical_name(" Thanos ") == "Thanos"


def test_dash_like_characters_become_plain_hyphens():
    assert normalize_dashes("Spider‑Man – X—Men") == "Spider-Man - X-Men"
    assert name_key("Spider‑Man") == name_key("spider man") == "spiderman"
