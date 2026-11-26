"""Helpers for matching character names however they are typed.

Users write "spiderman", "spider man", "Spider-Man" or "Spider‑Man" (with a
non-breaking hyphen); the dataset and the Superhero API write "Spider-Man".
These helpers make every step compare names the same way.
"""
import re

# Characters the keyword fallback recognises, in the spelling the Superhero API uses.
CANONICAL_HEROES = [
    "Spider-Man", "Wonder Woman", "Captain America", "Iron Man", "Superman", "Batman",
    "Wolverine", "Storm", "Nightcrawler", "Colossus", "Flash", "Human Torch", "Namor",
    "Hulk", "Thor", "Black Widow", "Aquaman", "Green Lantern",
]

# Hyphen-like characters that look like "-" but are different characters.
_DASHES = re.compile("[‐‑‒–—―−]")


def normalize_dashes(text: str) -> str:
    """Turn every dash-like character into a plain "-"."""
    return _DASHES.sub("-", text)


def name_key(name: str) -> str:
    """Comparison key: lower case, letters and digits only ("Spider-Man" -> "spiderman")."""
    return re.sub(r"[^a-z0-9]", "", normalize_dashes(name).lower())


_CANONICAL_BY_KEY = {name_key(h): h for h in CANONICAL_HEROES}


def canonical_name(name: str) -> str:
    """Known spelling for a hero name ("spiderman" -> "Spider-Man"); unknown names unchanged."""
    return _CANONICAL_BY_KEY.get(name_key(name), name.strip())
