from scripts.fetch_wikipedia import clean_extract, slug

SAMPLE = """Intro paragraph about a fictional hero, long enough to be kept as a real paragraph of text.
Short caption.

== History ==
A history paragraph that explains when the character was created and by whom, in enough words.
=== Later years ===
A later paragraph describing how the character changed over several decades of publication.

== See also ==
A see-also line that is long enough to be a paragraph but should be skipped entirely here.
== References ==
A reference line that is long enough to be a paragraph but should also be skipped entirely.
"""


def test_clean_extract_keeps_content_and_drops_reference_sections():
    paragraphs = clean_extract(SAMPLE)
    assert len(paragraphs) == 3
    assert paragraphs[0].startswith("Intro paragraph")
    assert not any("see-also" in p or "reference line" in p for p in paragraphs)


def test_clean_extract_respects_length_cap():
    assert len(clean_extract(SAMPLE, max_chars=150)) == 1


def test_slug():
    assert slug("Wolverine (character)") == "wolverine_character"
