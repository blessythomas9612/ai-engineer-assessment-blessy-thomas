"""Build the text dataset from Wikipedia comics-history articles.

Run once from the project root:  python scripts/fetch_wikipedia.py
It writes one markdown file per article to data/comics/, each starting with:

    # <Title>
    Source: https://en.wikipedia.org/wiki/<Title>
    License: CC BY-SA 4.0

followed by the article's paragraphs. Reference sections (See also, References,
External links, ...) are dropped and each article is capped in length, so the
dataset stays small and focused.
"""
import re
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

ARTICLES = [
    # Characters and teams
    "Superman", "Batman", "Wonder Woman", "Spider-Man", "X-Men", "Captain America",
    "Iron Man", "Fantastic Four", "Wolverine (character)", "Hulk",
    # Publishers and creators
    "Marvel Comics", "DC Comics", "Timely Comics", "Stan Lee", "Jack Kirby",
    # Eras, industry history and film
    "Golden Age of Comic Books", "Silver Age of Comic Books", "Bronze Age of Comic Books",
    "Comics Code Authority", "Superhero film",
]

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API clients to identify themselves.
USER_AGENT = "ai-engineer-assessment/1.0 (educational project; https://github.com/blessythomas9612)"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "comics"
SKIP_SECTIONS = {"see also", "references", "notes", "external links", "further reading",
                 "bibliography", "footnotes", "citations", "sources", "collected editions",
                 "in other media", "other versions"}
MAX_CHARS = 12_000       # per article, keeps the dataset focused
MIN_PARAGRAPH = 60       # drop captions and fragments


def clean_extract(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Turn a plain-text Wikipedia extract into a list of clean paragraphs."""
    paragraphs, size, skipping = [], 0, False
    for line in text.splitlines():
        line = line.strip()
        heading = re.fullmatch(r"(=+)\s*(.*?)\s*\1", line)
        if heading:
            name = heading.group(2).lower()
            level = len(heading.group(1))
            if level == 2:                      # top-level section decides skipping
                skipping = name in SKIP_SECTIONS
            elif name in SKIP_SECTIONS:
                skipping = True
            continue
        if skipping or len(line) < MIN_PARAGRAPH:
            continue
        if size + len(line) > max_chars:
            break
        paragraphs.append(line)
        size += len(line)
    return paragraphs


def fetch(client: httpx.Client, title: str) -> tuple[str, str]:
    params = {"action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
              "redirects": 1, "titles": title}
    resp = client.get(API_URL, params=params)
    resp.raise_for_status()
    page = next(iter(resp.json()["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"article not found: {title}")
    return page["title"], page["extract"]


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20) as client:
        for requested in ARTICLES:
            try:
                title, extract = fetch(client, requested)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                print(f"FAILED  {requested}: {exc}")
                failures += 1
                continue
            paragraphs = clean_extract(extract)
            url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
            body = f"# {title}\nSource: {url}\nLicense: CC BY-SA 4.0\n\n" + "\n\n".join(paragraphs) + "\n"
            path = OUT_DIR / f"{slug(title)}.md"
            path.write_text(body, encoding="utf-8")
            print(f"saved   {path.name:40} {len(paragraphs):3} paragraphs")
    print(f"\nDone: {len(ARTICLES) - failures} saved, {failures} failed -> {OUT_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
