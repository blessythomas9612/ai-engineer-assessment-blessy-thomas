"""Retrieval over the local text dataset.

Each markdown file starts with a header:
    # <Title>
    Source: <url>          (optional)
    License: <licence>     (optional)
The rest is split into paragraphs ("chunks"). A question is matched against all
chunks with TF-IDF and cosine similarity; chunks below `min_score` are treated as
not relevant, so unrelated questions return nothing.

The best-matching article's lead paragraph is always included.

Considered and not used: hybrid search (TF-IDF + embeddings) with a cross-encoder
re-ranker. It ranked better, but added ~6 s per request on CPU and ~170 MB of models,
so TF-IDF was kept for a light, fast, explainable setup.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.names import CANONICAL_HEROES, name_key, normalize_dashes



@dataclass(frozen=True)
class Passage:
    doc_title: str
    file_name: str
    text: str
    source_url: str | None = None
    score: float = 0.0
    position: int = 0          # paragraph number within its article; 0 = the lead paragraph

    @property
    def reference(self) -> str:
        """Where this passage came from: the original URL if known, else the file."""
        return self.source_url or self.file_name

    @property
    def indexed_text(self) -> str:
        # The title is prepended so a paragraph that never names its subject still matches.
        return f"{self.doc_title}. {self.text}"


_MULTI_WORD_HEROES = [h for h in CANONICAL_HEROES if " " in h or "-" in h]


def with_joined_names(text: str) -> str:
    """Add joined forms of names, so a query typed "spiderman" or "ironman" matches text
    written "Spider-Man" or "Iron Man" (the tokenizer splits on "-" and spaces).
    Applies to every hyphenated word, and to the multi-word hero names."""
    text = normalize_dashes(text)
    joined = [w.replace("-", "") for w in re.findall(r"\b\w+(?:-\w+)+\b", text)]
    lowered = text.lower()
    joined += [name_key(h) for h in _MULTI_WORD_HEROES if h.lower() in lowered]
    return f"{text} {' '.join(joined)}" if joined else text


def load_passages(data_dir: Path) -> list[Passage]:
    chunks = []
    for path in sorted(Path(data_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue
        title = lines[0].lstrip("# ").strip() or path.stem
        source_url, body_start = None, 1
        for i, line in enumerate(lines[1:3], start=1):      # optional header lines
            if line.lower().startswith("source:"):
                source_url = line.split(":", 1)[1].strip() or None
                body_start = i + 1
            elif line.lower().startswith("license:"):
                body_start = i + 1
        position = 0
        for para in "\n".join(lines[body_start:]).split("\n\n"):
            para = " ".join(para.split())
            if len(para) > 40:
                chunks.append(Passage(title, path.name, para, source_url, position=position))
                position += 1
    if not chunks:
        raise ValueError(f"No documents found in {data_dir}")
    return chunks


def _top_k(chunks: list[Passage], scores: np.ndarray, k: int, min_score: float) -> list[Passage]:
    ranked = np.argsort(scores)[::-1][:k]
    return [
        _scored(chunks[i], scores[i]) for i in ranked if scores[i] >= min_score
    ]


def _scored(c: Passage, score: float) -> Passage:
    return Passage(c.doc_title, c.file_name, c.text, c.source_url, round(float(score), 3), c.position)


def with_lead(chunks: list[Passage], results: list[Passage]) -> list[Passage]:
    """Make sure the best-matching article's lead paragraph is included.

    A Wikipedia lead summarises the key facts (who created a character, when it first
    appeared). Ranking can miss it in favour of paragraphs that merely repeat the name,
    so it is added right after the top result."""
    if not results:
        return results
    top = results[0]
    if any(p.file_name == top.file_name and p.position == 0 for p in results):
        return results
    lead = next((c for c in chunks if c.file_name == top.file_name and c.position == 0), None)
    return results if lead is None else [top, lead] + results[1:]


class Retriever:
    """TF-IDF: rare, specific words count more than common ones."""

    def __init__(self, chunks: list[Passage], min_score: float = 0.1):
        self.chunks, self.min_score = chunks, min_score
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([with_joined_names(c.indexed_text) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.vectorizer.transform([with_joined_names(query)]), self.matrix)[0]

    def search(self, query: str, k: int = 3) -> list[Passage]:
        return with_lead(self.chunks, _top_k(self.chunks, self.scores(query), k, self.min_score))
