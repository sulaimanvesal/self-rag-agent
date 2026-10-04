"""A tiny deterministic retriever: token-overlap (BM25-flavoured) ranking.

Self-RAG is retriever-agnostic -- the paper plugs in Contriever. Offline,
a transparent lexical retriever keeps the demo reproducible and shows
exactly *why* a passage was ranked where it was.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for with and or as at by "
    "it its this that these those what who when where why how which does do did".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str


@dataclass
class ScoredPassage:
    passage: Passage
    score: float
    matched_terms: list[str] = field(default_factory=list)


class LexicalRetriever:
    """Rank passages by IDF-weighted query-term overlap."""

    def __init__(self, corpus: list[Passage]):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = list(corpus)
        self._tokens: dict[str, list[str]] = {p.id: tokenize(p.title + " " + p.text) for p in corpus}
        df: dict[str, int] = {}
        for toks in self._tokens.values():
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(corpus)
        self._idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}

    def retrieve(self, query: str, k: int = 3) -> list[ScoredPassage]:
        if k < 1:
            raise ValueError("k must be >= 1")
        q = tokenize(query)
        if not q:
            return []
        results: list[ScoredPassage] = []
        for p in self.corpus:
            toks = self._tokens[p.id]
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            score, matched = 0.0, []
            for term in dict.fromkeys(q):
                if term in counts:
                    matched.append(term)
                    # saturating term frequency, BM25-style
                    tf = counts[term]
                    score += self._idf.get(term, 0.0) * (tf / (tf + 1.5))
            if score > 0:
                results.append(ScoredPassage(p, round(score, 6), matched))
        results.sort(key=lambda r: (-r.score, r.passage.id))
        return results[:k]
