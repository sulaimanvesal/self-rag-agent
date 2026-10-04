"""The critic: emits IsREL / IsSUP / IsUSE reflection tokens.

In the paper a separate critic LM labels training data offline, and the
generator learns to emit the tokens itself. Here the critic is a small,
transparent rule system over token overlap -- every label it emits can be
explained from the passage and the segment, which is the point of an
educational re-implementation.
"""
from __future__ import annotations

from .retriever import Passage, tokenize
from .tokens import Critique

# Markers that a query needs external factual knowledge (Retrieve=yes).
_FACTUAL_STARTERS = (
    "who", "when", "where", "what year", "in what", "how tall", "how high",
    "how many", "how much", "which", "name the",
)
_NO_RETRIEVE_MARKERS = (
    "hello", "hi ", "hey ", "thank", "write a poem", "write a haiku",
    "tell me a joke", "opinion", "translate 'hello'",
)


def needs_retrieval(query: str) -> bool:
    """Predict the [Retrieve] token: retrieve *on demand*, not always.

    Conventional RAG retrieves for every input. Self-RAG's first
    reflection token lets the model skip retrieval for inputs that do
    not need external facts (greetings, creative tasks, chit-chat).
    """
    q = " " + query.lower().strip() + " "
    if any(m in q for m in _NO_RETRIEVE_MARKERS):
        return False
    if "?" in query or any(s in q for s in _FACTUAL_STARTERS):
        return True
    # Bare factual lookups without a question mark ("capital of France")
    words = tokenize(query)
    return len(words) >= 3 and any(w in query.lower() for w in ("capital", "born", "invented", "boils", "tallest", "highest"))


class RuleCritic:
    """Deterministic critic producing reflection tokens by overlap."""

    def is_relevant(self, query: str, passage: Passage) -> str:
        q = set(tokenize(query))
        if not q:
            return "irrelevant"
        p = set(tokenize(passage.title + " " + passage.text))
        overlap = len(q & p) / len(q)
        return "relevant" if overlap >= 0.4 else "irrelevant"

    def is_supported(self, segment: str, passage: Passage) -> str:
        s = set(tokenize(segment))
        if not s:
            return "no_support"
        p = set(tokenize(passage.text))
        containment = len(s & p) / len(s)
        if containment >= 0.85:
            return "fully_supported"
        if containment >= 0.45:
            return "partially_supported"
        return "no_support"

    def utility(self, query: str, segment: str) -> int:
        """IsUSE 1..5: does the segment actually answer the query?"""
        q = set(tokenize(query))
        s = set(tokenize(segment))
        if not q or not s:
            return 1
        coverage = len(q & s) / len(q)
        # A segment that covers the query terms and adds new content words
        # (the answer itself) scores highest.
        novelty = len(s - q)
        score = 1 + round(4 * coverage)
        if novelty == 0:
            score = min(score, 2)
        return max(1, min(5, score))

    def critique(self, query: str, passage: Passage, segment: str) -> Critique:
        return Critique(
            isrel=self.is_relevant(query, passage),
            issup=self.is_supported(segment, passage),
            isuse=self.utility(query, segment),
        )
