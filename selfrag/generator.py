"""Generators: an offline extractive generator + an optional real LM.

Offline, generation is *extractive*: for a (query, passage) pair, pick
the passage sentence that best answers the query. That mirrors what the
paper's generator does abstractly -- produce one candidate segment per
retrieved passage, in parallel -- while staying deterministic, free,
and honestly supportable by its source passage.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .retriever import Passage, tokenize

_SENTENCE = re.compile(r"[^.!?]+[.!?]")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


class ExtractiveGenerator:
    """Produce one candidate segment per passage (paper Sec. 3, Step 2)."""

    def generate(self, query: str, passage: Passage) -> str:
        q = set(tokenize(query))
        sentences = split_sentences(passage.text)
        if not sentences:
            return passage.text.strip()
        # Questions asking when / how much / how tall are answered by the
        # sentence carrying a number or date, even when another sentence
        # overlaps more query words ("first mission to land..." vs the
        # sentence that actually contains "20 July 1969").
        wants_number = query.lower().lstrip().startswith(
            ("when", "how tall", "how high", "how many", "how much", "what year", "in what")
        )
        wants_year = query.lower().lstrip().startswith(("when", "what year", "in what"))
        best, best_key = sentences[0], None
        for s in sentences:
            toks = set(tokenize(s))
            if wants_year:
                signal = 1 if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", s) else 0
            elif wants_number:
                signal = 1 if any(ch.isdigit() for ch in s) else 0
            else:
                signal = 0
            key = (signal, len(q & toks))
            if best_key is None or key > best_key:
                best, best_key = s, key
        return best

    def generate_direct(self, query: str) -> str:
        """Generation when [Retrieve]=no: no external facts are needed."""
        q = query.lower()
        if any(g in q for g in ("hello", "hi", "hey")):
            return "Hello! How can I help you today?"
        if "joke" in q:
            return "Why did the retriever break up with the generator? It needed space -- vector space."
        if "poem" in q or "haiku" in q:
            return "Old books, quiet light / a question finds its answer / pages turn to spring."
        if "thank" in q:
            return "You are welcome!"
        return "I can answer that directly, without retrieving any documents."


class OpenAILM:
    """Optional OpenAI-compatible backend (real runs only, needs a key).

    The pipeline only calls ``generate(query, passage)`` /
    ``generate_direct(query)``; swap this in and the same Self-RAG loop
    runs on a real model. Never used by the demo or the tests.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def _complete(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()

    def generate(self, query: str, passage: Passage) -> str:
        return self._complete(
            f"Answer the question in one sentence using only this passage.\n"
            f"Passage: {passage.text}\nQuestion: {query}"
        )

    def generate_direct(self, query: str) -> str:
        return self._complete(f"Answer in one short sentence: {query}")
