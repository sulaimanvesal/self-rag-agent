#!/usr/bin/env python3
"""Offline Self-RAG demo -- no API key, no network.

Shows, per query: the [Retrieve] decision, the parallel candidate
segments with their reflection tokens, and the winning segment with its
citation. Also contrasts a greeting (Self-RAG skips retrieval) with
conventional always-retrieve RAG.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from selfrag import SelfRAG, conventional_rag  # noqa: E402

QUERIES = [
    "How tall is the Eiffel Tower?",
    "Who won Nobel Prizes in Physics and Chemistry?",
    "When did humans first land on the Moon?",
    "Hello!",
]


def main() -> int:
    rag = SelfRAG(k=3)
    baseline = conventional_rag()
    print("=" * 72)
    print("Self-RAG demo (offline) -- Asai et al., arXiv:2310.11511")
    print("=" * 72)
    for q in QUERIES:
        result = rag.run(q)
        print(f"\nQ: {q}")
        print(f"  [Retrieve: {'yes' if result.retrieve else 'no'}]")
        for c in result.candidates:
            marker = " <-- selected" if c.passage.id in result.citations else ""
            print(f"  candidate [{c.passage.id}] score={c.score:.2f} {c.critique.render()}{marker}")
            print(f"    {c.segment}")
        print(f"  ANSWER: {result.answer}")
        if result.citations:
            print(f"  citations: {result.citations}  fully_supported: {result.converged_supported}")
        print(f"  conventional RAG (always retrieves): {baseline(q)}")
    print("\nDone: retrieval happened only on demand, every factual answer")
    print("carries a citation, and the winner was chosen by critique tokens,")
    print("not by retrieval rank alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
