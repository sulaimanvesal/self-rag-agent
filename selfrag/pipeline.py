"""The Self-RAG pipeline: retrieve on demand, generate in parallel,
critique with reflection tokens, and select the best segment.

Paper (Asai et al., arXiv:2310.11511), Sec. 3 / Fig. 1, right side:

1. Predict ``[Retrieve]`` -- skip retrieval when it is not needed.
2. If retrieving: process the top-k passages *in parallel*, generating
   one candidate segment per passage.
3. Emit critique tokens (``IsREL`` / ``IsSUP`` / ``IsUSE``) per segment.
4. Segment-level beam search: keep the segment maximising the weighted
   critique score, and cite its passage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import BUILTIN_CORPUS
from .critic import RuleCritic, needs_retrieval
from .generator import ExtractiveGenerator
from .retriever import LexicalRetriever, Passage
from .tokens import Critique


@dataclass
class Candidate:
    passage: Passage
    segment: str
    critique: Critique

    @property
    def score(self) -> float:
        return self.critique.score()


@dataclass
class SelfRAGResult:
    query: str
    retrieve: bool                       # the [Retrieve] decision
    answer: str
    candidates: list[Candidate] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)   # passage ids
    best_score: float | None = None

    @property
    def converged_supported(self) -> bool:
        """True when the winning segment is fully supported by its source."""
        if not self.citations or not self.candidates:
            return False
        best = self.candidates[0]
        return best.passage.id in self.citations and best.critique.issup == "fully_supported"


class SelfRAG:
    def __init__(self, corpus: list[Passage] | None = None, generator=None,
                 critic: RuleCritic | None = None, k: int = 3):
        self.corpus = list(corpus) if corpus is not None else list(BUILTIN_CORPUS)
        self.retriever = LexicalRetriever(self.corpus)
        self.generator = generator or ExtractiveGenerator()
        self.critic = critic or RuleCritic()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def run(self, query: str) -> SelfRAGResult:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        # Step 1 -- [Retrieve] on demand.
        if not needs_retrieval(query):
            return SelfRAGResult(
                query=query, retrieve=False,
                answer=self.generator.generate_direct(query),
            )

        # Step 2 -- retrieve, then generate one segment per passage.
        scored = self.retriever.retrieve(query, k=self.k)
        candidates: list[Candidate] = []
        for sp in scored:
            segment = self.generator.generate(query, sp.passage)
            # Step 3 -- critique tokens for this segment.
            critique = self.critic.critique(query, sp.passage, segment)
            if critique.isrel == "irrelevant":
                # The paper still scores the segment, but an irrelevant
                # passage cannot win a well-weighted beam; keep it in the
                # trace for transparency and let the score decide.
                pass
            candidates.append(Candidate(sp.passage, segment, critique))

        if not candidates:
            return SelfRAGResult(
                query=query, retrieve=True,
                answer="I retrieved on demand but found no relevant passages in the corpus.",
            )

        # Step 4 -- segment-level selection by weighted critique score.
        candidates.sort(key=lambda c: (-c.score, c.passage.id))
        best = candidates[0]
        return SelfRAGResult(
            query=query, retrieve=True,
            answer=best.segment,
            candidates=candidates,
            citations=[best.passage.id],
            best_score=best.score,
        )


def conventional_rag(corpus: list[Passage] | None = None, k: int = 1):
    """Baseline factory: always retrieve top-1, never critique.

    Returns a callable ``query -> str`` so the demo/tests can contrast
    fixed retrieval (paper Fig. 1, left) with Self-RAG (right).
    """
    rag_corpus = list(corpus) if corpus is not None else list(BUILTIN_CORPUS)
    retriever = LexicalRetriever(rag_corpus)
    gen = ExtractiveGenerator()

    def _run(query: str) -> str:
        scored = retriever.retrieve(query, k=k)
        if not scored:
            return gen.generate_direct(query)
        return gen.generate(query, scored[0].passage)

    return _run
