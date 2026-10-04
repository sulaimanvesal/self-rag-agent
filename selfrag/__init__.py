"""Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection (Asai et al., ICLR 2024, arXiv:2310.11511).

A runnable, offline re-implementation: on-demand retrieval, parallel
per-passage generation, reflection tokens (Retrieve / IsREL / IsSUP /
IsUSE), and segment-level selection by critique score.
"""
from .corpus import BUILTIN_CORPUS
from .critic import RuleCritic, needs_retrieval
from .generator import ExtractiveGenerator, OpenAILM, split_sentences
from .pipeline import Candidate, SelfRAG, SelfRAGResult, conventional_rag
from .retriever import LexicalRetriever, Passage, ScoredPassage, tokenize
from .tokens import Critique

__all__ = [
    "BUILTIN_CORPUS",
    "Candidate",
    "Critique",
    "ExtractiveGenerator",
    "LexicalRetriever",
    "OpenAILM",
    "Passage",
    "RuleCritic",
    "ScoredPassage",
    "SelfRAG",
    "SelfRAGResult",
    "conventional_rag",
    "needs_retrieval",
    "split_sentences",
    "tokenize",
]

__version__ = "0.1.0"
