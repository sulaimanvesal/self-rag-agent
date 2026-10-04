"""Reflection tokens from Self-RAG (Asai et al., 2023, arXiv:2310.11511).

The paper expands the LM vocabulary with special *reflection tokens* that
the model emits as part of generation:

==========  ============================================================
Token       Values in the paper
==========  ============================================================
Retrieve    ``yes`` / ``no`` / ``continue`` -- retrieve on demand?
IsREL       ``relevant`` / ``irrelevant`` -- is a passage relevant?
IsSUP       ``fully supported`` / ``partially supported`` / ``no support``
IsUSE       ``{1, 2, 3, 4, 5}`` -- overall utility of a segment
==========  ============================================================

This module is the single source of truth for those vocabularies and for
the numeric scores used by segment-level beam search (paper Sec. 3.2):
a segment is scored by a weighted sum of its critique-token values.
"""
from __future__ import annotations

from dataclasses import dataclass

RETRIEVE_VALUES = ("yes", "no", "continue")
ISREL_VALUES = ("relevant", "irrelevant")
ISSUP_VALUES = ("fully_supported", "partially_supported", "no_support")

# Numeric values used for ranking (paper: segment-level beam search
# maximises a weighted sum of critique scores).
ISREL_SCORE = {"relevant": 1.0, "irrelevant": 0.0}
ISSUP_SCORE = {"fully_supported": 1.0, "partially_supported": 0.5, "no_support": 0.0}


@dataclass(frozen=True)
class Critique:
    """The three critique tokens emitted for one (passage, segment) pair."""

    isrel: str = "relevant"
    issup: str = "fully_supported"
    isuse: int = 5

    def __post_init__(self) -> None:
        if self.isrel not in ISREL_VALUES:
            raise ValueError(f"IsREL must be one of {ISREL_VALUES}, got {self.isrel!r}")
        if self.issup not in ISSUP_VALUES:
            raise ValueError(f"IsSUP must be one of {ISSUP_VALUES}, got {self.issup!r}")
        if not 1 <= int(self.isuse) <= 5:
            raise ValueError(f"IsUSE must be in 1..5, got {self.isuse!r}")

    def score(self, w_rel: float = 1.0, w_sup: float = 1.0, w_use: float = 1.0) -> float:
        """Weighted segment score, as in the paper's beam-search objective."""
        return (
            w_rel * ISREL_SCORE[self.isrel]
            + w_sup * ISSUP_SCORE[self.issup]
            + w_use * (self.isuse / 5.0)
        )

    def render(self) -> str:
        """Render as the paper's bracketed reflection tokens."""
        sup = {
            "fully_supported": "Fully Supported",
            "partially_supported": "Partially Supported",
            "no_support": "No Support",
        }[self.issup]
        rel = "Relevant" if self.isrel == "relevant" else "Irrelevant"
        return f"[IsREL:{rel}] [IsSUP:{sup}] [IsUSE:{self.isuse}]"
