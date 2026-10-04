"""Tests for the Self-RAG re-implementation (offline, deterministic)."""
import pytest

from selfrag import (
    BUILTIN_CORPUS,
    Critique,
    LexicalRetriever,
    Passage,
    RuleCritic,
    SelfRAG,
    conventional_rag,
    needs_retrieval,
    tokenize,
)
from selfrag.generator import split_sentences


# --- reflection tokens -------------------------------------------------

def test_critique_score_ordering():
    good = Critique(isrel="relevant", issup="fully_supported", isuse=5)
    mid = Critique(isrel="relevant", issup="partially_supported", isuse=3)
    bad = Critique(isrel="irrelevant", issup="no_support", isuse=1)
    assert good.score() > mid.score() > bad.score()
    assert good.score() == pytest.approx(3.0)


def test_critique_validation():
    with pytest.raises(ValueError):
        Critique(isrel="maybe")
    with pytest.raises(ValueError):
        Critique(issup="supported-ish")
    with pytest.raises(ValueError):
        Critique(isuse=9)


def test_critique_render_contains_all_tokens():
    rendered = Critique().render()
    assert "IsREL" in rendered and "IsSUP" in rendered and "IsUSE" in rendered


# --- retrieve-on-demand decision ---------------------------------------

@pytest.mark.parametrize("q", [
    "How tall is the Eiffel Tower?",
    "Who won Nobel Prizes in Physics and Chemistry?",
    "When did humans first land on the Moon?",
])
def test_factual_questions_trigger_retrieval(q):
    assert needs_retrieval(q) is True


@pytest.mark.parametrize("q", ["Hello!", "Thank you so much", "Tell me a joke"])
def test_non_factual_inputs_skip_retrieval(q):
    assert needs_retrieval(q) is False


# --- retriever ----------------------------------------------------------

def test_tokenize_drops_stopwords():
    assert tokenize("What is the Eiffel Tower?") == ["eiffel", "tower"]


def test_retriever_ranks_eiffel_first():
    r = LexicalRetriever(BUILTIN_CORPUS).retrieve("How tall is the Eiffel Tower?", k=3)
    assert r and r[0].passage.id == "eiffel"
    assert "eiffel" in r[0].matched_terms


def test_retriever_empty_query_and_bad_k():
    retriever = LexicalRetriever(BUILTIN_CORPUS)
    assert retriever.retrieve("the is a", k=2) == []
    with pytest.raises(ValueError):
        retriever.retrieve("eiffel", k=0)
    with pytest.raises(ValueError):
        LexicalRetriever([])


# --- critic --------------------------------------------------------------

def test_critic_support_levels():
    critic = RuleCritic()
    passage = BUILTIN_CORPUS[0]
    exact = split_sentences(passage.text)[1]
    assert critic.is_supported(exact, passage) == "fully_supported"
    assert critic.is_supported("Dragons invented pizza in Atlantis.", passage) == "no_support"


def test_critic_relevance_and_utility_bounds():
    critic = RuleCritic()
    assert critic.is_relevant("Eiffel Tower height Paris", BUILTIN_CORPUS[0]) == "relevant"
    assert critic.is_relevant("quantum chromodynamics lattice gauge", BUILTIN_CORPUS[0]) == "irrelevant"
    for q in ("eiffel tower", "", "moon"):
        assert 1 <= critic.utility(q, "The Eiffel Tower stands 330 metres tall.") <= 5


# --- pipeline -------------------------------------------------------------

def test_pipeline_answers_eiffel_with_citation():
    result = SelfRAG().run("How tall is the Eiffel Tower?")
    assert result.retrieve is True
    assert "330 metres" in result.answer
    assert result.citations == ["eiffel"]
    assert result.converged_supported is True


def test_pipeline_answers_curie_and_moon():
    assert "Nobel" in SelfRAG().run("Who won Nobel Prizes in Physics and Chemistry?").answer
    moon = SelfRAG().run("When did humans first land on the Moon?")
    assert "1969" in moon.answer and moon.citations == ["moon"]


def test_pipeline_skips_retrieval_for_greeting():
    result = SelfRAG().run("Hello!")
    assert result.retrieve is False
    assert result.candidates == [] and result.citations == []
    assert "Hello" in result.answer


def test_pipeline_candidates_are_scored_and_sorted():
    result = SelfRAG(k=3).run("How tall is the Eiffel Tower?")
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.best_score == scores[0]


def test_pipeline_no_matching_passages():
    corpus = [Passage(id="x", title="Zebra", text="Zebras are striped equids native to Africa.")]
    result = SelfRAG(corpus=corpus).run("How tall is the Eiffel Tower?")
    assert result.retrieve is True
    assert "no relevant passages" in result.answer


def test_pipeline_rejects_empty_query():
    with pytest.raises(ValueError):
        SelfRAG().run("   ")
    with pytest.raises(ValueError):
        SelfRAG(k=0)


def test_conventional_rag_always_retrieves():
    baseline = conventional_rag()
    # Even for a greeting, conventional RAG has no skip mechanism; with no
    # lexical match it falls back, but for a factual query it answers from
    # the top-1 passage with no critique or citation bookkeeping.
    assert "330 metres" in baseline("How tall is the Eiffel Tower?")


def test_every_factual_demo_query_is_supported():
    rag = SelfRAG()
    for q in (
        "How tall is the Eiffel Tower?",
        "At what temperature does water boil at sea level?",
        "How high is Mount Everest?",
    ):
        result = rag.run(q)
        assert result.retrieve and result.converged_supported, q
