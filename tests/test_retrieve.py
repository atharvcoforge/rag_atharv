"""Lexical search, rank fusion, reranking, and parent expansion.

BM25 and RRF are pure and tested directly. Reranking is tested against a stub scorer so
the ordering logic is verified without a model in the loop.
"""

from __future__ import annotations

import math

import pytest

from ragpolicy.config import REPO_ROOT
from ragpolicy.ingest import Chunk, build_corpus
from ragpolicy.retrieve import BM25, expand_to_sections, reciprocal_rank_fusion, rerank
from ragpolicy.store import Hit

RAW = (REPO_ROOT / "policy.md").read_text(encoding="utf-8")
CORPUS = build_corpus(RAW)
PROPOSITIONS = [c for c in CORPUS if c.kind == "proposition"]


def ids(results: list[tuple[str, float]] | list[Hit] | list[Chunk]) -> list[str]:
    out = []
    for item in results:
        if isinstance(item, tuple):
            out.append(item[0])
        elif isinstance(item, Hit):
            out.append(item.chunk.chunk_id)
        else:
            out.append(item.chunk_id)
    return out


# -- Tokenisation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("hotels", "hotel"),
        ("receipts", "receipt"),
        ("expenses", "expense"),
        ("submitted", "submitting"),
        ("requires", "required"),
        ("days", "day"),
        ("policies", "policy"),
    ],
)
def test_stemmer_folds_query_and_policy_vocabulary_together(first: str, second: str) -> None:
    from ragpolicy.retrieve import stem

    assert stem(first) == stem(second)


@pytest.mark.parametrize("word", ["business", "$25", "class", "per"])
def test_stemmer_leaves_words_it_should_not_touch(word: str) -> None:
    from ragpolicy.retrieve import stem

    assert stem(word) == word


def test_tokenizer_keeps_amounts_and_splits_hyphens() -> None:
    from ragpolicy.retrieve import tokenize

    assert tokenize("Business-class costs $225!") == ["business", "class", "cost", "$225"]


def test_tokenizer_drops_function_words() -> None:
    from ragpolicy.retrieve import tokenize

    assert tokenize("is it in the policy") == ["policy"]


# -- BM25 --------------------------------------------------------------------------


def test_bm25_finds_the_numeric_threshold() -> None:
    """The case dense retrieval is worst at: an exact dollar amount."""
    bm25 = BM25(PROPOSITIONS)
    top = bm25.search("do I need a receipt for $25", limit=1)[0]

    assert "Receipts are required" in next(c.text for c in PROPOSITIONS if c.chunk_id == top[0])


def test_bm25_matches_on_a_duration() -> None:
    bm25 = BM25(PROPOSITIONS)
    top = bm25.search("30 days to submit my expense report", limit=1)[0]

    assert "30 days" in next(c.text for c in PROPOSITIONS if c.chunk_id == top[0])


def test_bm25_scores_are_non_negative_and_descending() -> None:
    results = BM25(PROPOSITIONS).search("hotel rate per night", limit=5)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(score >= 0 for score in scores)


def test_bm25_returns_nothing_for_wholly_unmatched_terms() -> None:
    assert BM25(PROPOSITIONS).search("parental leave sabbatical", limit=3) == []


def test_bm25_is_case_and_punctuation_insensitive() -> None:
    bm25 = BM25(PROPOSITIONS)
    assert ids(bm25.search("ALCOHOL!", limit=1)) == ids(bm25.search("alcohol", limit=1))


def test_bm25_rewards_term_frequency_and_penalises_length() -> None:
    """Sanity check the scoring formula rather than trusting it."""
    short = Chunk("short", "D", "1", "1", "T", "alcohol", "proposition", 0, 7)
    padded = Chunk("padded", "D", "1", "1", "T", "alcohol " + "filler " * 40, "proposition", 0, 10)
    results = dict(BM25([short, padded]).search("alcohol", limit=2))

    assert results["short"] > results["padded"]


def test_bm25_ranks_rarer_terms_higher() -> None:
    common = Chunk("common", "D", "1", "1", "T", "expenses are reimbursable", "proposition", 0, 1)
    rare = Chunk("rare", "D", "1", "1", "T", "expenses alcohol", "proposition", 0, 1)
    results = dict(BM25([common, rare]).search("alcohol expenses", limit=2))

    assert results["rare"] > results["common"]


def test_bm25_limit_is_respected() -> None:
    assert len(BM25(PROPOSITIONS).search("expenses reimbursable", limit=2)) == 2


# -- Reciprocal rank fusion --------------------------------------------------------


def test_rrf_rewards_appearing_in_both_rankings() -> None:
    """Agreement beats a single strong opinion, which is the whole point of fusing."""
    dense = ["solo_dense", "agreed", "c"]
    lexical = ["solo_lexical", "agreed", "d"]
    fused = ids(reciprocal_rank_fusion([dense, lexical]))

    # "agreed" is only second in each list but is the only one both retrievers found.
    assert fused[0] == "agreed"


def test_rrf_preserves_order_for_a_single_ranking() -> None:
    assert ids(reciprocal_rank_fusion([["a", "b", "c"]])) == ["a", "b", "c"]


def test_rrf_scores_match_the_published_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b"]], k=60))

    assert math.isclose(fused["a"], 1 / 61)
    assert math.isclose(fused["b"], 1 / 62 + 1 / 61)


def test_rrf_handles_an_empty_ranking() -> None:
    assert ids(reciprocal_rank_fusion([[], ["a"]])) == ["a"]


def test_rrf_needs_no_score_normalisation() -> None:
    """Ranks only: incomparable cosine and BM25 scales never meet."""
    assert ids(reciprocal_rank_fusion([["x"], ["y"]])) == ["x", "y"]


# -- Reranking ---------------------------------------------------------------------


def hits(*chunk_ids: str) -> list[Hit]:
    by_id = {c.chunk_id: c for c in CORPUS}
    return [Hit(chunk=by_id[i], distance=0.5) for i in chunk_ids]


def test_rerank_orders_by_descending_relevance_probability() -> None:
    scored = rerank(
        "when is my report due?",
        hits("expense-policy:v2.0:section-1:p1", "expense-policy:v2.0:section-6:p1"),
        scorer=lambda q, text: 0.9 if "30 days" in text else 0.2,
    )

    assert [s.chunk.chunk_id for s in scored] == [
        "expense-policy:v2.0:section-6:p1",
        "expense-policy:v2.0:section-1:p1",
    ]
    assert scored[0].score == 0.9


def test_rerank_keeps_the_original_distance_for_reporting() -> None:
    scored = rerank("q", hits("expense-policy:v2.0:section-1:p1"), scorer=lambda q, t: 0.7)

    assert scored[0].distance == 0.5
    assert scored[0].score == 0.7


def test_rerank_of_nothing_is_nothing() -> None:
    assert rerank("q", [], scorer=lambda q, t: 1.0) == []


# -- Parent expansion --------------------------------------------------------------


def test_propositions_expand_to_their_parent_section() -> None:
    sections = expand_to_sections(hits("expense-policy:v2.0:section-4:p2"), CORPUS)

    assert ids(sections) == ["expense-policy:v2.0:section-4"]
    assert sections[0].section_title == "Ground Transportation"


def test_expansion_dedupes_siblings_but_keeps_best_rank() -> None:
    sections = expand_to_sections(
        hits(
            "expense-policy:v2.0:section-4:p1",
            "expense-policy:v2.0:section-1:p1",
            "expense-policy:v2.0:section-4:p2",
        ),
        CORPUS,
    )

    assert ids(sections) == [
        "expense-policy:v2.0:section-4",
        "expense-policy:v2.0:section-1",
    ]


def test_expansion_passes_sections_through_unchanged() -> None:
    sections = expand_to_sections(hits("expense-policy:v2.0:section-2"), CORPUS)

    assert ids(sections) == ["expense-policy:v2.0:section-2"]


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("how much for a hotel room", "2"),
        ("do I need a receipt for a $30 dinner", "5"),
        ("deadline for submitting expenses", "6"),
        ("can I fly business class", "3"),
        ("is public transit covered", "4"),
        ("alcohol on the company card", "1"),
    ],
)
def test_bm25_covers_queries_that_share_vocabulary_with_the_policy(
    query: str, expected_section: str
) -> None:
    """Includes two cases that only pass because of deliberate design choices.

    "deadline" appears nowhere in the body text, only in the section title, and is found
    because BM25 indexes the contextual prefix. "business class" and "public transit" are
    written hyphenated in the policy and are found because the tokenizer splits hyphens.
    """
    results = BM25(PROPOSITIONS).search(query, limit=3)
    sections = {next(c.section for c in PROPOSITIONS if c.chunk_id == cid) for cid, _ in results}

    assert expected_section in sections


@pytest.mark.parametrize("query", ["can I drink wine with dinner", "is booze covered"])
def test_bm25_cannot_bridge_a_synonym_gap(query: str) -> None:
    """Documents the limit that justifies keeping a dense retriever at all.

    The policy says "Alcohol". Nothing lexical reaches it from "wine" or "booze".
    """
    results = BM25(PROPOSITIONS).search(query, limit=3)
    sections = {next(c.section for c in PROPOSITIONS if c.chunk_id == cid) for cid, _ in results}

    assert "1" not in sections
