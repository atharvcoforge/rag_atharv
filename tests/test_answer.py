"""Response contract, span verification, and the two abstention gates.

The pipeline is exercised with stub models so behaviour is deterministic. Live model
behaviour is the eval harness's job, not this file's.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from ragpolicy.answer import (
    REFUSAL,
    Answerer,
    Evidence,
    Thresholds,
    build_prompt,
    locate_quote,
    verify_spans,
)
from ragpolicy.config import REPO_ROOT
from ragpolicy.ingest import build_corpus
from ragpolicy.retrieve import ScoredHit

RAW = (REPO_ROOT / "policy.md").read_text(encoding="utf-8")
CORPUS = build_corpus(RAW)
BY_ID = {c.chunk_id: c for c in CORPUS}


def scored(chunk_id: str, score: float, distance: float = 0.2) -> ScoredHit:
    return ScoredHit(chunk=BY_ID[chunk_id], distance=distance, score=score)


class StubClient:
    """Stands in for OllamaClient with scripted generation and yes/no answers."""

    #: Any test that does not care about quoting gets a real sentence from section 1,
    #: which every fixture below retrieves. A quote absent from the evidence is refused.
    DEFAULT_RULE = "Alcohol is not reimbursable."

    def __init__(
        self,
        payload: dict[str, object],
        entailment: float | Callable[[str], float] = 0.99,
    ) -> None:
        self.payload = payload
        if payload and "governing_rule" not in payload:
            self.payload = {**payload, "governing_rule": self.DEFAULT_RULE}
        self.entailment = entailment
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.payload)

    def yes_probability(self, prompt: str, **kwargs: object) -> float:
        if callable(self.entailment):
            return self.entailment(prompt)
        return self.entailment


def make_answerer(client: StubClient, **overrides: float) -> Answerer:
    return Answerer(
        client=client,
        raw_policy=RAW,
        corpus=CORPUS,
        thresholds=Thresholds(**overrides) if overrides else Thresholds(),
    )


# -- Span verification -------------------------------------------------------------


def test_verify_spans_accepts_an_exact_quote() -> None:
    chunk = BY_ID["expense-policy:v2.0:section-1:p2"]
    assert verify_spans(RAW, [Evidence(chunk, chunk.start, chunk.end)]) is True


def test_verify_spans_rejects_an_offset_that_does_not_match_the_text() -> None:
    """A citation whose offsets do not contain its text is fabricated by definition."""
    chunk = BY_ID["expense-policy:v2.0:section-1:p2"]
    assert verify_spans(RAW, [Evidence(chunk, chunk.start + 5, chunk.end)]) is False


def test_verify_spans_rejects_out_of_bounds_offsets() -> None:
    chunk = BY_ID["expense-policy:v2.0:section-1:p2"]
    assert verify_spans(RAW, [Evidence(chunk, 0, len(RAW) + 50)]) is False


# -- Prompt construction -----------------------------------------------------------


def test_prompt_numbers_the_excerpts_and_names_their_sections() -> None:
    prompt = build_prompt("what is the meal cap?", [scored("expense-policy:v2.0:section-1", 0.9)])

    assert "[1]" in prompt
    assert "1. Meals" in prompt
    assert "$65" in prompt
    assert "what is the meal cap?" in prompt


def test_prompt_states_the_refusal_string_verbatim() -> None:
    prompt = build_prompt("q", [scored("expense-policy:v2.0:section-1", 0.9)])

    assert REFUSAL in prompt


def test_prompt_contains_only_the_supplied_excerpts() -> None:
    prompt = build_prompt("q", [scored("expense-policy:v2.0:section-1", 0.9)])

    assert "$225" not in prompt  # Section 2 was not retrieved.


# -- Gate 1: retrieval confidence --------------------------------------------------


def test_abstains_when_nothing_was_retrieved() -> None:
    answerer = make_answerer(StubClient({"answer": "anything", "section": "1"}))
    response = answerer.answer("what is the parental leave policy?", [])

    assert response.answer == REFUSAL
    assert response.citation is None
    assert response.retrieved_chunks == []
    assert response.trace["abstained_at"] == "retrieval"


def test_abstains_when_the_top_score_is_below_tau() -> None:
    client = StubClient({"answer": "Rental cars are covered.", "section": "4"})
    answerer = make_answerer(client, tau=0.6, delta=0.0)
    response = answerer.answer(
        "can I expense a rental car?", [scored("expense-policy:v2.0:section-4", 0.3)]
    )

    assert response.answer == REFUSAL
    assert response.trace["abstained_at"] == "retrieval"
    assert client.prompts == []  # Cheapest possible path: no generation at all.


def test_abstains_when_the_top_two_are_too_close() -> None:
    client = StubClient({"answer": "something", "section": "1"})
    answerer = make_answerer(client, tau=0.5, delta=0.25)
    response = answerer.answer(
        "ambiguous",
        [
            scored("expense-policy:v2.0:section-1", 0.80),
            scored("expense-policy:v2.0:section-2", 0.79),
        ],
    )

    assert response.answer == REFUSAL
    assert response.trace["abstained_at"] == "retrieval"


def test_answers_when_the_margin_is_wide_enough() -> None:
    client = StubClient(
        {
            "answer": "Employees may claim up to $65 per day.",
            "section": "1",
            "governing_rule": "Alcohol is not reimbursable.",
        }
    )
    answerer = make_answerer(client, tau=0.5, delta=0.25)
    response = answerer.answer(
        "meal cap?",
        [
            scored("expense-policy:v2.0:section-1", 0.95),
            scored("expense-policy:v2.0:section-2", 0.10),
        ],
    )

    assert response.answer == "Employees may claim up to $65 per day."
    assert response.trace["abstained_at"] is None


# -- Gate 2: post-generation verification ------------------------------------------


def test_abstains_when_a_claim_is_not_entailed_by_the_evidence() -> None:
    client = StubClient(
        {
            "answer": "Rental cars are reimbursable.",
            "section": "4",
            "governing_rule": "Luxury vehicle upgrades are not reimbursable.",
        },
        entailment=0.05,
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0, entailment=0.7)
    response = answerer.answer("rental car?", [scored("expense-policy:v2.0:section-4", 0.9)])

    assert response.answer == REFUSAL
    assert response.trace["abstained_at"] == "verification"


def test_keeps_an_entailed_answer() -> None:
    client = StubClient(
        {
            "answer": "Alcohol is not reimbursable.",
            "section": "1",
            "governing_rule": "Alcohol is not reimbursable.",
        },
        entailment=0.98,
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0, entailment=0.7)
    response = answerer.answer("wine?", [scored("expense-policy:v2.0:section-1", 0.9)])

    assert response.answer == "Alcohol is not reimbursable."
    assert response.trace["abstained_at"] is None


def test_honours_a_model_that_refuses_on_its_own() -> None:
    client = StubClient({"answer": REFUSAL, "section": ""})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    response = answerer.answer("q", [scored("expense-policy:v2.0:section-1", 0.9)])

    assert response.answer == REFUSAL
    assert response.citation is None


# -- Output contract ---------------------------------------------------------------


def test_response_matches_the_lab_schema() -> None:
    client = StubClient({"answer": "Employees may claim up to $65 per day.", "section": "1"})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    payload = answerer.answer(
        "meal cap?",
        [
            scored("expense-policy:v2.0:section-1", 0.95, distance=0.08),
            scored("expense-policy:v2.0:section-5", 0.30, distance=0.41),
        ],
    ).to_dict()

    assert set(payload) >= {"answer", "citation", "retrieved_chunks"}
    assert payload["citation"] == {
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1. Meals",
    }
    assert payload["retrieved_chunks"] == [
        {"section": "1. Meals", "distance": 0.08},
        {"section": "5. Receipts", "distance": 0.41},
    ]


def test_distances_are_json_numbers_not_strings() -> None:
    client = StubClient({"answer": "x", "section": "1"})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    payload = answerer.answer("q", [scored("expense-policy:v2.0:section-1", 0.9)]).to_dict()

    reloaded = json.loads(json.dumps(payload))
    for entry in reloaded["retrieved_chunks"]:
        assert isinstance(entry["distance"], float)


def test_at_most_three_chunks_are_returned() -> None:
    client = StubClient({"answer": "x", "section": "1"})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    hits = [
        scored(f"expense-policy:v2.0:section-{n}", 0.9 - n / 100, distance=n / 10)
        for n in range(1, 7)
    ]
    payload = answerer.answer("q", hits).to_dict()

    assert len(payload["retrieved_chunks"]) == 3


def test_retrieved_chunks_are_sorted_by_ascending_distance() -> None:
    client = StubClient({"answer": "x", "section": "1"})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    payload = answerer.answer(
        "q",
        [
            scored("expense-policy:v2.0:section-1", 0.9, distance=0.42),
            scored("expense-policy:v2.0:section-2", 0.8, distance=0.11),
        ],
    ).to_dict()

    distances = [c["distance"] for c in payload["retrieved_chunks"]]
    assert distances == sorted(distances)


def test_refusal_response_still_matches_the_schema() -> None:
    answerer = make_answerer(StubClient({"answer": "x", "section": "1"}), tau=0.99)
    payload = answerer.answer("q", [scored("expense-policy:v2.0:section-1", 0.1)]).to_dict()

    assert payload["answer"] == REFUSAL
    assert payload["citation"] is None
    assert isinstance(payload["retrieved_chunks"], list)


def test_trace_span_points_at_the_quoted_sentence_not_the_whole_section() -> None:
    """Sentence-level citation is what lets the UI highlight one line of the policy."""
    client = StubClient({"answer": "No, alcohol is excluded.", "section": "1"})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    response = answerer.answer("wine?", [scored("expense-policy:v2.0:section-1", 0.9)])

    span = response.trace["spans"][0]
    assert RAW[span["start"] : span["end"]] == "Alcohol is not reimbursable."


def test_a_quote_absent_from_the_evidence_is_refused() -> None:
    """The structural anti-hallucination guard: invented citations cannot survive."""
    client = StubClient(
        {
            "answer": "Rental cars are reimbursable up to $80 per day.",
            "section": "4",
            "governing_rule": "Rental cars are reimbursable up to $80 per day.",
        }
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    response = answerer.answer("rental car?", [scored("expense-policy:v2.0:section-4", 0.9)])

    assert response.answer == REFUSAL
    assert response.trace["abstained_at"] == "quote"


def test_locate_quote_matches_across_newline_whitespace() -> None:
    """The policy separates sentences with newlines; models join them with a space."""
    quote = (
        "Employees must purchase economy airfare. Business-class airfare "
        "requires written approval from a vice president."
    )
    found = locate_quote(quote, [scored("expense-policy:v2.0:section-3", 0.9)])
    assert found is not None
    hit, start, end = found
    assert hit.chunk.chunk_id == "expense-policy:v2.0:section-3"
    assert RAW[start:end] == BY_ID["expense-policy:v2.0:section-3"].text


def test_a_multi_sentence_quote_narrows_to_the_supporting_sentence() -> None:
    """Joined airfare rules are OK only when one sentence alone entails the answer."""
    both = (
        "Employees must purchase economy airfare. Business-class airfare "
        "requires written approval from a vice president."
    )

    def score(prompt: str) -> float:
        # Per-sentence prompts contain one rule; the economy sentence should win.
        if "must purchase economy" in prompt and "Business-class" not in prompt:
            return 0.99
        return 0.01

    client = StubClient(
        {
            "answer": "You are allowed to buy economy airfare.",
            "section": "3",
            "governing_rule": both,
        },
        entailment=score,
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0, entailment=0.5)
    response = answerer.answer("what class?", [scored("expense-policy:v2.0:section-3", 0.9)])

    assert response.answer == "You are allowed to buy economy airfare."
    assert response.trace["governing_rule"] == "Employees must purchase economy airfare."
    assert response.citation is not None
    assert response.citation["section"] == "3. Airfare"


def test_first_class_cannot_borrow_the_business_class_approval_rule() -> None:
    """Regression: first-class is not business-class; neither airfare sentence supports it."""
    both = (
        "Employees must purchase economy airfare. Business-class airfare "
        "requires written approval from a vice president."
    )

    def score(prompt: str) -> float:
        return 0.01

    client = StubClient(
        {
            "answer": (
                "You cannot book first-class airfare without written approval "
                "from a vice president."
            ),
            "section": "3",
            "governing_rule": both,
        },
        entailment=score,
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0, entailment=0.5)
    response = answerer.answer(
        "Can I book first-class airfare?",
        [scored("expense-policy:v2.0:section-3", 0.9)],
    )

    assert response.answer == REFUSAL
    assert response.citation is None
    # Peer-rule guard fires before entailment when the quoted span names business-class.
    assert response.trace["abstained_at"] in {"verification", "peer_rule"}


def test_first_class_plus_business_rule_is_refused_even_if_entailment_is_high() -> None:
    """Live failure mode: the model quotes business-class and a weak entailment says yes."""
    from ragpolicy.answer import peer_rule_mismatch

    rule = "Business-class airfare requires written approval from a vice president."
    answer = (
        "You cannot book first-class airfare without written approval from a vice president."
    )
    assert peer_rule_mismatch("Can I book first-class airfare?", rule, answer) is True

    client = StubClient(
        {"answer": answer, "section": "3", "governing_rule": rule},
        entailment=0.99,
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0, entailment=0.5)
    response = answerer.answer(
        "Can I book first-class airfare?",
        [scored("expense-policy:v2.0:section-3", 0.9)],
    )

    assert response.answer == REFUSAL
    assert response.citation is None
    assert response.trace["abstained_at"] == "peer_rule"


def test_sibling_sentence_recovery_when_model_quotes_the_limit_not_the_approval() -> None:
    """Hotels section has two sentences; quoting the cap must not kill a manager answer."""
    client = StubClient(
        {
            "answer": "You need manager approval before booking a $250 hotel.",
            "section": "2",
            "governing_rule": "Hotels are reimbursable up to $225 per night.",
        },
        entailment=lambda prompt: (
            0.99 if "manager must approve" in prompt.lower() else 0.01
        ),
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0, entailment=0.5)
    response = answerer.answer(
        "My hotel costs $250. What do I need?",
        [scored("expense-policy:v2.0:section-2", 0.9)],
    )

    assert "manager" in response.answer.lower()
    assert response.citation is not None
    assert response.citation["section"] == "2. Hotels"
    assert "manager must approve" in response.trace["governing_rule"].lower()
    assert response.trace["abstained_at"] is None


def test_the_economy_sentence_alone_grounds_a_first_class_answer() -> None:
    client = StubClient(
        {
            "answer": "No. Employees must purchase economy airfare.",
            "section": "3",
            "governing_rule": "Employees must purchase economy airfare.",
        }
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    response = answerer.answer(
        "Can I book first-class airfare?",
        [scored("expense-policy:v2.0:section-3", 0.9)],
    )

    assert response.answer == "No. Employees must purchase economy airfare."
    assert response.citation is not None
    assert response.citation["section"] == "3. Airfare"
    assert response.trace["governing_rule"] == "Employees must purchase economy airfare."


def test_an_empty_quote_is_refused() -> None:
    client = StubClient({"answer": "Probably fine.", "section": "", "governing_rule": ""})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    response = answerer.answer("q", [scored("expense-policy:v2.0:section-1", 0.9)])

    assert response.answer == REFUSAL
    assert response.trace["abstained_at"] == "quote"


def test_the_quote_decides_the_citation_even_if_the_section_field_disagrees() -> None:
    """The model's own quote is authoritative; a stray section number cannot override it."""
    client = StubClient(
        {
            "answer": "Hotels are capped at $225 per night.",
            "section": "9",  # nonsense
            "governing_rule": "Hotels are reimbursable up to $225 per night.",
        }
    )
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    response = answerer.answer(
        "hotel cap?",
        [
            scored("expense-policy:v2.0:section-1", 0.95),
            scored("expense-policy:v2.0:section-2", 0.90),
        ],
    )

    assert response.citation is not None
    assert response.citation["section"] == "2. Hotels"


def test_trace_records_confidence_and_stage_timings() -> None:
    client = StubClient({"answer": "x", "section": "1"})
    answerer = make_answerer(client, tau=0.1, delta=0.0)
    trace = answerer.answer("q", [scored("expense-policy:v2.0:section-1", 0.9)]).trace

    assert 0.0 <= trace["confidence"] <= 1.0
    assert trace["retrieval_score"] == 0.9


@pytest.mark.parametrize("malformed", ["not json at all", "{}", '{"answer": ""}'])
def test_malformed_generation_falls_back_to_refusing(malformed: str) -> None:
    class Broken(StubClient):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return malformed

    answerer = make_answerer(Broken({}), tau=0.1, delta=0.0)
    response = answerer.answer("q", [scored("expense-policy:v2.0:section-1", 0.9)])

    assert response.answer == REFUSAL
