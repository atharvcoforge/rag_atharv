"""The gate that stops a reranker being promoted because it is small and quick.

Scores here are synthetic and chosen to reproduce the failures this project measured
against real models: uniform logprobs, confident inversion on a near miss, and
confidence on a question the policy cannot answer at all.
"""

from __future__ import annotations

from typing import Any

from ragpolicy.rerank_probe import CASES, CaseScore, ProbeResult, run_probe


def result(*scores: CaseScore, ms: float = 40.0) -> ProbeResult:
    return ProbeResult(model="candidate", scores=scores, ms_per_call=ms)


class PerfectClient:
    """A scorer that reads the question: high only where the excerpt truly governs it.

    Scoring by sentence alone would not work, and that is the point of the case set —
    the taxi rule governs "can I expense a taxi" and is a near miss for "can I expense a
    rental car".
    """

    def __init__(self) -> None:
        self.models: list[str] = []
        self.governing = {(c.question, c.governing) for c in CASES if c.governing}

    def yes_probability(self, prompt: str, **kwargs: Any) -> float:
        self.models.append(str(kwargs.get("model")))
        governs = any(question in prompt and rule in prompt for question, rule in self.governing)
        return 0.9 if governs else 0.1


def test_a_separated_and_faster_model_passes() -> None:
    probe = result(
        CaseScore("meals?", near_miss=0.10, governing=0.91),
        CaseScore("wine?", near_miss=0.60, governing=0.88),
        CaseScore("rental car?", near_miss=0.05),
    )

    assert probe.failures == []
    assert probe.passed is True


def test_uniform_scores_fail_as_degenerate() -> None:
    """The rejected Qwen3-Reranker GGUF returned -11.93 for every input, including junk."""
    probe = result(CaseScore("meals?", near_miss=0.5, governing=0.5))

    assert probe.passed is False
    assert any("degenerate" in reason for reason in probe.failures)


def test_a_near_miss_scored_above_the_rule_fails() -> None:
    probe = result(
        CaseScore("meals?", near_miss=0.10, governing=0.99),
        CaseScore("first-class?", near_miss=0.94, governing=0.30),
    )

    assert probe.passed is False
    assert any("not separated" in reason for reason in probe.failures)


def test_the_failure_names_the_question_that_broke() -> None:
    probe = result(
        CaseScore("meals?", near_miss=0.10, governing=0.99),
        CaseScore("first-class?", near_miss=0.94, governing=0.30),
    )

    assert any("'first-class?'" in reason for reason in probe.failures)


def test_margin_is_the_worst_case_not_the_average() -> None:
    """Rerank sorts within one question, so an average over the easy ones hides a miss."""
    probe = result(
        CaseScore("meals?", near_miss=0.01, governing=0.99),
        CaseScore("wine?", near_miss=0.80, governing=0.90),
    )

    assert abs(probe.margin - 0.10) < 1e-9


def test_confidence_on_an_unanswerable_question_fails() -> None:
    """qwen3:1.7b scored 'rental car' at 0.945 against the luxury-upgrade sentence."""
    probe = result(
        CaseScore("meals?", near_miss=0.10, governing=0.99),
        CaseScore("rental car?", near_miss=0.945),
    )

    assert probe.passed is False
    assert any("unanswerable" in reason for reason in probe.failures)


def test_a_separated_model_that_is_not_faster_fails() -> None:
    probe = result(CaseScore("meals?", near_miss=0.05, governing=0.95), ms=180.0)

    assert probe.passed is False
    assert any("no latency win" in reason for reason in probe.failures)


def test_probe_scores_every_governing_and_near_miss_sentence_with_the_named_model() -> None:
    client = PerfectClient()

    probe = run_probe(client, model="candidate:0.6b")

    governed = [case for case in CASES if case.governing]
    assert len(probe.scores) == len(CASES)
    assert len([s for s in probe.scores if s.governing is not None]) == len(governed)
    assert set(client.models) == {"candidate:0.6b"}
    # One warm-up call precedes the timed ones so a cold load is not charged to a case.
    assert len(client.models) == len(governed) + len(CASES) + 1
    assert probe.passed is True


def test_the_rental_car_question_contributes_only_near_misses() -> None:
    """No sentence governs rental cars; every excerpt for it must score as a negative."""
    rental = [case for case in CASES if "rental car" in case.question]

    assert len(rental) == 2
    assert all(case.governing is None for case in rental)
