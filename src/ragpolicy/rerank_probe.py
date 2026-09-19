"""Does a candidate cross-encoder actually discriminate, and is it actually faster?

Three rerankers have already been tried and rejected here, and neither failure was
visible from the model card. ``dengcao/Qwen3-Reranker-0.6B:Q8_0`` returned the same
-11.93 logprob for every input, including "The capital of France is". ``qwen3:0.6b`` and
``qwen3:1.7b`` were fast and confident and wrong: the 1.7B scored "rental car" at 0.945
against a sentence about luxury vehicle upgrades.

So this module exists to make promoting a reranker a measurement rather than a
preference. It scores a fixed set of policy questions against the sentence that governs
them and the sentence most likely to be mistaken for it, and fails the candidate unless
the scores are non-degenerate, separated, and cheaper than the 8B they would replace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ragpolicy.retrieve import RERANK_SYSTEM, rerank_prompt

#: What ``qwen3:8b`` costs per candidate today. A replacement has to beat it.
BASELINE_MS = 153.0
#: Below this, the model is returning one number regardless of input (the 0.6B GGUF).
MIN_SPREAD = 0.05
#: Worst per-case gap between the governing sentence and its near miss. ``qwen3:8b``
#: measures 0.17, held down by "wine" against the meal-allowance sentence, which is a
#: defensible near miss rather than a mistake. A replacement may not be sloppier.
MIN_MARGIN = 0.15
#: On a question the policy cannot settle, every excerpt must score low in absolute
#: terms, because that is what the calibrated tau gate reads. qwen3:1.7b scored the
#: luxury-upgrade sentence at 0.945 for "rental car" and would have sailed through.
MAX_UNANSWERABLE = 0.5


@dataclass(frozen=True, slots=True)
class ProbeCase:
    """A question, the sentence that governs it, and the sentence that nearly does."""

    question: str
    near_miss: str
    governing: str | None = None


#: Drawn from the failures this system actually had, not from generic relevance pairs.
CASES: tuple[ProbeCase, ...] = (
    ProbeCase(
        question="How much can I claim for meals per day?",
        governing="Employees may claim up to $65 per day for meals while traveling overnight.",
        near_miss="Hotels are reimbursable up to $225 per night.",
    ),
    ProbeCase(
        question="Can I expense a glass of wine with dinner?",
        governing="Alcohol is not reimbursable.",
        near_miss="Employees may claim up to $65 per day for meals while traveling overnight.",
    ),
    ProbeCase(
        question="Do I need a receipt for a $20 lunch?",
        governing="Receipts are required for individual expenses of $25 or more.",
        near_miss="Expense reports must be submitted within 30 days after travel ends.",
    ),
    # The measured trap: the business-class sentence is a peer of first-class, not its rule.
    ProbeCase(
        question="Can I book first-class airfare?",
        governing="Employees must purchase economy airfare.",
        near_miss="Business-class airfare requires written approval from a vice president.",
    ),
    ProbeCase(
        question="Can I expense a taxi to the airport?",
        governing="Taxi, rideshare, train, and public-transit expenses are reimbursable.",
        near_miss="Luxury vehicle upgrades are not reimbursable.",
    ),
    ProbeCase(
        question="What is the nightly limit for hotels?",
        governing="Hotels are reimbursable up to $225 per night.",
        near_miss="Employees may claim up to $65 per day for meals while traveling overnight.",
    ),
    # No sentence governs rental cars. Both of these are near misses, and a reranker that
    # scores either one highly is the reason this question gets a fabricated answer.
    ProbeCase(
        question="Can I expense a rental car?",
        near_miss="Taxi, rideshare, train, and public-transit expenses are reimbursable.",
    ),
    ProbeCase(
        question="Can I expense a rental car?",
        near_miss="Luxury vehicle upgrades are not reimbursable.",
    ),
)


@dataclass(frozen=True, slots=True)
class CaseScore:
    question: str
    near_miss: float
    governing: float | None = None

    @property
    def margin(self) -> float | None:
        """How far the rule beat its near miss. ``None`` where no rule governs."""
        return None if self.governing is None else self.governing - self.near_miss


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model: str
    scores: tuple[CaseScore, ...]
    ms_per_call: float
    baseline_ms: float = BASELINE_MS

    @property
    def spread(self) -> float:
        values = [s.near_miss for s in self.scores]
        values += [s.governing for s in self.scores if s.governing is not None]
        return max(values) - min(values) if values else 0.0

    @property
    def margin(self) -> float:
        """The worst case, not the average.

        Reranking only ever sorts candidates within one question, so the comparison that
        matters is per question. One inverted question is enough to cite the wrong rule,
        and an average over the easy ones would hide it.
        """
        margins = [s.margin for s in self.scores if s.margin is not None]
        return min(margins) if margins else 0.0

    @property
    def unanswerable_high(self) -> float:
        """Best score any excerpt got on a question the policy does not answer."""
        orphans = [s.near_miss for s in self.scores if s.governing is None]
        return max(orphans) if orphans else 0.0

    @property
    def failures(self) -> list[str]:
        reasons = []
        if self.spread < MIN_SPREAD:
            reasons.append(f"degenerate: scores span only {self.spread:.3f}")
        if self.margin < MIN_MARGIN:
            worst = min(
                (s for s in self.scores if s.margin is not None),
                key=lambda s: s.margin or 0.0,
                default=None,
            )
            where = f" on {worst.question!r}" if worst else ""
            reasons.append(f"not separated: worst margin {self.margin:.3f}{where}")
        if self.unanswerable_high > MAX_UNANSWERABLE:
            reasons.append(
                f"confident on an unanswerable question: {self.unanswerable_high:.3f} "
                f"> {MAX_UNANSWERABLE}"
            )
        if self.ms_per_call >= self.baseline_ms:
            reasons.append(
                f"no latency win: {self.ms_per_call:.0f}ms/call vs {self.baseline_ms:.0f}ms"
            )
        return reasons

    @property
    def passed(self) -> bool:
        return not self.failures


def run_probe(
    client: Any,
    model: str,
    cases: tuple[ProbeCase, ...] = CASES,
    baseline_ms: float = BASELINE_MS,
) -> ProbeResult:
    """Score every case with ``model``, timing the calls the way the pipeline makes them."""
    # Load the weights before the clock starts: a cold start is what keep_alive is for,
    # and charging it to the first case would misreport the steady-state cost.
    client.yes_probability(rerank_prompt("warm", "warm"), system=RERANK_SYSTEM, model=model)

    def score(question: str, excerpt: str) -> float:
        return float(
            client.yes_probability(
                rerank_prompt(question, excerpt), system=RERANK_SYSTEM, model=model
            )
        )

    scores: list[CaseScore] = []
    calls = 0
    started = time.perf_counter()

    for case in cases:
        governing = None if case.governing is None else score(case.question, case.governing)
        scores.append(
            CaseScore(
                question=case.question,
                near_miss=score(case.question, case.near_miss),
                governing=governing,
            )
        )
        calls += 1 if case.governing is None else 2

    elapsed_ms = (time.perf_counter() - started) * 1000
    return ProbeResult(
        model=model,
        scores=tuple(scores),
        ms_per_call=elapsed_ms / max(calls, 1),
        baseline_ms=baseline_ms,
    )


def format_result(result: ProbeResult) -> str:
    lines = [f"model: {result.model}", ""]
    for case in result.scores:
        governing = "    -" if case.governing is None else f"{case.governing:.3f}"
        margin = "  (no rule governs)" if case.margin is None else f"  margin {case.margin:+.3f}"
        lines.append(
            f"  rule {governing}   near-miss {case.near_miss:.3f}{margin}   {case.question}"
        )
    lines += [
        "",
        f"worst margin:   {result.margin:.3f}  (need >= {MIN_MARGIN})",
        f"spread:         {result.spread:.3f}  (need >= {MIN_SPREAD})",
        f"unanswerable:   {result.unanswerable_high:.3f}  (need <= {MAX_UNANSWERABLE})",
        f"latency:        {result.ms_per_call:.0f}ms/call  (need < {result.baseline_ms:.0f}ms)",
        "",
        "PASS — safe to promote, subject to the golden-set gate" if result.passed else "FAIL",
    ]
    lines.extend(f"  - {reason}" for reason in result.failures)
    return "\n".join(lines)
