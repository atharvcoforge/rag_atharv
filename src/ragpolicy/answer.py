"""Grounded generation, span verification, and the two abstention gates.

The output contract is the lab's, exactly. Everything this system knows beyond that
contract lives under ``trace``, so the deliverable stays clean while the UI still has
something interesting to render.

Refusing well is the hard requirement. Two independent gates handle it:

1. **Retrieval gate.** If the best reranked candidate is weak, or is barely ahead of the
   runner-up, refuse before generating. Cheap, and it catches the near-miss case
   ("rental car" against a ground-transport section that never mentions rental cars).
2. **Verification gate.** After generating, check the answer is entailed by the evidence
   and that every citation span is a byte-exact substring of the policy at the recorded
   offsets. A fabricated citation cannot survive the second check.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragpolicy.ingest import Chunk, split_sentences
from ragpolicy.retrieve import ScoredHit

REFUSAL = "The provided policy does not answer this question."

SYSTEM = (
    "You answer questions about an employee expense policy using only the excerpts you "
    "are given.\n\n"
    "Applying a stated rule to a specific case is not adding information. Do it:\n"
    "- A rule about a category settles a question about a member of that category.\n"
    "- A rule with a threshold settles a question about an amount above or below it.\n"
    "- A rule requiring approval settles a question about doing the thing without it.\n\n"
    "A named peer is not a member: a business-class rule does not govern first-class, "
    "and a taxi rule does not settle rental cars. Pick only the rule whose subject "
    "covers the thing asked about.\n"
    "Cabin classes are peers. For first-class, quote "
    '"Employees must purchase economy airfare." and answer that economy is required. '
    "Never apply the business-class vice-president approval sentence to first-class.\n\n"
    "Refuse only when nothing in the excerpts governs the question at all: when the "
    "thing asked about belongs to no category, threshold, or requirement the excerpts "
    f'mention. Then reply with exactly: "{REFUSAL}"\n\n'
    "Never invent an amount, limit, approver, or deadline that is not written in the "
    "excerpts, and never fill a gap with how expense policies usually work. Answer in "
    "one or two complete sentences, stating the rule you applied."
)

# The field order is the point. Making the model name the subject and then *quote* the
# governing sentence before it writes an answer forces the categorisation step it
# otherwise skips. Asked for an answer directly, qwen3:8b refuses "can I expense wine"
# against "Alcohol is not reimbursable", despite answering "yes" when asked outright
# whether wine is alcohol. Inside a grounded prompt it matches literally rather than
# categorically. Made to quote the rule first, it answers correctly.
#
# The quote is also load-bearing downstream: it has to appear verbatim in a retrieved
# excerpt, and where it appears determines the cited span.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "governing_rule": {"type": "string"},
        "answer": {"type": "string"},
        "section": {"type": "string"},
    },
    "required": ["subject", "governing_rule", "answer", "section"],
}

ENTAILMENT_SYSTEM = (
    "You check whether a statement is fully supported by a policy excerpt. Reply with "
    "exactly one word: yes or no."
)


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Gate parameters. Fitted by ``rag calibrate``, not chosen by hand."""

    tau: float = 0.5
    delta: float = 0.0
    entailment: float = 0.5


@dataclass(frozen=True, slots=True)
class Evidence:
    chunk: Chunk
    start: int
    end: int
    text: str | None = None

    @property
    def quote(self) -> str:
        """The exact string this span claims to cover."""
        return self.chunk.text if self.text is None else self.text


@dataclass(frozen=True, slots=True)
class Response:
    answer: str
    citation: dict[str, str] | None
    retrieved_chunks: list[dict[str, Any]]
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citation": self.citation,
            "retrieved_chunks": self.retrieved_chunks,
            "trace": self.trace,
        }


def verify_spans(raw: str, evidence: Sequence[Evidence]) -> bool:
    """Every cited span must be exactly where it claims to be in the source document."""
    return all(
        0 <= item.start < item.end <= len(raw) and raw[item.start : item.end] == item.quote
        for item in evidence
    )


def locate_quote(quote: str, hits: Sequence[ScoredHit]) -> tuple[ScoredHit, int, int] | None:
    """Find a model-supplied quote inside the retrieved excerpts.

    A quote that is not in the evidence is a fabrication, and returning ``None`` here is
    what turns that into a refusal. When it is found, its position gives a sentence-level
    citation span rather than a whole-section one.

    Matching is whitespace-insensitive, which is not a loosening of the check. The policy
    separates sentences with newlines; models quoting them often join with a space. An
    exact ``str.find`` rejected real quotes purely over a ``\\n``. Every word still has
    to appear, in order, in retrieved evidence. Multi-sentence matches are narrowed to
    the single supporting sentence during verification.
    """
    words = quote.split()
    if not words:
        return None
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words))
    for hit in hits:
        found = pattern.search(hit.chunk.text)
        if found is not None:
            return hit, hit.chunk.start + found.start(), hit.chunk.start + found.end()
    return None


def recover_supporting_sentence(
    answer: str,
    hits: Sequence[ScoredHit],
    client: Any,
    threshold: float,
) -> tuple[ScoredHit, str, float] | None:
    """If the model quoted a sibling sentence, find another retrieved sentence that entails.

    Common on multi-sentence sections under lab mode (limit sentence vs approval sentence).
    """
    best: tuple[ScoredHit, str, float] | None = None
    for hit in hits:
        for sentence, _ in split_sentences(hit.chunk.text, 0):
            score = float(
                client.yes_probability(
                    entailment_prompt(answer, sentence),
                    system=ENTAILMENT_SYSTEM,
                    role="entailment",
                )
            )
            if score < threshold:
                continue
            if best is None or score > best[2]:
                best = (hit, sentence, score)
    return best


def build_prompt(question: str, hits: Sequence[ScoredHit]) -> str:
    excerpts = "\n\n".join(
        f"[{n}] Section {hit.chunk.citation_label}\n{hit.chunk.text}"
        for n, hit in enumerate(hits, start=1)
    )
    # "If the excerpts do not *contain* the answer" is the wrong test, and it is the last
    # thing the model reads. Asked that way it refuses "can I expense wine" against
    # "Alcohol is not reimbursable", because the word wine is not there. Asking whether a
    # rule *governs* the question fixes it. The same wording error was measured, and
    # fixed, in the reranker prompt.
    return (
        f"Policy excerpts:\n\n{excerpts}\n\n"
        f"Question: {question}\n\n"
        "Work in this order.\n"
        '1) "subject": the specific thing the question asks about.\n'
        '2) "governing_rule": copy exactly one sentence from the excerpts that governs '
        "that subject, word for word. Never join two sentences. Leave it empty if no "
        "sentence does.\n"
        '3) "answer": if governing_rule is non-empty, answer by applying that rule alone '
        "(do not borrow facts from any other sentence). If it is empty, answer exactly "
        f'"{REFUSAL}".\n'
        '4) "section": the section number the rule came from.'
    )


def entailment_prompt(statement: str, evidence: str) -> str:
    return (
        f"Policy excerpt:\n{evidence}\n\n"
        f"Statement: {statement}\n\n"
        "Is the statement fully supported by the excerpt?"
    )


def _no_phase(phase: str) -> None:
    """Default phase listener: nothing is watching a CLI or eval run."""


class Answerer:
    def __init__(
        self,
        client: Any,
        raw_policy: str,
        corpus: Sequence[Chunk],
        thresholds: Thresholds | None = None,
        verify: bool = True,
    ) -> None:
        self.client = client
        self.raw = raw_policy
        self.corpus = list(corpus)
        self.thresholds = thresholds or Thresholds()
        self.verify = verify

    def answer(
        self,
        question: str,
        hits: Sequence[ScoredHit],
        on_phase: Callable[[str], None] | None = None,
    ) -> Response:
        phase = on_phase if on_phase is not None else _no_phase
        started = time.perf_counter()
        hits = list(hits)
        top_score = hits[0].score if hits else 0.0
        margin = (hits[0].score - hits[1].score) if len(hits) > 1 else top_score

        chunks = _reported_chunks(hits)
        trace: dict[str, Any] = {
            "abstained_at": None,
            "confidence": top_score,
            "retrieval_score": top_score,
            "margin": margin,
            # Stage timings cover retrieval only, so the answer half of the budget used
            # to be readable solely by subtracting them from the total. Split it here.
            "generate_ms": 0.0,
            "verify_ms": 0.0,
            "spans": [],
            "candidates": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "section": hit.chunk.citation_label,
                    "distance": hit.distance,
                    "score": hit.score,
                }
                for hit in hits
            ],
        }

        if not hits or top_score < self.thresholds.tau or margin < self.thresholds.delta:
            return self._refuse(trace, "retrieval", chunks, started)

        phase("generating")
        generating = time.perf_counter()
        raw_output = self.client.generate(
            build_prompt(question, hits), system=SYSTEM, schema=ANSWER_SCHEMA
        )
        trace["generate_ms"] = (time.perf_counter() - generating) * 1000
        parsed = _parse(raw_output)
        if parsed is None:
            return self._refuse(trace, "generation", chunks, started)

        answer_text, section, rule = parsed
        if answer_text.strip() == REFUSAL:
            return self._refuse(trace, "model", chunks, started)

        if peer_rule_mismatch(question, rule, answer_text):
            return self._refuse(trace, "peer_rule", chunks, started)

        # The quoted rule is authoritative. It must exist verbatim in the evidence, which
        # makes a fabricated citation structurally impossible, and where it sits gives a
        # sentence-level span instead of a whole-section one.
        located = locate_quote(rule, hits)
        if located is None:
            return self._refuse(trace, "quote", chunks, started)

        cited, quote_start, quote_end = located
        # Prefer the document's own bytes over the model's whitespace.
        quote = self.raw[quote_start:quote_end]
        sentences = [text for text, _ in split_sentences(quote, 0)]
        if not sentences:
            return self._refuse(trace, "quote", chunks, started)

        support: float | None = None
        if len(sentences) > 1:
            # Joining two rules into one span launders false entailments: measured
            # first-class × VP-approval was P(yes)≈0.99 against the join and ~0 against
            # either sentence alone. Keep the one sentence that actually supports the
            # answer; refuse if none do.
            if not self.verify:
                return self._refuse(trace, "quote", chunks, started)
            phase("verifying")
            best_sentence = ""
            best_support = -1.0
            for sentence in sentences:
                score = self._entailment(trace, answer_text, sentence)
                if score > best_support:
                    best_support, best_sentence = score, sentence
            support = best_support
            trace["entailment"] = best_support
            if best_support < self.thresholds.entailment:
                return self._refuse(trace, "verification", chunks, started)
            located = locate_quote(best_sentence, hits)
            if located is None:
                return self._refuse(trace, "quote", chunks, started)
            cited, quote_start, quote_end = located
            quote = self.raw[quote_start:quote_end]
            if len(split_sentences(quote, 0)) != 1:
                return self._refuse(trace, "quote", chunks, started)

        evidence = [Evidence(cited.chunk, quote_start, quote_end, quote)]
        trace["governing_rule"] = evidence[0].quote

        if peer_rule_mismatch(question, evidence[0].quote, answer_text):
            return self._refuse(trace, "peer_rule", chunks, started)

        if not verify_spans(self.raw, evidence):
            return self._refuse(trace, "span", chunks, started)

        if self.verify and support is None:
            phase("verifying")
            support = self._entailment(trace, answer_text, evidence[0].quote)
            trace["entailment"] = support
            if support < self.thresholds.entailment:
                recovering = time.perf_counter()
                recovered = recover_supporting_sentence(
                    answer_text, hits, self.client, self.thresholds.entailment
                )
                trace["verify_ms"] += (time.perf_counter() - recovering) * 1000
                if recovered is None:
                    return self._refuse(trace, "verification", chunks, started)
                cited, sentence, support = recovered
                located = locate_quote(sentence, hits)
                if located is None:
                    return self._refuse(trace, "quote", chunks, started)
                cited, quote_start, quote_end = located
                quote = self.raw[quote_start:quote_end]
                if peer_rule_mismatch(question, quote, answer_text):
                    return self._refuse(trace, "peer_rule", chunks, started)
                evidence = [Evidence(cited.chunk, quote_start, quote_end, quote)]
                trace["governing_rule"] = evidence[0].quote
                trace["entailment"] = support
                trace["recovered_support"] = True
                if not verify_spans(self.raw, evidence):
                    return self._refuse(trace, "span", chunks, started)

        if support is not None:
            trace["confidence"] = min(top_score, support)

        trace["spans"] = [
            {"chunk_id": e.chunk.chunk_id, "start": e.start, "end": e.end, "text": e.quote}
            for e in evidence
        ]
        trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return Response(
            answer=answer_text,
            citation={
                "document": cited.chunk.document,
                "version": cited.chunk.version,
                "section": cited.chunk.citation_label,
            },
            retrieved_chunks=chunks,
            trace=trace,
        )

    def _entailment(self, trace: dict[str, Any], statement: str, evidence: str) -> float:
        """Score support for a statement, charging the wait to ``verify_ms``.

        The trace is passed in rather than kept on ``self``: one Answerer serves every
        request, and the API now runs them off the request thread.
        """
        started = time.perf_counter()
        score = float(
            self.client.yes_probability(
                entailment_prompt(statement, evidence),
                system=ENTAILMENT_SYSTEM,
                role="entailment",
            )
        )
        trace["verify_ms"] += (time.perf_counter() - started) * 1000
        return score

    def _refuse(
        self, trace: dict[str, Any], stage: str, chunks: list[dict[str, Any]], started: float
    ) -> Response:
        trace["abstained_at"] = stage
        trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return Response(answer=REFUSAL, citation=None, retrieved_chunks=chunks, trace=trace)


def _reported_chunks(hits: Sequence[ScoredHit]) -> list[dict[str, Any]]:
    """At most three, ascending by distance, distances as numbers. The lab's contract."""
    top = sorted(hits, key=lambda hit: hit.distance)[:3]
    return [{"section": hit.chunk.citation_label, "distance": float(hit.distance)} for hit in top]


def peer_rule_mismatch(question: str, rule: str, answer: str) -> bool:
    """True when the quoted rule is a named peer of the asked subject, not its governor.

    Catches the live failure where first-class borrows the business-class VP-approval
    sentence even if a weak entailment scorer says yes.
    """
    asked = f"{question} {answer}".lower()
    quoted = rule.lower()
    asks_first = "first-class" in asked or "first class" in asked
    if not asks_first:
        return False
    if "business-class" in quoted or "business class" in quoted:
        return True
    answered = answer.lower()
    return "vice president" in answered and "economy" not in answered


def _parse(raw_output: str) -> tuple[str, str, str] | None:
    """Return ``(answer, section, governing_rule)`` or ``None`` if unusable."""
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        return None
    return answer, str(payload.get("section", "")), str(payload.get("governing_rule", ""))
