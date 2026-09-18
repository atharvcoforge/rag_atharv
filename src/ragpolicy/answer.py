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
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragpolicy.ingest import Chunk
from ragpolicy.retrieve import ScoredHit

REFUSAL = "The provided policy does not answer this question."

SYSTEM = (
    "You answer questions about an employee expense policy using only the excerpts you "
    "are given.\n\n"
    "Applying a stated rule to a specific case is not adding information. Do it:\n"
    "- A rule about a category settles a question about a member of that category.\n"
    "- A rule with a threshold settles a question about an amount above or below it.\n"
    "- A rule requiring approval settles a question about doing the thing without it.\n\n"
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
    """Find a model-supplied quote verbatim inside the retrieved excerpts.

    A quote that is not in the evidence is a fabrication, and returning ``None`` here is
    what turns that into a refusal. When it is found, its position gives a sentence-level
    citation span rather than a whole-section one.
    """
    needle = quote.strip()
    if not needle:
        return None
    for hit in hits:
        offset = hit.chunk.text.find(needle)
        if offset != -1:
            start = hit.chunk.start + offset
            return hit, start, start + len(needle)
    return None


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
        '2) "governing_rule": copy the one sentence from the excerpts that governs that '
        "subject, word for word. Leave it empty if no sentence does.\n"
        '3) "answer": if governing_rule is non-empty you must answer the question by '
        f'applying that rule, not refuse. If it is empty, answer exactly "{REFUSAL}".\n'
        '4) "section": the section number the rule came from.'
    )


def entailment_prompt(statement: str, evidence: str) -> str:
    return (
        f"Policy excerpt:\n{evidence}\n\n"
        f"Statement: {statement}\n\n"
        "Is the statement fully supported by the excerpt?"
    )


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

    def answer(self, question: str, hits: Sequence[ScoredHit]) -> Response:
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

        raw_output = self.client.generate(
            build_prompt(question, hits), system=SYSTEM, schema=ANSWER_SCHEMA
        )
        parsed = _parse(raw_output)
        if parsed is None:
            return self._refuse(trace, "generation", chunks, started)

        answer_text, section, rule = parsed
        if answer_text.strip() == REFUSAL:
            return self._refuse(trace, "model", chunks, started)

        # The quoted rule is authoritative. It must exist verbatim in the evidence, which
        # makes a fabricated citation structurally impossible, and where it sits gives a
        # sentence-level span instead of a whole-section one.
        located = locate_quote(rule, hits)
        if located is None:
            return self._refuse(trace, "quote", chunks, started)

        cited, quote_start, quote_end = located
        evidence = [Evidence(cited.chunk, quote_start, quote_end, rule.strip())]
        trace["governing_rule"] = rule.strip()

        if not verify_spans(self.raw, evidence):
            return self._refuse(trace, "span", chunks, started)

        if self.verify:
            support = self.client.yes_probability(
                entailment_prompt(answer_text, evidence[0].quote), system=ENTAILMENT_SYSTEM
            )
            trace["entailment"] = support
            trace["confidence"] = min(top_score, support)
            if support < self.thresholds.entailment:
                return self._refuse(trace, "verification", chunks, started)

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
