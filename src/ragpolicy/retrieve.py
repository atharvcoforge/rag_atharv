"""Lexical search, rank fusion, cross-encoder reranking, and parent expansion.

Dense retrieval lives in :mod:`ragpolicy.store`. This module supplies everything that
sits around it.

BM25 is written out rather than pulled in as a dependency. It is about thirty lines, it
is exactly testable, and it is genuinely BM25 rather than Postgres ``ts_rank_cd``'s
approximation. It exists because embeddings compress exact numerals badly: "$25" and
"30 days" are the queries dense retrieval loses, and lexical matching wins them outright.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ragpolicy.ingest import Chunk, contextual_text
from ragpolicy.store import Hit

_TOKEN = re.compile(r"[a-z0-9$][a-z0-9$.]*")

# On a ten-document corpus, IDF does not suppress function words: "is" occurs in three
# chunks, which leaves it a healthy IDF and lets "is booze covered" match on "is" alone.
# That noise would propagate straight into the fused ranking.
_STOPWORDS = frozenset(
    """a an and are as at be been by can could do does for from had has have how i if in
    is it may must my of on or our shall should that the their there these this to up was
    we were what when where which while who will with would you your""".split()
)

# Framing matters more than the model here. Measured over six positive and five negative
# pairs, asking "does the excerpt contain the answer" leaves the positive minimum at
# 0.0002 against a negative maximum of 0.0003, which is no separation at all: the model
# reads "$20 lunch" against a "$25 or more" rule and says no because $20 is not written
# down. Asking which excerpt *governs* the question, and saying outright that a general
# threshold governs the specific amounts under it, moves the positive minimum to 0.1245
# against a negative maximum of 0.0320. Same model, same candidates, usable margin.
RERANK_SYSTEM = (
    "You decide whether a policy excerpt is the rule that governs a question. A general "
    "threshold or limit governs any specific amount it covers. An excerpt that never "
    "mentions the thing asked about does not govern it. Reply with exactly one word: "
    "yes or no."
)


def stem(word: str) -> str:
    """Fold plurals and common verb endings onto a shared form.

    The policy writes "Hotels", "Receipts" and "submitted"; people type "hotel",
    "receipt" and "submitting". Without this, the lexical half misses its best cases.

    Correctness here means *consistency*, not linguistics: index and query run through
    the same function, so "expense" and "expenses" both landing on "expens" is a hit, not
    a bug. ponytail: a real Snowball stemmer is the upgrade if the corpus ever stops
    being hand-written policy English.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith(("ss", "us")):
        word = word[:-1]
    if word.endswith("ing") and len(word) >= 6:
        word = word[:-3]
    elif word.endswith("ed") and len(word) >= 5:
        word = word[:-2]
    if word.endswith("e") and len(word) >= 6:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, stopword-filtered, stemmed tokens.

    Hyphens split because the policy writes "Business-class" and "public-transit" while
    people type "business class" and "public transit". ``$25`` survives intact.
    """
    tokens = (token.strip(".") for token in _TOKEN.findall(text.lower().replace("-", " ")))
    return [stem(token) for token in tokens if token and token not in _STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed corpus.

    ponytail: scores every document on every query, which is O(corpus). Fine for tens of
    chunks; past ~100k move the lexical half into Postgres full-text or ParadeDB.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        # Index the same contextual string the embedder sees. Both halves of the hybrid
        # then search identical surface, and section titles ("Submission Deadline")
        # become searchable even though no body sentence contains the word.
        self._docs = [Counter(tokenize(contextual_text(chunk))) for chunk in self.chunks]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._docs else 0.0

        document_frequency = Counter(term for doc in self._docs for term in doc)
        total = len(self._docs)
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` for documents sharing at least one query term."""
        terms = tokenize(query)
        scored: list[tuple[str, float]] = []

        for index, doc in enumerate(self._docs):
            score = 0.0
            for term in terms:
                frequency = doc.get(term, 0)
                if frequency == 0:
                    continue
                norm = 1 - self.b + self.b * (self._lengths[index] / (self._avg_length or 1))
                score += (
                    self._idf[term] * (frequency * (self.k1 + 1)) / (frequency + self.k1 * norm)
                )
            if score > 0:
                scored.append((self.chunks[index].chunk_id, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse rankings by ``sum(1 / (k + rank))``.

    Ranks only, never scores. Cosine distance and BM25 live on incomparable scales, and
    every attempt to normalise them into each other needs a constant nobody can justify.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


@dataclass(frozen=True, slots=True)
class ScoredHit:
    chunk: Chunk
    distance: float
    score: float


Scorer = Callable[[str, str], float]


def rerank(query: str, hits: Sequence[Hit], scorer: Scorer) -> list[ScoredHit]:
    """Score every candidate against the query with a cross-encoder, then sort.

    Scored serially on purpose. A thread pool was tried and removed: Ollama serialises
    requests to a single model, so eight candidates took 1190ms sequentially and 1288ms
    across four threads. The pool bought nothing but contention.
    """
    scored = [
        ScoredHit(chunk=hit.chunk, distance=hit.distance, score=scorer(query, hit.chunk.text))
        for hit in hits
    ]
    scored.sort(key=lambda hit: (-hit.score, hit.distance, hit.chunk.chunk_id))
    return scored


def rerank_prompt(query: str, excerpt: str) -> str:
    return f"Excerpt: {excerpt}\nQuestion: {query}\nDoes this excerpt govern the question?"


def expand_to_sections(hits: Sequence[Hit], corpus: Sequence[Chunk]) -> list[Chunk]:
    """Map proposition hits up to their parent sections, deduped, best rank first.

    Retrieval happens over propositions because they discriminate; citation happens over
    sections because that is what the policy reader recognises.
    """
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    sections: list[Chunk] = []
    seen: set[str] = set()

    for hit in hits:
        section_id = hit.chunk.parent_chunk_id or hit.chunk.chunk_id
        if section_id in seen:
            continue
        seen.add(section_id)
        sections.append(by_id.get(section_id, hit.chunk))
    return sections
