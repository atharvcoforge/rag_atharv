"""End-to-end wiring: index the policy, then answer questions against it.

Every retrieval stage is switchable from :class:`RetrievalConfig`. That is not
configurability for its own sake; it is what lets the eval harness run an ablation and
decide which stages have earned their latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragpolicy.answer import Answerer, Response, Thresholds
from ragpolicy.config import Settings
from ragpolicy.ingest import build_corpus, contextual_text
from ragpolicy.models import OllamaClient
from ragpolicy.retrieve import (
    BM25,
    RERANK_SYSTEM,
    ScoredHit,
    expand_to_sections,
    reciprocal_rank_fusion,
    rerank,
    rerank_prompt,
)
from ragpolicy.store import Hit, VectorStore, open_store

HYDE_SYSTEM = (
    "You write a single sentence in the style of a corporate expense policy that would "
    "answer the user's question. Invent plausible specifics. Output the sentence only."
)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Which stages run. Named configurations live in :data:`ABLATIONS`."""

    dense: bool = True
    bm25: bool = True
    rerank: bool = True
    hyde: bool = False
    verify: bool = True
    candidates: int = 12
    # Reranking costs ~153ms per candidate and cannot be parallelised, so this number is
    # the single biggest lever on latency. Six covers every golden-set question whose
    # answer survives fusion at all.
    rerank_candidates: int = 6
    final_sections: int = 3


ABLATIONS: dict[str, RetrievalConfig] = {
    "dense": RetrievalConfig(bm25=False, rerank=False, verify=False),
    "dense+bm25": RetrievalConfig(rerank=False, verify=False),
    "dense+bm25+rerank": RetrievalConfig(verify=False),
    "full": RetrievalConfig(),
    # Isolates BM25's contribution with rerank and verification switched on. Without
    # this row the sweep cannot say whether the lexical half is carrying its weight.
    "full-no-bm25": RetrievalConfig(bm25=False),
    "full+hyde": RetrievalConfig(hyde=True),
    "bm25-only": RetrievalConfig(dense=False, rerank=False, verify=False),
}


@dataclass
class Stage:
    name: str
    ms: float
    detail: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        config: RetrievalConfig | None = None,
        thresholds: Thresholds | None = None,
        client: OllamaClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.config = config or RetrievalConfig()
        self.raw = Path(settings.policy_path).read_text(encoding="utf-8")
        self.corpus = build_corpus(self.raw)
        self.client = client or OllamaClient(
            base_url=settings.ollama_base_url,
            embed_model=settings.embed_model,
            rerank_model=settings.rerank_model,
            gen_model=settings.gen_model,
            cache_path=settings.embed_cache_path,
        )
        self.store = (
            store if store is not None else open_store(settings.postgres_dsn, settings.vector_store)
        )
        self.bm25 = BM25([c for c in self.corpus if c.kind == "proposition"])
        self.answerer = Answerer(
            client=self.client,
            raw_policy=self.raw,
            corpus=self.corpus,
            thresholds=thresholds,
            verify=self.config.verify,
        )

    # -- indexing ------------------------------------------------------------------

    def index(self) -> int:
        """Embed and store the whole corpus. Cached embeddings make a rebuild near-free."""
        texts = [contextual_text(chunk) for chunk in self.corpus]
        vectors = self.client.embed_documents(texts)
        self.store.create(dim=int(vectors.shape[1]))
        self.store.upsert(self.corpus, vectors)
        return len(self.corpus)

    # -- querying ------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[list[ScoredHit], list[Stage]]:
        stages: list[Stage] = []
        propositions = {c.chunk_id: c for c in self.corpus if c.kind == "proposition"}
        rankings: list[list[str]] = []
        distances: dict[str, float] = {}

        if self.config.dense:
            started = time.perf_counter()
            query = question
            if self.config.hyde:
                query = f"{question}\n{self._hyde(question)}"
            vector = self.client.embed_query(query)
            dense_hits = [
                hit
                for hit in self.store.search(vector, limit=self.config.candidates * 2)
                if hit.chunk.kind == "proposition"
            ][: self.config.candidates]
            for hit in dense_hits:
                distances[hit.chunk.chunk_id] = hit.distance
            rankings.append([hit.chunk.chunk_id for hit in dense_hits])
            stages.append(
                Stage("dense", (time.perf_counter() - started) * 1000, {"hits": len(dense_hits)})
            )

        if self.config.bm25:
            started = time.perf_counter()
            lexical = self.bm25.search(question, limit=self.config.candidates)
            rankings.append([chunk_id for chunk_id, _ in lexical])
            stages.append(
                Stage("bm25", (time.perf_counter() - started) * 1000, {"hits": len(lexical)})
            )

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(rankings)
        candidates = [
            Hit(chunk=propositions[cid], distance=distances.get(cid, 1.0))
            for cid, _ in fused
            if cid in propositions
        ][: self.config.rerank_candidates]
        stages.append(
            Stage("fuse", (time.perf_counter() - started) * 1000, {"candidates": len(candidates)})
        )

        scored: list[ScoredHit]
        if self.config.rerank:
            started = time.perf_counter()
            scored = rerank(question, candidates, scorer=self._relevance)
            stages.append(
                Stage("rerank", (time.perf_counter() - started) * 1000, {"scored": len(scored)})
            )
        else:
            # Without a reranker, fused rank order is the score. Normalised so the
            # abstention gate sees a comparable 0-1 scale either way.
            total = max(len(candidates), 1)
            scored = [
                ScoredHit(chunk=hit.chunk, distance=hit.distance, score=1.0 - i / total)
                for i, hit in enumerate(candidates)
            ]

        started = time.perf_counter()
        sections = expand_to_sections(
            [Hit(chunk=s.chunk, distance=s.distance) for s in scored], self.corpus
        )
        best: dict[str, ScoredHit] = {}
        for ranked in scored:
            parent = ranked.chunk.parent_chunk_id or ranked.chunk.chunk_id
            if parent not in best:
                best[parent] = ranked
        expanded = [
            ScoredHit(
                chunk=section,
                distance=best[section.chunk_id].distance,
                score=best[section.chunk_id].score,
            )
            for section in sections
            if section.chunk_id in best
        ][: self.config.final_sections]
        stages.append(
            Stage("expand", (time.perf_counter() - started) * 1000, {"sections": len(expanded)})
        )
        return expanded, stages

    def ask(self, question: str) -> Response:
        started = time.perf_counter()
        hits, stages = self.retrieve(question)
        response = self.answerer.answer(question, hits)
        response.trace["stages"] = [{"name": s.name, "ms": s.ms, **s.detail} for s in stages]
        response.trace["total_ms"] = (time.perf_counter() - started) * 1000
        return response

    # -- model helpers -------------------------------------------------------------

    def _relevance(self, question: str, excerpt: str) -> float:
        return float(
            self.client.yes_probability(rerank_prompt(question, excerpt), system=RERANK_SYSTEM)
        )

    def _hyde(self, question: str) -> str:
        """Hypothetical Document Embeddings: search with an imagined answer.

        A fabricated policy sentence sits closer in embedding space to the real policy
        sentence than a question does, because questions and statements are written
        differently. Whether that helps *here* is what the ablation decides.
        """
        return self.client.generate(question, system=HYDE_SYSTEM, num_predict=64).strip()


def build(settings: Settings | None = None, **kwargs: Any) -> Pipeline:
    return Pipeline(settings or Settings.from_env(), **kwargs)
