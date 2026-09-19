"""End-to-end wiring: index the policy, then answer questions against it.

Every retrieval stage is switchable from :class:`RetrievalConfig`. That is not
configurability for its own sake; it is what lets the eval harness run an ablation and
decide which stages have earned their latency.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragpolicy.answer import Answerer, Response, Thresholds
from ragpolicy.config import Settings
from ragpolicy.ingest import Chunk, build_corpus, contextual_text
from ragpolicy.models import OllamaClient
from ragpolicy.response_cache import AnswerCache
from ragpolicy.retrieve import (
    BM25,
    RERANK_SYSTEM,
    ScoredHit,
    expand_to_sections,
    reciprocal_rank_fusion,
    rerank,
    rerank_prompt,
)
from ragpolicy.store import Hit, NumpyStore, VectorStore, open_store


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Which stages run. Named configurations live in :data:`ABLATIONS`."""

    dense: bool = True
    bm25: bool = True
    rerank: bool = True
    verify: bool = True
    candidates: int = 12
    # Reranking costs ~153ms per candidate and cannot be parallelised, so this number is
    # the single biggest lever on latency. Six covers every golden-set question whose
    # answer survives fusion at all.
    rerank_candidates: int = 6
    final_sections: int = 3
    # Lab contract adapter: six bare section vectors, dense cosine LIMIT 3, no hybrid.
    lab: bool = False


ABLATIONS: dict[str, RetrievalConfig] = {
    "dense": RetrievalConfig(bm25=False, rerank=False, verify=False),
    "dense+bm25": RetrievalConfig(rerank=False, verify=False),
    "dense+bm25+rerank": RetrievalConfig(verify=False),
    "full": RetrievalConfig(),
    # Isolates BM25's contribution with rerank and verification switched on. Without
    # this row the sweep cannot say whether the lexical half is carrying its weight.
    "full-no-bm25": RetrievalConfig(bm25=False),
    "bm25-only": RetrievalConfig(dense=False, rerank=False, verify=False),
    # Assignment contract path. Does not replace ``full``; use ``--config lab``.
    "lab": RetrievalConfig(
        lab=True,
        bm25=False,
        rerank=False,
        verify=True,
        candidates=3,
        rerank_candidates=3,
        final_sections=3,
    ),
}


@dataclass
class Stage:
    name: str
    ms: float
    detail: dict[str, Any] = field(default_factory=dict)


StageListener = Callable[[Stage], None]
#: ``(event name, payload)`` — the SSE endpoint's two event types, ``stage`` and ``status``.
EventListener = Callable[[str, dict[str, Any]], None]


def _silent(name: str, data: dict[str, Any]) -> None:
    """Default listener: the CLI and the eval harness do not watch progress."""


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
            rerank_base_url=settings.rerank_ollama_url or settings.ollama_base_url,
            keep_alive=settings.keep_alive,
        )
        self.store = (
            store if store is not None else open_store(settings.postgres_dsn, settings.vector_store)
        )
        self._lab_store: NumpyStore | None = None
        self.bm25 = BM25([c for c in self.corpus if c.kind == "proposition"])
        self.answerer = Answerer(
            client=self.client,
            raw_policy=self.raw,
            corpus=self.corpus,
            thresholds=thresholds,
            verify=self.config.verify,
        )
        self._answers: AnswerCache | None = (
            AnswerCache(settings.answer_cache_path) if settings.answer_cache else None
        )

    @property
    def sections(self) -> list[Chunk]:
        return [c for c in self.corpus if c.kind == "section"]

    # -- indexing ------------------------------------------------------------------

    def index(self) -> int:
        """Embed and store the corpus. Lab mode writes the six bare section chunks only."""
        if self.config.lab:
            return self._index_lab(self.store)

        texts = [contextual_text(chunk) for chunk in self.corpus]
        vectors = self.client.embed_documents(texts)
        self.store.create(dim=int(vectors.shape[1]))
        self.store.upsert(self.corpus, vectors)
        if self._answers is not None:
            # Old answers cite spans into the previous policy bytes.
            self._answers.clear()
        return len(self.corpus)

    def ensure_lab_store(self) -> NumpyStore:
        """Ephemeral six-section store for ``--config lab`` ask/eval without wiping full."""
        if self._lab_store is not None:
            return self._lab_store
        store = NumpyStore()
        self._index_lab(store)
        self._lab_store = store
        return store

    def _index_lab(self, store: VectorStore) -> int:
        sections = [c for c in self.corpus if c.kind == "section"]
        texts = [chunk.text for chunk in sections]
        vectors = self.client.embed_documents(texts)
        store.create(dim=int(vectors.shape[1]))
        store.upsert(sections, vectors)
        if isinstance(store, NumpyStore):
            self._lab_store = store
        return len(sections)

    # -- querying ------------------------------------------------------------------

    def retrieve(
        self, question: str, on_stage: StageListener | None = None
    ) -> tuple[list[ScoredHit], list[Stage]]:
        if self.config.lab:
            return self._retrieve_lab(question, on_stage)

        stages: list[Stage] = []

        def record(name: str, started: float, **detail: Any) -> None:
            """Close a stage and publish it now: the stream should not wait for the answer."""
            stage = Stage(name, (time.perf_counter() - started) * 1000, detail)
            stages.append(stage)
            if on_stage is not None:
                on_stage(stage)

        propositions = {c.chunk_id: c for c in self.corpus if c.kind == "proposition"}
        rankings: list[list[str]] = []
        distances: dict[str, float] = {}

        if self.config.dense:
            started = time.perf_counter()
            vector = self.client.embed_query(question)
            dense_hits = [
                hit
                for hit in self.store.search(vector, limit=self.config.candidates * 2)
                if hit.chunk.kind == "proposition"
            ][: self.config.candidates]
            for hit in dense_hits:
                distances[hit.chunk.chunk_id] = hit.distance
            rankings.append([hit.chunk.chunk_id for hit in dense_hits])
            record("dense", started, hits=len(dense_hits))

        if self.config.bm25:
            started = time.perf_counter()
            lexical = self.bm25.search(question, limit=self.config.candidates)
            rankings.append([chunk_id for chunk_id, _ in lexical])
            record("bm25", started, hits=len(lexical))

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(rankings)
        candidates = [
            Hit(chunk=propositions[cid], distance=distances.get(cid, 1.0))
            for cid, _ in fused
            if cid in propositions
        ][: self.config.rerank_candidates]
        record("fuse", started, candidates=len(candidates))

        scored: list[ScoredHit]
        if self.config.rerank:
            started = time.perf_counter()
            scored = rerank(question, candidates, scorer=self._relevance)
            record("rerank", started, scored=len(scored))
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
        record("expand", started, sections=len(expanded))
        return expanded, stages

    def _retrieve_lab(
        self, question: str, on_stage: StageListener | None = None
    ) -> tuple[list[ScoredHit], list[Stage]]:
        """Dense cosine over the six bare section vectors, ascending distance, LIMIT 3."""
        started = time.perf_counter()
        store = self.ensure_lab_store()
        vector = self.client.embed_query(question)
        hits = store.search(vector, limit=self.config.final_sections)
        scored = [
            ScoredHit(
                chunk=hit.chunk,
                distance=hit.distance,
                score=max(0.0, 1.0 - hit.distance),
            )
            for hit in hits
        ]
        stage = Stage("dense", (time.perf_counter() - started) * 1000, {"hits": len(scored)})
        if on_stage is not None:
            on_stage(stage)
        return scored, [stage]

    def ask(self, question: str, on_event: EventListener | None = None) -> Response:
        """Answer one question, reporting progress to ``on_event`` as each step lands.

        The listener is how the SSE endpoint shows retrieval finishing in milliseconds
        while generation and verification are still running. An exact cache hit skips
        the models entirely and announces ``cached`` instead.
        """
        started = time.perf_counter()
        emit = on_event if on_event is not None else _silent

        if self._answers is not None:
            cached = self._answers.get(question, self.config)
            if cached is not None:
                emit("status", {"phase": "cached"})
                response = Response(
                    answer=cached["answer"],
                    citation=cached["citation"],
                    retrieved_chunks=cached["retrieved_chunks"],
                    trace=dict(cached.get("trace") or {}),
                )
                response.trace["cache_hit"] = True
                response.trace["total_ms"] = (time.perf_counter() - started) * 1000
                return response

        hits, stages = self.retrieve(
            question, on_stage=lambda stage: emit("stage", {"name": stage.name, "ms": stage.ms})
        )
        response = self.answerer.answer(
            question, hits, on_phase=lambda phase: emit("status", {"phase": phase})
        )
        response.trace["stages"] = [{"name": s.name, "ms": s.ms, **s.detail} for s in stages]
        response.trace["total_ms"] = (time.perf_counter() - started) * 1000
        response.trace["cache_hit"] = False
        if self._answers is not None:
            self._answers.put(question, self.config, response.to_dict())
        return response

    # -- model helpers -------------------------------------------------------------

    def _relevance(self, question: str, excerpt: str) -> float:
        return float(
            self.client.yes_probability(rerank_prompt(question, excerpt), system=RERANK_SYSTEM)
        )


def build(settings: Settings | None = None, **kwargs: Any) -> Pipeline:
    return Pipeline(settings or Settings.from_env(), **kwargs)
