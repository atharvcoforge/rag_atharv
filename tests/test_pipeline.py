"""Full (non-lab) pipeline: index, hybrid retrieve, ask, cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from ragpolicy.answer import Thresholds
from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.ingest import build_corpus
from ragpolicy.pipeline import ABLATIONS, Pipeline, RetrievalConfig, build
from ragpolicy.store import NumpyStore

RAW = (REPO_ROOT / "policy.md").read_text(encoding="utf-8")
CORPUS = build_corpus(RAW)


class StubClient:
    """Deterministic embeddings + scripted generation for the full retrieval path."""

    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload or {
            "answer": "Employees may claim up to $65 per day for meals.",
            "section": "1",
            "governing_rule": (
                "Employees may claim up to $65 per day for meals while traveling overnight."
            ),
        }
        self.embed_calls = 0

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self.embed_calls += 1
        dim = 8
        rows = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            # Stable hash into a unit basis so cosine search is deterministic.
            idx = abs(hash(text)) % dim
            vec[idx] = 1.0
            # Prefer meals propositions for meal-ish queries by giving section-1 a fixed slot.
            if "Meals" in text or "meals" in text.lower() or "$65" in text:
                vec = np.zeros(dim, dtype=np.float32)
                vec[0] = 1.0
            rows.append(vec)
        return np.vstack(rows)

    def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        vec = np.zeros(8, dtype=np.float32)
        if any(w in query.lower() for w in ("meal", "food", "wine", "alcohol", "$65")):
            vec[0] = 1.0
        else:
            vec[abs(hash(query)) % 8] = 1.0
        return vec

    def yes_probability(self, prompt: str, **kwargs: object) -> float:
        # Prefer meals evidence for meal questions during rerank/entailment.
        if "meals" in prompt.lower() or "$65" in prompt.lower() or "alcohol" in prompt.lower():
            return 0.95
        return 0.2

    def generate(self, prompt: str, **kwargs: object) -> str:
        return json.dumps(self.payload)

    def close(self) -> None:
        return None


def _settings(tmp_path: Path, *, answer_cache: bool = True) -> Settings:
    return Settings(
        ollama_base_url="http://localhost:0",
        embed_model="stub",
        rerank_model="stub",
        gen_model="stub",
        vector_store="numpy",
        postgres_dsn="postgresql://unused",
        policy_path=REPO_ROOT / "policy.md",
        embed_cache_path=tmp_path / "emb.sqlite",
        answer_cache_path=tmp_path / "ans.sqlite",
        cassette_dir=tmp_path / "cassettes",
        cassette_mode="off",
        answer_cache=answer_cache,
    )


def _pipeline(tmp_path: Path, config: RetrievalConfig | None = None) -> Pipeline:
    client = StubClient()
    store = NumpyStore()
    pipeline = Pipeline(
        _settings(tmp_path),
        config=config or RetrievalConfig(),
        thresholds=Thresholds(tau=0.0, delta=0.0, entailment=0.5),
        client=client,  # type: ignore[arg-type]
        store=store,
    )
    pipeline.index()
    return pipeline


def test_sections_are_the_six_policy_sections(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert len(pipeline.sections) == 6
    assert {s.section for s in pipeline.sections} == {"1", "2", "3", "4", "5", "6"}


def test_full_index_stores_corpus_and_clears_answer_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = StubClient()
    store = NumpyStore()
    pipeline = Pipeline(
        settings,
        client=client,  # type: ignore[arg-type]
        store=store,
        thresholds=Thresholds(tau=0.0),
    )
    assert pipeline._answers is not None
    pipeline._answers.put("old?", RetrievalConfig(), {"answer": "stale"})
    count = pipeline.index()
    assert count == len(CORPUS)
    assert store.count() == len(CORPUS)
    assert pipeline._answers.get("old?", RetrievalConfig()) is None


def test_full_retrieve_runs_dense_bm25_fuse_rerank_expand(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    seen: list[str] = []
    hits, stages = pipeline.retrieve(
        "How much can I spend on meals?",
        on_stage=lambda stage: seen.append(stage.name),
    )
    assert [s.name for s in stages] == ["dense", "bm25", "fuse", "rerank", "expand"]
    assert seen == [s.name for s in stages]
    assert 1 <= len(hits) <= 3
    assert hits == sorted(hits, key=lambda h: h.distance)
    assert all(h.chunk.kind == "section" for h in hits)


def test_retrieve_without_rerank_uses_fused_rank_scores(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, config=ABLATIONS["dense+bm25"])
    hits, stages = pipeline.retrieve("Is alcohol reimbursable?")
    assert not any(s.name == "rerank" for s in stages)
    assert hits
    assert hits[0].score >= hits[-1].score


def test_retrieve_dense_only(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, config=ABLATIONS["dense"])
    hits, stages = pipeline.retrieve("meal allowance overnight")
    assert [s.name for s in stages] == ["dense", "fuse", "expand"]
    assert hits


def test_bm25_only_retrieve(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, config=ABLATIONS["bm25-only"])
    hits, stages = pipeline.retrieve("receipt required twenty five dollars")
    assert any(s.name == "bm25" for s in stages)
    assert not any(s.name == "dense" for s in stages)
    assert hits


def test_ask_attaches_stages_and_uses_answer_cache(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    events: list[tuple[str, dict[str, object]]] = []
    first = pipeline.ask(
        "Can I expense wine with dinner?",
        on_event=lambda name, data: events.append((name, data)),
    )
    assert first.trace["cache_hit"] is False
    assert any(e[0] == "stage" for e in events)
    assert "stages" in first.trace

    events.clear()
    second = pipeline.ask(
        "Can I expense wine with dinner?",
        on_event=lambda name, data: events.append((name, data)),
    )
    assert second.trace["cache_hit"] is True
    assert events and events[0][0] == "status" and events[0][1]["phase"] == "cached"


def test_ask_skips_cache_when_disabled(tmp_path: Path) -> None:
    client = StubClient()
    pipeline = Pipeline(
        _settings(tmp_path, answer_cache=False),
        thresholds=Thresholds(tau=0.0),
        client=client,  # type: ignore[arg-type]
        store=NumpyStore(),
    )
    pipeline.index()
    assert pipeline._answers is None
    response = pipeline.ask("Is alcohol reimbursable?")
    assert response.trace["cache_hit"] is False


def test_relevance_delegates_to_yes_probability(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert pipeline._relevance("meals", "Employees may claim up to $65") >= 0.9


def test_build_constructs_a_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ragpolicy.pipeline.Settings.from_env",
        staticmethod(lambda: _settings(tmp_path, answer_cache=False)),
    )
    monkeypatch.setattr("ragpolicy.pipeline.open_store", lambda dsn, backend: NumpyStore())
    monkeypatch.setattr(
        "ragpolicy.pipeline.OllamaClient",
        lambda **kwargs: StubClient(),
    )
    pipeline = build()
    assert isinstance(pipeline, Pipeline)
    assert pipeline.sections
