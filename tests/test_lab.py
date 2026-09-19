"""Lab-contract adapter: six bare section chunks, dense cosine only, LIMIT 3."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.ingest import parse_sections
from ragpolicy.pipeline import ABLATIONS, Pipeline, RetrievalConfig
from ragpolicy.store import NumpyStore

RAW = (REPO_ROOT / "policy.md").read_text(encoding="utf-8")
SECTIONS = parse_sections(RAW)


class StubEmbedClient:
    """Deterministic unit vectors keyed by section number so cosine ranks are stable."""

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self.embed_calls.append(list(texts))
        rows = []
        for text in texts:
            # One-hot-ish: meals→e0, hotels→e1, ... so queries can pick a section.
            idx = 0
            for i, section in enumerate(SECTIONS):
                if section.text in text or text in section.text or text.strip() == section.text:
                    idx = i
                    break
            # Bare section bodies match exactly on index.
            for i, section in enumerate(SECTIONS):
                if text == section.text:
                    idx = i
                    break
            vec = np.zeros(6, dtype=np.float32)
            vec[idx] = 1.0
            rows.append(vec)
        return np.vstack(rows)

    def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        mapping = {
            "food": 0,
            "hotel": 1,
            "airfare": 2,
            "first-class": 2,
            "limousine": 3,
            "receipt": 4,
            "taxi": 4,
            "gym": 0,
        }
        idx = 0
        lowered = query.lower()
        for key, value in mapping.items():
            if key in lowered:
                idx = value
        vec = np.zeros(6, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def generate(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("lab retrieve tests must not generate")

    def yes_probability(self, *args: object, **kwargs: object) -> float:
        return 0.99

    def close(self) -> None:
        return None


def _settings() -> Settings:
    return Settings(
        ollama_base_url="http://localhost:0",
        embed_model="stub",
        rerank_model="stub",
        gen_model="stub",
        vector_store="numpy",
        postgres_dsn="postgresql://unused",
        policy_path=REPO_ROOT / "policy.md",
        embed_cache_path=REPO_ROOT / ".cache" / "test-lab-embeddings.sqlite",
        answer_cache_path=REPO_ROOT / ".cache" / "test-lab-answers.sqlite",
        cassette_dir=REPO_ROOT / "tests" / "cassettes",
        cassette_mode="off",
        answer_cache=False,
    )


def test_lab_config_is_dense_only_with_three_final_sections() -> None:
    cfg = ABLATIONS["lab"]
    assert cfg.lab is True
    assert cfg.dense is True
    assert cfg.bm25 is False
    assert cfg.rerank is False
    assert cfg.final_sections == 3


def test_lab_index_stores_exactly_six_bare_section_chunks() -> None:
    client = StubEmbedClient()
    pipeline = Pipeline(
        _settings(),
        config=ABLATIONS["lab"],
        client=client,  # type: ignore[arg-type]
        store=NumpyStore(),
    )
    count = pipeline.index()

    assert count == 6
    assert pipeline.store.count() == 6
    assert all(c.kind == "section" for c in pipeline.store.all_chunks())
    # Bare body only — no contextual "Employee Expense Policy v2.0" prefix.
    assert client.embed_calls
    assert all(not text.startswith("Employee Expense Policy") for text in client.embed_calls[0])
    assert {c.text for c in pipeline.store.all_chunks()} == {s.text for s in SECTIONS}


def test_lab_retrieve_returns_at_most_three_sorted_by_cosine_distance() -> None:
    client = StubEmbedClient()
    store = NumpyStore()
    pipeline = Pipeline(
        _settings(),
        config=ABLATIONS["lab"],
        client=client,  # type: ignore[arg-type]
        store=store,
    )
    pipeline.index()
    hits, stages = pipeline.retrieve("Can I book first-class airfare?")

    assert len(hits) <= 3
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)
    assert hits[0].chunk.section == "3"
    assert any(s.name == "dense" for s in stages)
    assert not any(s.name in {"bm25", "rerank", "fuse"} for s in stages)


def test_full_remains_default_retrieval_config() -> None:
    assert RetrievalConfig().lab is False
    assert RetrievalConfig().bm25 is True
    assert RetrievalConfig().rerank is True


def test_lab_ask_builds_ephemeral_store_without_wiping_full_index() -> None:
    """Lab ask uses an in-memory section store; an existing full index stays untouched."""
    client = StubEmbedClient()
    full_store = NumpyStore()
    from ragpolicy.ingest import build_corpus

    corpus = build_corpus(RAW)
    dim = 6
    full_store.create(dim)
    matrix = np.eye(len(corpus), dim, dtype=np.float32)
    full_store.upsert(corpus, matrix)
    before = full_store.count()

    pipeline = Pipeline(
        _settings(),
        config=ABLATIONS["lab"],
        client=client,  # type: ignore[arg-type]
        store=full_store,
    )
    lab_store = pipeline.ensure_lab_store()
    hits, _ = pipeline.retrieve("How much can I spend on food each day?")

    assert full_store.count() == before
    assert lab_store.count() == 6
    assert lab_store is not full_store
    assert hits[0].chunk.section == "1"
