"""HTTP contract for the API, exercised against a fake pipeline.

No Ollama and no Postgres: responses are produced by the real :class:`Answerer` with a
stub client, then served through a pipeline stand-in. What is under test here is that
the lab's contract survives the HTTP and JSON round trip, not the model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ragpolicy import api
from ragpolicy.answer import REFUSAL, Answerer, Response, Thresholds
from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.ingest import build_corpus
from ragpolicy.pipeline import ABLATIONS, Pipeline
from ragpolicy.retrieve import ScoredHit
from ragpolicy.store import NumpyStore

RAW = (REPO_ROOT / "policy.md").read_text(encoding="utf-8")
CORPUS = build_corpus(RAW)
BY_ID = {c.chunk_id: c for c in CORPUS}
STAGES = [{"name": "dense", "ms": 6.0, "hits": 12}, {"name": "rerank", "ms": 918.0}]


class StubClient:
    """Scripted generation and entailment, same style as tests/test_answer.py."""

    def __init__(self, payload: Mapping[str, object], entailment: float = 0.99) -> None:
        self.payload = payload
        self.entailment = entailment

    def generate(self, prompt: str, **kwargs: object) -> str:
        return json.dumps(self.payload)

    def yes_probability(self, prompt: str, **kwargs: object) -> float:
        return self.entailment

    def warm(self) -> dict[str, float]:
        return {"rerank_ms": 12.0, "gen_ms": 34.0}


def scored(chunk_id: str, score: float, distance: float) -> ScoredHit:
    return ScoredHit(chunk=BY_ID[chunk_id], distance=distance, score=score)


def answered(payload: Mapping[str, object], hits: list[ScoredHit], tau: float = 0.1) -> Response:
    """A real Response, built by the real answerer, with pipeline stage timings attached."""
    answerer = Answerer(
        client=StubClient(payload),
        raw_policy=RAW,
        corpus=CORPUS,
        thresholds=Thresholds(tau=tau, delta=0.0),
    )
    response = answerer.answer("q", hits)
    response.trace["stages"] = STAGES
    response.trace["total_ms"] = 1200.0
    return response


class FakePipeline:
    """Stands in for Pipeline: same attributes, no connections, no models."""

    def __init__(self, response: Response, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.raw = RAW
        self.corpus = CORPUS
        self.settings = Settings.from_env()
        self.store = NumpyStore()
        self.client = StubClient({})
        self.asked: list[str] = []

    def ask(self, question: str, on_event: Any = None) -> Response:
        self.asked.append(question)
        if self.error is not None:
            raise self.error
        if on_event is not None:
            for stage in STAGES:
                on_event("stage", {"name": stage["name"], "ms": stage["ms"]})
            for phase in ("generating", "verifying"):
                on_event("status", {"phase": phase})
        return self.response

    def index(self) -> int:
        return len(self.corpus)


ANSWERED = {
    "answer": "Employees may claim up to $65 per day for meals.",
    "section": "1",
    "governing_rule": "Employees may claim up to $65 per day for meals while traveling overnight.",
}
HITS = [
    scored("expense-policy:v2.0:section-1", 0.95, distance=0.41),
    scored("expense-policy:v2.0:section-2", 0.30, distance=0.08),
    scored("expense-policy:v2.0:section-3", 0.20, distance=0.22),
    scored("expense-policy:v2.0:section-4", 0.10, distance=0.63),
]


@contextmanager
def client_for(pipeline: FakePipeline) -> Iterator[TestClient]:
    api.app.dependency_overrides[api.pipeline_factory] = lambda: lambda _config: pipeline
    with TestClient(api.app) as client:
        yield client
    api.app.dependency_overrides.clear()


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline(answered(ANSWERED, HITS))


@pytest.fixture
def client(pipeline: FakePipeline) -> Iterator[TestClient]:
    with client_for(pipeline) as http:
        yield http


def events(body: str) -> list[tuple[str, Any]]:
    """Parse an SSE stream into ``(event, decoded data)`` pairs."""
    parsed = []
    for block in body.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        parsed.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return parsed


# -- POST /api/ask -----------------------------------------------------------------


def test_ask_returns_the_lab_response_schema(client: TestClient) -> None:
    payload = client.post("/api/ask", json={"question": "meal cap?"}).json()

    assert set(payload) == {"answer", "citation", "retrieved_chunks", "trace"}
    assert payload["answer"] == ANSWERED["answer"]
    assert payload["citation"] == {
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1. Meals",
    }


def test_ask_returns_at_most_three_chunks_ascending_by_numeric_distance(
    client: TestClient,
) -> None:
    """Four hits go in; the contract allows three, sorted, with distances as numbers."""
    chunks = client.post("/api/ask", json={"question": "meal cap?"}).json()["retrieved_chunks"]

    assert len(chunks) == 3
    distances = [c["distance"] for c in chunks]
    assert all(isinstance(d, float) for d in distances)
    assert distances == sorted(distances)
    assert all(set(c) == {"section", "distance"} for c in chunks)


def test_ask_passes_the_question_through(client: TestClient, pipeline: FakePipeline) -> None:
    client.post("/api/ask", json={"question": "what is the hotel cap?"})

    assert pipeline.asked == ["what is the hotel cap?"]


def test_refusal_keeps_the_schema_with_a_null_citation() -> None:
    refused = FakePipeline(answered(ANSWERED, HITS, tau=0.99))
    with client_for(refused) as http:
        payload = http.post("/api/ask", json={"question": "parental leave?"}).json()

    assert payload["answer"] == REFUSAL
    assert payload["citation"] is None
    assert isinstance(payload["retrieved_chunks"], list)
    assert payload["trace"]["abstained_at"] == "retrieval"


def test_ask_rejects_an_unknown_config(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "q", "config": "nonsense"})

    assert response.status_code == 422


def test_ask_requires_a_question(client: TestClient) -> None:
    assert client.post("/api/ask", json={}).status_code == 422


# -- GET /api/ask/stream -----------------------------------------------------------


def test_stream_emits_stages_then_phases_then_done(client: TestClient) -> None:
    response = client.get("/api/ask/stream", params={"question": "meal cap?"})

    assert response.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in events(response.text)]
    assert [name for name, _ in groupby(names)] == ["stage", "status", "done"]
    assert names.count("stage") == len(STAGES)


def test_stream_stages_carry_a_name_and_a_duration(client: TestClient) -> None:
    body = client.get("/api/ask/stream", params={"question": "q"}).text
    stages = [data for name, data in events(body) if name == "stage"]

    assert stages == [{"name": "dense", "ms": 6.0}, {"name": "rerank", "ms": 918.0}]


def test_stream_announces_the_generation_and_verification_phases(client: TestClient) -> None:
    body = client.get("/api/ask/stream", params={"question": "q"}).text
    statuses = [data for name, data in events(body) if name == "status"]

    assert statuses == [{"phase": "generating"}, {"phase": "verifying"}]


def test_stream_withholds_the_answer_until_the_gates_have_run(client: TestClient) -> None:
    """No token replay: text the verification gate may still refuse must not be shown."""
    body = client.get("/api/ask/stream", params={"question": "q"}).text
    before_done = [name for name, _ in events(body)][:-1]

    assert "token" not in before_done
    assert ANSWERED["answer"] not in body.split("event: done")[0]


def test_pipeline_publishes_retrieval_stages_before_the_answer_phases(
    tmp_path: Path,
) -> None:
    """The seam the stream depends on: real Pipeline, real Answerer, events in order."""
    from dataclasses import replace

    class EmbedAndGenerate(StubClient):
        """One-hot section vectors so section 1 wins, plus a quote it really contains."""

        def embed_documents(self, texts: list[str]) -> Any:
            return np.eye(len(texts), 6, dtype=np.float32)

        def embed_query(self, query: str) -> Any:
            return np.eye(1, 6, dtype=np.float32)[0]

    pipeline = Pipeline(
        replace(
            Settings.from_env(),
            answer_cache=False,
            answer_cache_path=tmp_path / "unused.sqlite",
        ),
        config=ABLATIONS["lab"],
        client=EmbedAndGenerate(  # type: ignore[arg-type]
            {
                "answer": "No, alcohol is not reimbursable.",
                "section": "1",
                "governing_rule": "Alcohol is not reimbursable.",
            }
        ),
        store=NumpyStore(),
        thresholds=Thresholds(tau=0.1, delta=0.0),
    )

    seen: list[tuple[str, dict[str, Any]]] = []
    response = pipeline.ask("can I expense wine?", on_event=lambda n, d: seen.append((n, d)))

    assert response.citation is not None
    assert [name for name, _ in seen] == ["stage", "status", "status"]
    assert seen[0][1]["name"] == "dense"
    assert [data["phase"] for _, data in seen[1:]] == ["generating", "verifying"]


def test_identical_questions_are_served_from_the_answer_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry must not re-pay the 8B — that is the UI complaint this cache answers."""
    from dataclasses import replace

    class Counting(StubClient):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.generations = 0

        def embed_documents(self, texts: list[str]) -> Any:
            return np.eye(len(texts), 6, dtype=np.float32)

        def embed_query(self, query: str) -> Any:
            return np.eye(1, 6, dtype=np.float32)[0]

        def generate(self, prompt: str, **kwargs: object) -> str:
            self.generations += 1
            return super().generate(prompt, **kwargs)

    client = Counting(
        {
            "answer": "No, alcohol is not reimbursable.",
            "section": "1",
            "governing_rule": "Alcohol is not reimbursable.",
        }
    )
    settings = replace(
        Settings.from_env(),
        answer_cache=True,
        answer_cache_path=tmp_path / "answers.sqlite",
    )
    pipeline = Pipeline(
        settings,
        config=ABLATIONS["lab"],
        client=client,  # type: ignore[arg-type]
        store=NumpyStore(),
        thresholds=Thresholds(tau=0.1, delta=0.0),
    )

    first_events: list[tuple[str, dict[str, Any]]] = []
    first = pipeline.ask("can I expense wine?", on_event=lambda n, d: first_events.append((n, d)))
    second_events: list[tuple[str, dict[str, Any]]] = []
    second = pipeline.ask("Can I expense wine?", on_event=lambda n, d: second_events.append((n, d)))

    assert client.generations == 1
    assert first.answer == second.answer
    assert second.trace["cache_hit"] is True
    assert second.trace["total_ms"] < 50
    assert [data["phase"] for name, data in second_events if name == "status"] == ["cached"]


def test_stream_reports_a_pipeline_failure_as_an_error_event() -> None:
    broken = FakePipeline(answered(ANSWERED, HITS), error=RuntimeError("ollama is down"))
    with client_for(broken) as http:
        body = http.get("/api/ask/stream", params={"question": "q"}).text

    name, data = events(body)[-1]
    assert (name, data) == ("error", {"message": "ollama is down"})


def test_stream_done_carries_the_full_ask_payload(client: TestClient) -> None:
    body = client.get("/api/ask/stream", params={"question": "meal cap?"}).text
    name, done = events(body)[-1]

    assert name == "done"
    assert done == client.post("/api/ask", json={"question": "meal cap?"}).json()


# -- GET /api/document -------------------------------------------------------------


def test_document_offsets_index_the_raw_text_exactly(client: TestClient) -> None:
    """The UI highlights by character offset, so this invariant is load-bearing."""
    payload = client.get("/api/document").json()

    assert payload["document"] == "Employee Expense Policy"
    assert payload["version"] == "2.0"
    assert payload["sections"]
    for section in payload["sections"]:
        assert payload["raw"][section["start"] : section["end"]] == section["text"]


def test_document_returns_sections_only(client: TestClient) -> None:
    payload = client.get("/api/document").json()

    fields = {"chunk_id", "section", "section_title", "text", "start", "end"}
    assert len(payload["sections"]) == len([c for c in CORPUS if c.kind == "section"])
    assert all(set(section) == fields for section in payload["sections"])


# -- the rest ----------------------------------------------------------------------


def test_index_reports_how_many_chunks_were_stored(client: TestClient) -> None:
    assert client.post("/api/index").json() == {"indexed": len(CORPUS)}


def test_health_reports_the_store_and_models(client: TestClient, pipeline: FakePipeline) -> None:
    payload = client.get("/api/health").json()

    assert payload["ok"] is True
    assert payload["store"] == "NumpyStore"
    assert payload["chunks"] == len(CORPUS)
    assert payload["models"]["gen"] == pipeline.settings.gen_model


def test_health_reports_both_ollama_endpoints(client: TestClient, pipeline: FakePipeline) -> None:
    """Ops needs to see whether rerank is actually running somewhere of its own."""
    endpoints = client.get("/api/health").json()["endpoints"]

    assert endpoints["ollama"] == pipeline.settings.ollama_base_url
    assert endpoints["rerank"] == (
        pipeline.settings.rerank_ollama_url or pipeline.settings.ollama_base_url
    )


def test_warm_loads_both_models(client: TestClient) -> None:
    assert client.post("/api/warm").json() == {"rerank_ms": 12.0, "gen_ms": 34.0}


def test_eval_latest_is_empty_when_no_sweep_has_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api, "ABLATION_REPORT", tmp_path / "missing.json")

    assert client.get("/api/eval/latest").json() == {}


def test_eval_latest_returns_the_report(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = tmp_path / "ablation.json"
    report.write_text(json.dumps({"full": {"recall_at_3": 1.0}}), encoding="utf-8")
    monkeypatch.setattr(api, "ABLATION_REPORT", report)

    assert client.get("/api/eval/latest").json() == {"full": {"recall_at_3": 1.0}}
