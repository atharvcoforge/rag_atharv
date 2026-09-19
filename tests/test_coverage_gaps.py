"""Fill remaining coverage gaps that the main suites leave behind."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragpolicy import api
from ragpolicy.answer import _parse, peer_rule_mismatch
from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.ingest import parse_document
from ragpolicy.models import OllamaClient
from ragpolicy.pipeline import ABLATIONS, Pipeline
from ragpolicy.rerank_probe import CaseScore, ProbeResult, format_result
from ragpolicy.store import NumpyStore, PostgresStore, open_store


def test_parse_rejects_non_object_json() -> None:
    assert _parse("[1, 2]") is None
    assert _parse("null") is None
    assert _parse('{"answer": ""}') is None


def test_peer_rule_mismatch_for_first_class() -> None:
    assert peer_rule_mismatch(
        "Can I book first-class?",
        "Business-class requires vice president approval.",
        "Yes with VP approval.",
    )
    assert not peer_rule_mismatch(
        "Can I book first-class?",
        "Employees must purchase economy airfare.",
        "Economy is required.",
    )


def test_format_result_pass_and_fail() -> None:
    passed = ProbeResult(
        model="good",
        scores=(
            CaseScore(question="q1", governing=0.9, near_miss=0.1),
            CaseScore(question="rental?", governing=None, near_miss=0.05),
        ),
        ms_per_call=40.0,
        baseline_ms=150.0,
    )
    text = format_result(passed)
    assert "PASS" in text
    assert "no rule governs" in text

    failed = ProbeResult(
        model="bad",
        scores=(CaseScore(question="q", governing=0.2, near_miss=0.8),),
        ms_per_call=200.0,
        baseline_ms=150.0,
    )
    assert "FAIL" in format_result(failed)


def test_open_store_falls_back_to_numpy() -> None:
    assert isinstance(open_store("postgresql://unused", "numpy"), NumpyStore)
    assert isinstance(open_store("postgresql://127.0.0.1:1/nope", "postgres"), NumpyStore)


def test_postgres_rejects_unsafe_table_name() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        PostgresStore("postgresql://unused", table="chunks;drop")


def test_parse_document_requires_heading() -> None:
    with pytest.raises(ValueError, match="heading"):
        parse_document("No title here\n\n## 1. Section\n")


def test_ollama_close_and_empty_logprobs(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"response": "x", "logprobs": []})
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    client = OllamaClient(
        base_url="http://ollama.test",
        embed_model="e",
        rerank_model="r",
        gen_model="g",
        cache_path=tmp_path / "c.sqlite",
        transport=httpx.MockTransport(handler),
        rerank_base_url="http://rerank.test",
    )
    assert client.yes_probability("q") == 0.5
    assert client.generate("p", system="sys", schema={"type": "object"}) == "x"
    client.close()


def test_ollama_raises_on_error_payload(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "model missing"})

    client = OllamaClient(
        base_url="http://ollama.test",
        embed_model="e",
        rerank_model="r",
        gen_model="g",
        cache_path=tmp_path / "c.sqlite",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="ollama error"):
        client.generate("x")


def test_cached_pipeline_builds_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api._PIPELINES.clear()
    settings = Settings(
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
        answer_cache=False,
    )
    monkeypatch.setattr("ragpolicy.api.Settings.from_env", staticmethod(lambda: settings))
    monkeypatch.setattr("ragpolicy.evaluate._load_thresholds", lambda: None)
    monkeypatch.setattr("ragpolicy.pipeline.open_store", lambda dsn, backend: NumpyStore())

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            return None

    monkeypatch.setattr("ragpolicy.pipeline.OllamaClient", StubClient)
    first = api._cached_pipeline("full")
    second = api._cached_pipeline("full")
    assert first is second
    assert isinstance(first, Pipeline)
    assert api.pipeline_factory() is api._cached_pipeline
    api._PIPELINES.clear()


def test_stream_skips_blank_generate_chunks(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Blank wire lines and empty response tokens are both skipped.
        text = (
            "\n"
            + json.dumps({"response": "", "done": False})
            + "\n\n"
            + json.dumps({"response": "hi", "done": False})
            + "\n"
            + json.dumps({"response": "", "done": True})
            + "\n"
        )
        return httpx.Response(200, text=text)

    client = OllamaClient(
        base_url="http://ollama.test",
        embed_model="e",
        rerank_model="r",
        gen_model="g",
        cache_path=tmp_path / "c.sqlite",
        transport=httpx.MockTransport(handler),
    )
    assert list(client.generate_stream("p")) == ["hi"]


def test_ablations_include_lab_and_full() -> None:
    assert "lab" in ABLATIONS
    assert ABLATIONS["full"].verify is True
