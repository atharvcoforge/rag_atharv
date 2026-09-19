"""Exact-match ask cache: second identical question must not touch the models."""

from __future__ import annotations

from pathlib import Path

from ragpolicy.pipeline import RetrievalConfig
from ragpolicy.response_cache import AnswerCache, normalise_question


def test_normalise_folds_case_and_whitespace() -> None:
    assert normalise_question("  Wine? ") == normalise_question("wine?")


def test_cache_round_trips_a_payload(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path / "answers.sqlite")
    config = RetrievalConfig()
    payload = {"answer": "No.", "citation": None, "retrieved_chunks": [], "trace": {}}

    assert cache.get("wine?", config) is None
    cache.put("Wine ?", config, payload)
    assert cache.get("  wine?  ", config) == payload


def test_different_retrieval_configs_do_not_share_an_entry(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path / "answers.sqlite")
    payload = {"answer": "No.", "citation": None, "retrieved_chunks": [], "trace": {}}
    cache.put("wine?", RetrievalConfig(), payload)

    assert cache.get("wine?", RetrievalConfig(rerank=False)) is None


def test_clear_drops_every_entry(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path / "answers.sqlite")
    cache.put(
        "wine?",
        RetrievalConfig(),
        {"answer": "x", "citation": None, "retrieved_chunks": [], "trace": {}},
    )
    cache.clear()

    assert cache.get("wine?", RetrievalConfig()) is None
