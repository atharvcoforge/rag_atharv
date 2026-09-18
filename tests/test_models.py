"""Ollama client: embeddings, the embedding cache, and yes/no logprob scoring.

Every test here runs against an httpx MockTransport, so the suite is hermetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from ragpolicy.models import QUERY_INSTRUCTION, OllamaClient


def make_client(
    handler: Any, tmp_path: Path, **kwargs: Any
) -> tuple[OllamaClient, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        response: httpx.Response = handler(request)
        return response

    client = OllamaClient(
        base_url="http://ollama.test",
        embed_model="embed-model",
        rerank_model="rerank-model",
        gen_model="gen-model",
        cache_path=tmp_path / "cache.sqlite",
        transport=httpx.MockTransport(record),
        **kwargs,
    )
    return client, seen


def embed_handler(dims: int = 4) -> Any:
    """Return distinct unnormalised vectors so normalisation is observable."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs = body["input"]
        vectors = [[float(len(text) + i) for i in range(dims)] for text in inputs]
        return httpx.Response(200, json={"embeddings": vectors})

    return handler


def logprob_handler(entries: list[tuple[str, float]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        top = [{"token": token, "logprob": lp} for token, lp in entries]
        return httpx.Response(
            200,
            json={
                "response": entries[0][0],
                "logprobs": [
                    {"token": entries[0][0], "logprob": entries[0][1], "top_logprobs": top}
                ],
            },
        )

    return handler


def test_embeddings_are_l2_normalised(tmp_path: Path) -> None:
    client, _ = make_client(embed_handler(), tmp_path)
    vectors = client.embed_documents(["alpha", "beta gamma"])

    assert vectors.shape == (2, 4)
    for row in vectors:
        assert math.isclose(float(np.linalg.norm(row)), 1.0, rel_tol=1e-6)


def test_documents_are_embedded_without_an_instruction_prefix(tmp_path: Path) -> None:
    client, seen = make_client(embed_handler(), tmp_path)
    client.embed_documents(["Alcohol is not reimbursable."])

    assert seen[0]["body"]["input"] == ["Alcohol is not reimbursable."]
    assert seen[0]["body"]["model"] == "embed-model"


def test_queries_get_the_instruction_prefix(tmp_path: Path) -> None:
    """Qwen3-Embedding is instruction aware and trained for asymmetric query/doc prompts."""
    client, seen = make_client(embed_handler(), tmp_path)
    client.embed_query("can I expense wine?")

    sent = seen[0]["body"]["input"][0]
    assert sent == f"Instruct: {QUERY_INSTRUCTION}\nQuery: can I expense wine?"


def test_query_embedding_is_a_single_normalised_vector(tmp_path: Path) -> None:
    client, _ = make_client(embed_handler(), tmp_path)
    vector = client.embed_query("hotels")

    assert vector.shape == (4,)
    assert math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=1e-6)


def test_repeated_text_is_served_from_cache(tmp_path: Path) -> None:
    client, seen = make_client(embed_handler(), tmp_path)
    first = client.embed_documents(["alpha", "beta"])
    second = client.embed_documents(["alpha", "beta"])

    assert len(seen) == 1
    np.testing.assert_allclose(first, second)


def test_cache_only_requests_the_missing_texts(tmp_path: Path) -> None:
    client, seen = make_client(embed_handler(), tmp_path)
    client.embed_documents(["alpha"])
    client.embed_documents(["alpha", "brand new text"])

    assert len(seen) == 2
    assert seen[1]["body"]["input"] == ["brand new text"]


def test_cache_is_keyed_by_model(tmp_path: Path) -> None:
    client, seen = make_client(embed_handler(), tmp_path)
    client.embed_documents(["alpha"])
    client.embed_model = "a-different-model"
    client.embed_documents(["alpha"])

    assert len(seen) == 2


def test_cache_survives_a_new_client(tmp_path: Path) -> None:
    client, _ = make_client(embed_handler(), tmp_path)
    expected = client.embed_documents(["alpha"])

    reopened, seen = make_client(embed_handler(), tmp_path)
    np.testing.assert_allclose(reopened.embed_documents(["alpha"]), expected)
    assert seen == []


def test_yes_probability_renormalises_over_yes_and_no(tmp_path: Path) -> None:
    # Equal mass on yes and no should read as exactly 0.5 even with other tokens present.
    client, _ = make_client(
        logprob_handler(
            [(" Yes", math.log(0.4)), (" No", math.log(0.4)), (" Maybe", math.log(0.2))]
        ),
        tmp_path,
    )
    assert math.isclose(client.yes_probability("is this relevant?"), 0.5, abs_tol=1e-6)


def test_yes_probability_sums_token_variants(tmp_path: Path) -> None:
    """Tokenisers emit ' Yes', 'Yes', ' yes' and 'yes'; all of them are a yes."""
    client, _ = make_client(
        logprob_handler(
            [
                (" Yes", math.log(0.3)),
                ("Yes", math.log(0.2)),
                ("yes", math.log(0.1)),
                (" No", math.log(0.4)),
            ]
        ),
        tmp_path,
    )
    assert math.isclose(client.yes_probability("q"), 0.6, abs_tol=1e-6)


def test_yes_probability_is_zero_when_no_yes_token_appears(tmp_path: Path) -> None:
    client, _ = make_client(
        logprob_handler([(" No", math.log(0.9)), (" Never", math.log(0.1))]), tmp_path
    )
    assert client.yes_probability("q") == 0.0


def test_yes_probability_is_neutral_when_neither_token_appears(tmp_path: Path) -> None:
    client, _ = make_client(
        logprob_handler([(" Maybe", math.log(0.6)), (" Perhaps", math.log(0.4))]), tmp_path
    )
    assert client.yes_probability("q") == 0.5


def test_yes_probability_requests_a_single_deterministic_token(tmp_path: Path) -> None:
    client, seen = make_client(logprob_handler([(" Yes", math.log(0.99))]), tmp_path)
    client.yes_probability("q")

    body = seen[0]["body"]
    assert body["model"] == "rerank-model"
    assert body["logprobs"] is True
    assert body["top_logprobs"] >= 5
    assert body["options"]["num_predict"] == 1
    assert body["options"]["temperature"] == 0
    assert body["stream"] is False


def test_generate_returns_the_response_text(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": '{"answer": "ok"}'})

    client, seen = make_client(handler, tmp_path)
    assert client.generate("prompt", schema={"type": "object"}) == '{"answer": "ok"}'
    assert seen[0]["body"]["format"] == {"type": "object"}
    assert seen[0]["body"]["options"]["temperature"] == 0


def test_generate_stream_yields_incremental_tokens(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"response": "Employees ", "done": False}),
            json.dumps({"response": "may claim", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        return httpx.Response(200, text="\n".join(lines))

    client, _ = make_client(handler, tmp_path)
    assert list(client.generate_stream("prompt")) == ["Employees ", "may claim"]


def test_embed_rejects_a_dimension_mismatch(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0], [1.0, 2.0, 3.0]]})

    client, _ = make_client(handler, tmp_path)
    with pytest.raises(ValueError, match="dimension"):
        client.embed_documents(["a", "b"])


def test_zero_vector_does_not_divide_by_zero(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.0, 0.0, 0.0]]})

    client, _ = make_client(handler, tmp_path)
    vectors = client.embed_documents(["a"])
    assert np.all(np.isfinite(vectors))
