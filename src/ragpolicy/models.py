"""Thin Ollama client: embeddings with a persistent cache, and yes/no logprob scoring.

The interesting part is :meth:`OllamaClient.yes_probability`. Ollama exposes
``logprobs``/``top_logprobs``, which turns any causal model into a scorer: constrain the
generation to a single token, read the distribution, and renormalise over the yes and no
mass. That is exactly how Qwen3-Reranker is meant to be used, so the reranker runs as a
genuine cross-encoder with no PyTorch in the dependency tree.

The same primitive then powers claim-level entailment checking during verification.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

QUERY_INSTRUCTION = (
    "Given an employee expense-policy question, " "retrieve the policy rule that answers it"
)

_YES_TOKENS = frozenset({"yes", "true"})
_NO_TOKENS = frozenset({"no", "false"})

Vector = npt.NDArray[np.float32]


def _normalise(matrix: Vector) -> Vector:
    """L2-normalise rows so cosine distance reduces to ``1 - dot``."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    scaled: Vector = (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)
    return scaled


class _EmbeddingCache:
    """Content-addressed embedding cache. Re-indexing an unchanged policy costs nothing."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vec BLOB)")
        self._db.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get(self, key: str) -> Vector | None:
        row = self._db.execute("SELECT vec FROM embeddings WHERE key = ?", (key,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def put(self, pairs: list[tuple[str, Vector]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vec) VALUES (?, ?)",
            [(key, vector.astype(np.float32).tobytes()) for key, vector in pairs],
        )
        self._db.commit()


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        embed_model: str,
        rerank_model: str,
        gen_model: str,
        cache_path: Path,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.gen_model = gen_model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._cache = _EmbeddingCache(cache_path)

    def close(self) -> None:
        self._http.close()

    # -- embeddings ----------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> Vector:
        """Embed raw document text. Returns an ``(n, dim)`` L2-normalised matrix."""
        keys = [_EmbeddingCache.key(self.embed_model, text) for text in texts]
        cached = [self._cache.get(key) for key in keys]

        missing = [i for i, vector in enumerate(cached) if vector is None]
        if missing:
            fetched = self._embed_uncached([texts[i] for i in missing])
            self._cache.put([(keys[i], fetched[n]) for n, i in enumerate(missing)])
            for n, i in enumerate(missing):
                cached[i] = fetched[n]

        stacked: Vector = np.vstack([v for v in cached if v is not None]).astype(np.float32)
        return stacked

    def embed_query(self, query: str) -> Vector:
        """Embed a question with the instruction prefix the embedding model expects."""
        prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        vector: Vector = self.embed_documents([prompt])[0]
        return vector

    def _embed_uncached(self, texts: list[str]) -> Vector:
        payload = {"model": self.embed_model, "input": texts}
        raw = self._post("/api/embed", payload)["embeddings"]
        if len({len(vector) for vector in raw}) != 1:
            raise ValueError("embedding dimension is inconsistent across the batch")
        return _normalise(np.asarray(raw, dtype=np.float32))

    # -- scoring -------------------------------------------------------------------

    def yes_probability(
        self, prompt: str, *, system: str | None = None, model: str | None = None
    ) -> float:
        """P(yes) from the first generated token, renormalised over yes and no mass.

        Renormalising rather than reading the raw probability matters: a model that puts
        0.4 on yes, 0.4 on no, and 0.2 elsewhere is undecided, and should score 0.5 rather
        than 0.4. Returns 0.5 when the model commits to neither.

        ``think`` must be off. With thinking enabled a Qwen3 model spends its first token
        on ``<think>`` and the yes/no distribution never appears.
        """
        payload: dict[str, Any] = {
            "model": model or self.rerank_model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "think": False,
            "options": {"num_predict": 1, "temperature": 0, "seed": 0},
        }
        if system is not None:
            payload["system"] = system
        entries = self._post("/api/generate", payload).get("logprobs") or []
        if not entries:
            return 0.5

        yes = no = 0.0
        for candidate in entries[0].get("top_logprobs", []):
            token = candidate["token"].strip().lower()
            probability = math.exp(candidate["logprob"])
            if token in _YES_TOKENS:
                yes += probability
            elif token in _NO_TOKENS:
                no += probability

        total = yes + no
        return 0.5 if total == 0.0 else yes / total

    # -- generation ----------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        num_predict: int = 512,
    ) -> str:
        payload = self._generation_payload(prompt, system, schema, model, num_predict)
        payload["stream"] = False
        return str(self._post("/api/generate", payload).get("response", ""))

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        num_predict: int = 512,
    ) -> Iterator[str]:
        payload = self._generation_payload(prompt, system, schema, model, num_predict)
        payload["stream"] = True
        with self._http.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("response"):
                    yield chunk["response"]
                if chunk.get("done"):
                    return

    def _generation_payload(
        self,
        prompt: str,
        system: str | None,
        schema: dict[str, Any] | None,
        model: str | None,
        num_predict: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.gen_model,
            "prompt": prompt,
            # Deterministic by construction: the eval numbers have to be reproducible.
            "options": {"temperature": 0, "top_p": 1, "seed": 0, "num_predict": num_predict},
            "think": False,
        }
        if system is not None:
            payload["system"] = system
        if schema is not None:
            payload["format"] = schema
        return payload

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post(path, json=payload)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        if "error" in result:
            raise RuntimeError(f"ollama error on {path}: {result['error']}")
        return result
