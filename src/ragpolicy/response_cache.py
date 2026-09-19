"""Exact-match answer cache so a repeated question does not re-pay the 8B.

Only embeddings were cached before. Rerank + generate + verify still ran in full on
every ask — including the same string typed twice. That is why the UI felt unchanged
on a retry.

Keying is the normalised question plus a fingerprint of the retrieval config. Near-
duplicate phrasing is intentionally *not* matched: a semantic cache would need its
own quality gate, and the rental-car failure mode is exactly the kind of near-miss
that looks similar and must not share an answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def normalise_question(question: str) -> str:
    """Fold case and whitespace so 'Wine?' and '  wine ? ' hit the same entry."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    return re.sub(r"\s+([?.!,;:])", r"\1", text)


def config_fingerprint(config: Any) -> str:
    """Stable key from the retrieval knobs that can change the answer."""
    fields = (
        "dense",
        "bm25",
        "rerank",
        "verify",
        "candidates",
        "rerank_candidates",
        "final_sections",
        "lab",
    )
    payload = {name: getattr(config, name) for name in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class AnswerCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers ("
            " key TEXT PRIMARY KEY, payload TEXT NOT NULL, stored_at REAL NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def key(question: str, config: Any) -> str:
        return hashlib.sha256(
            f"{normalise_question(question)}\x00{config_fingerprint(config)}".encode()
        ).hexdigest()

    def get(self, question: str, config: Any) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT payload FROM answers WHERE key = ?", (self.key(question, config),)
        ).fetchone()
        if row is None:
            return None
        loaded: dict[str, Any] = json.loads(row[0])
        return loaded

    def put(self, question: str, config: Any, payload: dict[str, Any]) -> None:
        import time

        self._db.execute(
            "INSERT OR REPLACE INTO answers (key, payload, stored_at) VALUES (?, ?, ?)",
            (self.key(question, config), json.dumps(payload), time.time()),
        )
        self._db.commit()

    def clear(self) -> None:
        self._db.execute("DELETE FROM answers")
        self._db.commit()
