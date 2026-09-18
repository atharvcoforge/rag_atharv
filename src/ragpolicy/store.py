"""Vector storage: pgvector in production, NumPy everywhere a container is inconvenient.

Both implementations satisfy the same protocol and are verified by the same test suite,
including an explicit parity test, so the fallback cannot quietly drift from pgvector.

Vectors arrive L2-normalised, so cosine distance is ``1 - dot`` and the two backends agree
to floating-point tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ragpolicy.ingest import Chunk

Vector = npt.NDArray[np.float32]

_COLUMNS = (
    "chunk_id, document, version, section, section_title, text, kind, "
    "span_start, span_end, parent_chunk_id"
)


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    distance: float


class VectorStore(Protocol):
    def create(self, dim: int) -> None:
        """Create the store empty, replacing anything already there."""

    def upsert(self, chunks: list[Chunk], vectors: Vector) -> None: ...

    def search(self, query: Vector, limit: int) -> list[Hit]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    return Chunk(
        chunk_id=row[0],
        document=row[1],
        version=row[2],
        section=row[3],
        section_title=row[4],
        text=row[5],
        kind=row[6],
        start=row[7],
        end=row[8],
        parent_chunk_id=row[9],
    )


def _chunk_to_row(chunk: Chunk) -> tuple[Any, ...]:
    return (
        chunk.chunk_id,
        chunk.document,
        chunk.version,
        chunk.section,
        chunk.section_title,
        chunk.text,
        chunk.kind,
        chunk.start,
        chunk.end,
        chunk.parent_chunk_id,
    )


class NumpyStore:
    """Exact brute-force search. For a corpus this size it is also the fastest option."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, Vector] = {}
        self._dim = 0

    def create(self, dim: int) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._dim = dim

    def upsert(self, chunks: list[Chunk], vectors: Vector) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = np.asarray(vector, dtype=np.float32)

    def search(self, query: Vector, limit: int) -> list[Hit]:
        if not self._vectors:
            return []
        ids = list(self._vectors)
        matrix = np.vstack([self._vectors[i] for i in ids])
        distances = 1.0 - matrix @ np.asarray(query, dtype=np.float32)
        order = np.argsort(distances, kind="stable")[:limit]
        return [Hit(chunk=self._chunks[ids[i]], distance=float(distances[i])) for i in order]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)


class PostgresStore:
    """pgvector with an HNSW index over ``vector_cosine_ops``.

    The dense query is the one the lab specifies::

        ORDER BY embedding <=> %(query)s ASC LIMIT %(limit)s
    """

    def __init__(self, dsn: str, table: str = "chunks") -> None:
        import psycopg

        if not table.replace("_", "").isalnum():
            raise ValueError(f"unsafe table name: {table!r}")
        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def create(self, dim: int) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        self._conn.execute(
            f"""
            CREATE TABLE {self.table} (
                chunk_id        TEXT PRIMARY KEY,
                document        TEXT NOT NULL,
                version         TEXT NOT NULL,
                section         TEXT NOT NULL,
                section_title   TEXT NOT NULL,
                text            TEXT NOT NULL,
                kind            TEXT NOT NULL,
                span_start      INTEGER NOT NULL,
                span_end        INTEGER NOT NULL,
                parent_chunk_id TEXT,
                embedding       vector({dim}) NOT NULL
            )
            """
        )
        # m/ef_construction are pgvector defaults; the corpus is far too small to tune.
        self._conn.execute(
            f"CREATE INDEX {self.table}_hnsw ON {self.table} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )

    def upsert(self, chunks: list[Chunk], vectors: Vector) -> None:
        matrix = np.asarray(vectors, dtype=np.float32).reshape(len(chunks), -1)
        rows = [(*_chunk_to_row(chunk), _to_pgvector(matrix[i])) for i, chunk in enumerate(chunks)]
        updates = ", ".join(
            f"{column.strip()} = EXCLUDED.{column.strip()}"
            for column in _COLUMNS.split(",")
            if column.strip() != "chunk_id"
        )
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} ({_COLUMNS}, embedding) "
                f"VALUES ({', '.join(['%s'] * 11)}) "
                f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}, embedding = EXCLUDED.embedding",
                rows,
            )

    def search(self, query: Vector, limit: int) -> list[Hit]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS}, embedding <=> %s AS distance "
            f"FROM {self.table} ORDER BY embedding <=> %s ASC LIMIT %s",
            (_to_pgvector(query), _to_pgvector(query), limit),
        ).fetchall()
        return [Hit(chunk=_row_to_chunk(row), distance=float(row[10])) for row in rows]

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(f"SELECT {_COLUMNS} FROM {self.table} ORDER BY chunk_id")
        return [_row_to_chunk(row) for row in rows.fetchall()]

    def count(self) -> int:
        row = self._conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()
        return 0 if row is None else int(row[0])

    def drop(self) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {self.table}")

    def close(self) -> None:
        self._conn.close()


def _to_pgvector(vector: Vector) -> str:
    """pgvector's text input format. Avoids needing the pgvector-python adapter."""
    return "[" + ",".join(repr(float(v)) for v in np.asarray(vector).ravel()) + "]"


def open_store(dsn: str, backend: str) -> VectorStore:
    """Open the requested backend, falling back to NumPy when Postgres is unreachable."""
    if backend != "postgres":
        return NumpyStore()
    try:
        return PostgresStore(dsn)
    except Exception:
        return NumpyStore()
