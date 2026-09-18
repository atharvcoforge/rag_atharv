"""One suite, both backends.

Every test is parametrised over the NumPy and Postgres stores. If the fallback ever
diverges from pgvector, this file fails rather than the divergence quietly shipping.
Postgres tests skip when the container is not up.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest

from ragpolicy.ingest import Chunk
from ragpolicy.store import NumpyStore, PostgresStore, VectorStore

DSN = os.getenv("POSTGRES_DSN", "postgresql://rag:rag@localhost:5433/ragpolicy")
DIM = 4


def chunk(chunk_id: str, section: str, text: str, kind: str = "section") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=f"Section {section}",
        text=text,
        kind=kind,  # type: ignore[arg-type]
        start=0,
        end=len(text),
        parent_chunk_id=None if kind == "section" else f"expense-policy:v2.0:section-{section}",
    )


def unit(*values: float) -> npt.NDArray[np.float32]:
    vector = np.asarray(values, dtype=np.float32)
    normalised: npt.NDArray[np.float32] = vector / np.linalg.norm(vector)
    return normalised


FIXTURE = [
    (chunk("c1", "1", "meals and food"), unit(1, 0, 0, 0)),
    (chunk("c2", "2", "hotels and lodging"), unit(0, 1, 0, 0)),
    (chunk("c3", "3", "airfare and flights"), unit(0, 0, 1, 0)),
    (chunk("c4", "4", "taxis and trains"), unit(0.7, 0.7, 0, 0)),
]


def postgres_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(params=["numpy", "postgres"])
def store(request: pytest.FixtureRequest) -> Iterator[VectorStore]:
    if request.param == "numpy":
        yield NumpyStore()
        return

    if not postgres_available():
        pytest.skip("Postgres is not running; `docker compose up -d`")
    pg = PostgresStore(DSN, table="chunks_test")
    try:
        yield pg
    finally:
        pg.drop()
        pg.close()


def seeded(store: VectorStore) -> VectorStore:
    store.create(dim=DIM)
    store.upsert([c for c, _ in FIXTURE], np.vstack([v for _, v in FIXTURE]))
    return store


def test_search_ranks_by_ascending_cosine_distance(store: VectorStore) -> None:
    hits = seeded(store).search(unit(1, 0, 0, 0), limit=4)

    assert [h.chunk.chunk_id for h in hits][0] == "c1"
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_identical_vector_has_zero_distance(store: VectorStore) -> None:
    hits = seeded(store).search(unit(0, 1, 0, 0), limit=1)

    assert hits[0].chunk.chunk_id == "c2"
    assert math.isclose(hits[0].distance, 0.0, abs_tol=1e-5)


def test_orthogonal_vector_has_unit_distance(store: VectorStore) -> None:
    hits = {h.chunk.chunk_id: h.distance for h in seeded(store).search(unit(0, 0, 0, 1), limit=4)}

    assert math.isclose(hits["c1"], 1.0, abs_tol=1e-5)
    assert math.isclose(hits["c3"], 1.0, abs_tol=1e-5)


def test_limit_is_respected(store: VectorStore) -> None:
    assert len(seeded(store).search(unit(1, 0, 0, 0), limit=2)) == 2


def test_distance_is_a_plain_float(store: VectorStore) -> None:
    """The lab requires numbers, not formatted strings, in the response."""
    hit = seeded(store).search(unit(1, 0, 0, 0), limit=1)[0]

    assert type(hit.distance) is float


def test_metadata_round_trips_intact(store: VectorStore) -> None:
    hit = seeded(store).search(unit(0, 0, 1, 0), limit=1)[0]

    assert hit.chunk.document == "Employee Expense Policy"
    assert hit.chunk.version == "2.0"
    assert hit.chunk.section == "3"
    assert hit.chunk.section_title == "Section 3"
    assert hit.chunk.text == "airfare and flights"
    assert hit.chunk.kind == "section"
    assert hit.chunk.end == len("airfare and flights")


def test_parent_linkage_round_trips(store: VectorStore) -> None:
    store.create(dim=DIM)
    child = chunk("c1:p1", "1", "a proposition", kind="proposition")
    store.upsert([child], unit(1, 0, 0, 0).reshape(1, DIM))

    hit = store.search(unit(1, 0, 0, 0), limit=1)[0]
    assert hit.chunk.kind == "proposition"
    assert hit.chunk.parent_chunk_id == "expense-policy:v2.0:section-1"


def test_upsert_replaces_rather_than_duplicates(store: VectorStore) -> None:
    seeded(store)
    store.upsert([chunk("c1", "1", "meals, rewritten")], unit(1, 0, 0, 0).reshape(1, DIM))

    assert store.count() == 4
    assert store.search(unit(1, 0, 0, 0), limit=1)[0].chunk.text == "meals, rewritten"


def test_create_is_idempotent_and_clears(store: VectorStore) -> None:
    seeded(store)
    store.create(dim=DIM)
    assert store.count() == 0


def test_all_chunks_returns_the_corpus_for_lexical_search(store: VectorStore) -> None:
    chunks = seeded(store).all_chunks()

    assert {c.chunk_id for c in chunks} == {"c1", "c2", "c3", "c4"}


def test_count_reflects_what_was_written(store: VectorStore) -> None:
    assert seeded(store).count() == 4


def test_search_on_an_empty_store_returns_nothing(store: VectorStore) -> None:
    store.create(dim=DIM)
    assert store.search(unit(1, 0, 0, 0), limit=3) == []


def test_both_backends_agree_on_distances() -> None:
    """The point of the shared suite, stated as one explicit assertion."""
    if not postgres_available():
        pytest.skip("Postgres is not running; `docker compose up -d`")

    query = unit(0.4, 0.9, 0.1, 0)
    numpy_store = seeded(NumpyStore())
    pg = PostgresStore(DSN, table="chunks_parity")
    try:
        seeded(pg)
        numpy_hits = numpy_store.search(query, limit=4)
        pg_hits = pg.search(query, limit=4)

        assert [h.chunk.chunk_id for h in numpy_hits] == [h.chunk.chunk_id for h in pg_hits]
        for a, b in zip(numpy_hits, pg_hits, strict=True):
            assert math.isclose(a.distance, b.distance, abs_tol=1e-5)
    finally:
        pg.drop()
        pg.close()
