"""Ingestion is pure: no model calls, no I/O beyond reading the corpus."""

from __future__ import annotations

import pytest

from ragpolicy.config import REPO_ROOT
from ragpolicy.ingest import build_corpus, contextual_text, parse_document, parse_sections

RAW = (REPO_ROOT / "policy.md").read_text(encoding="utf-8")


def test_parses_document_title_and_version() -> None:
    doc = parse_document(RAW)
    assert doc.title == "Employee Expense Policy"
    assert doc.version == "2.0"
    assert doc.slug == "expense-policy"


def test_produces_exactly_one_chunk_per_numbered_section() -> None:
    sections = parse_sections(RAW)
    assert len(sections) == 6
    assert [s.section for s in sections] == ["1", "2", "3", "4", "5", "6"]
    assert [s.section_title for s in sections] == [
        "Meals",
        "Hotels",
        "Airfare",
        "Ground Transportation",
        "Receipts",
        "Submission Deadline",
    ]


def test_chunk_ids_match_the_lab_format() -> None:
    sections = parse_sections(RAW)
    assert sections[0].chunk_id == "expense-policy:v2.0:section-1"
    assert sections[5].chunk_id == "expense-policy:v2.0:section-6"


def test_every_chunk_text_is_exactly_its_span_in_the_raw_document() -> None:
    """The invariant the whole citation system rests on."""
    for chunk in build_corpus(RAW):
        assert RAW[chunk.start : chunk.end] == chunk.text, chunk.chunk_id


def test_section_text_carries_whole_sentences_only() -> None:
    sections = parse_sections(RAW)
    assert sections[0].text == (
        "Employees may claim up to $65 per day for meals while traveling overnight.\n"
        "Alcohol is not reimbursable."
    )
    for section in sections:
        assert section.text.rstrip().endswith(".")
        assert not section.text.startswith("#")


def test_sections_cover_disjoint_ascending_spans() -> None:
    sections = parse_sections(RAW)
    for earlier, later in zip(sections, sections[1:], strict=False):
        assert earlier.end < later.start


def test_propositions_split_each_section_into_atomic_rules() -> None:
    props = [c for c in build_corpus(RAW) if c.kind == "proposition"]
    texts = [p.text for p in props]

    assert "Alcohol is not reimbursable." in texts
    assert "Luxury vehicle upgrades are not reimbursable." in texts
    assert "Receipts are required for individual expenses of $25 or more." in texts
    # Section 4 must become separate facts, otherwise "rental car" matches the blur.
    section_4 = [p for p in props if p.section == "4"]
    assert len(section_4) == 2
    # Sections 1-4 hold two sentences each, sections 5-6 hold one.
    assert len(props) == 10


def test_propositions_link_to_their_parent_section() -> None:
    corpus = build_corpus(RAW)
    by_id = {c.chunk_id: c for c in corpus}
    for prop in (c for c in corpus if c.kind == "proposition"):
        assert prop.parent_chunk_id is not None
        parent = by_id[prop.parent_chunk_id]
        assert parent.kind == "section"
        assert parent.section == prop.section
        assert parent.start <= prop.start and prop.end <= parent.end


def test_proposition_ids_are_stable_and_ordered() -> None:
    props = [c for c in build_corpus(RAW) if c.section == "1" and c.kind == "proposition"]
    assert [p.chunk_id for p in props] == [
        "expense-policy:v2.0:section-1:p1",
        "expense-policy:v2.0:section-1:p2",
    ]


def test_contextual_prefix_names_the_document_and_section() -> None:
    props = [c for c in build_corpus(RAW) if c.kind == "proposition"]
    luxury = next(p for p in props if "Luxury" in p.text)
    assert contextual_text(luxury) == (
        "Employee Expense Policy v2.0, Section 4 (Ground Transportation): "
        "Luxury vehicle upgrades are not reimbursable."
    )


def test_corpus_is_sections_plus_propositions() -> None:
    corpus = build_corpus(RAW)
    assert sum(1 for c in corpus if c.kind == "section") == 6
    assert sum(1 for c in corpus if c.kind == "proposition") == 10
    assert len({c.chunk_id for c in corpus}) == len(corpus)


def test_citation_label_is_the_lab_section_format() -> None:
    sections = parse_sections(RAW)
    assert sections[0].citation_label == "1. Meals"
    assert sections[3].citation_label == "4. Ground Transportation"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One sentence only.", ["One sentence only."]),
        ("First one.\nSecond one.", ["First one.", "Second one."]),
        ("Costs $2.50 today. Then more.", ["Costs $2.50 today.", "Then more."]),
        ("Version 2.0 applies.", ["Version 2.0 applies."]),
    ],
)
def test_sentence_splitter_handles_decimals(text: str, expected: list[str]) -> None:
    from ragpolicy.ingest import split_sentences

    assert [t for t, _ in split_sentences(text, 0)] == expected
