"""Parse the policy into section chunks and atomic propositions.

Two units come out of here and they do different jobs:

- **Sections** are the unit of citation. One per numbered heading, whole sentences only.
- **Propositions** are the unit of retrieval. One per sentence, linked to a parent section.

Splitting retrieval away from citation is what lets the system abstain on a near miss.
A single vector for "taxi, rideshare, train, transit, no luxury upgrades" sits moderately
close to *any* transport question including rental cars; four separate vectors do not.

Every chunk satisfies ``raw[chunk.start:chunk.end] == chunk.text``. Citations are checked
against that invariant later, which is what makes a fabricated citation impossible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ChunkKind = Literal["section", "proposition"]

_H1 = re.compile(r"^#\s+(?P<title>.+?)\s*[\u2014\u2013-]\s*Version\s+(?P<version>[\d.]+)\s*$", re.M)
_H2 = re.compile(r"^##\s+(?P<number>\d+)\.\s+(?P<title>.+?)\s*$", re.M)

# Split after sentence punctuation only when the next token starts a new sentence.
# The lookahead is what keeps "$2.50" and "Version 2.0" intact.
# ponytail: naive for prose with abbreviations ("e.g.", "Inc."); swap in a real
# segmenter (pysbd/spaCy) if the corpus ever grows past hand-written policy text.
_SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z$"\'(\[])')

_DEFAULT_SLUG = "expense-policy"


@dataclass(frozen=True, slots=True)
class Document:
    title: str
    version: str
    slug: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    text: str
    kind: ChunkKind
    start: int
    end: int
    parent_chunk_id: str | None = None

    @property
    def citation_label(self) -> str:
        """The section format the lab's output contract asks for, e.g. ``1. Meals``."""
        return f"{self.section}. {self.section_title}"


def parse_document(raw: str, slug: str = _DEFAULT_SLUG) -> Document:
    match = _H1.search(raw)
    if match is None:
        raise ValueError("policy is missing an '# <title> - Version <n>' heading")
    return Document(title=match["title"].strip(), version=match["version"], slug=slug)


def split_sentences(text: str, offset: int) -> list[tuple[str, int]]:
    """Split into sentences, returning each with its absolute start offset."""
    sentences: list[tuple[str, int]] = []
    cursor = 0
    for piece in _SENTENCE_BREAK.split(text):
        start = text.index(piece, cursor)
        sentences.append((piece, offset + start))
        cursor = start + len(piece)
    return sentences


def parse_sections(raw: str, slug: str = _DEFAULT_SLUG) -> list[Chunk]:
    """One chunk per numbered heading, holding the body text only."""
    doc = parse_document(raw, slug)
    headings = list(_H2.finditer(raw))
    sections: list[Chunk] = []

    for index, heading in enumerate(headings):
        body_start = heading.end()
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        body = raw[body_start:body_end]

        # Trim surrounding whitespace without losing the offsets it occupied.
        start = body_start + len(body) - len(body.lstrip())
        end = body_start + len(body.rstrip())

        number = heading["number"]
        sections.append(
            Chunk(
                chunk_id=f"{doc.slug}:v{doc.version}:section-{number}",
                document=doc.title,
                version=doc.version,
                section=number,
                section_title=heading["title"],
                text=raw[start:end],
                kind="section",
                start=start,
                end=end,
            )
        )
    return sections


def build_corpus(raw: str, slug: str = _DEFAULT_SLUG) -> list[Chunk]:
    """Sections followed by their propositions, in document order."""
    corpus: list[Chunk] = []
    for section in parse_sections(raw, slug):
        corpus.append(section)
        for ordinal, (text, start) in enumerate(
            split_sentences(section.text, section.start), start=1
        ):
            corpus.append(
                Chunk(
                    chunk_id=f"{section.chunk_id}:p{ordinal}",
                    document=section.document,
                    version=section.version,
                    section=section.section,
                    section_title=section.section_title,
                    text=text,
                    kind="proposition",
                    start=start,
                    end=start + len(text),
                    parent_chunk_id=section.chunk_id,
                )
            )
    return corpus


def contextual_text(chunk: Chunk) -> str:
    """The string that actually gets embedded.

    Contextual Retrieval (Anthropic, 2024) with the LLM-written context replaced by a
    template. An isolated "Luxury vehicle upgrades are not reimbursable." has no anchor to
    travel, expenses, or ground transport; the prefix supplies all three deterministically.
    """
    return (
        f"{chunk.document} v{chunk.version}, "
        f"Section {chunk.section} ({chunk.section_title}): {chunk.text}"
    )
