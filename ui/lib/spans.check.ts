/**
 * Self-check for the character-offset highlighting.
 * Run: node --experimental-strip-types lib/spans.check.ts
 */
import assert from "node:assert/strict";
import { MOCK_DOCUMENT, mockAsk } from "./mock.ts";
import { cut } from "./spans.ts";
import type { Span } from "./types.ts";

const span = (start: number, end: number): Span => ({
  chunk_id: "c",
  start,
  end,
  text: "",
});

const raw = "abcdefghij";

// Every character is emitted exactly once, in order.
for (const spans of [
  [],
  [span(0, 3)],
  [span(7, 10)],
  [span(2, 4), span(6, 9)],
  [span(2, 6), span(4, 8)], // overlapping -> clipped
  [span(5, 5)], // empty -> dropped
  [span(-2, 3)], // out of range -> dropped
  [span(8, 99)], // out of range -> dropped
  [span(6, 9), span(1, 3)], // unsorted input
]) {
  const pieces = cut(raw, spans);
  assert.equal(pieces.map((p) => p.text).join(""), raw);
  assert.ok(pieces.every((p) => p.text.length > 0));
}

// Highlighted pieces line up with the requested offsets.
const pieces = cut(raw, [span(2, 4), span(6, 9)]);
assert.deepEqual(
  pieces.filter((p) => p.spanIndex !== null).map((p) => p.text),
  ["cd", "ghi"],
);

// Clipping keeps the first span whole and trims the one that overlaps it.
assert.deepEqual(
  cut(raw, [span(2, 6), span(4, 8)])
    .filter((p) => p.spanIndex !== null)
    .map((p) => [p.spanIndex, p.text]),
  [
    [0, "cdef"],
    [1, "gh"],
  ],
);

// Fixture contract: raw.slice(start, end) === text, for sections and spans.
for (const s of MOCK_DOCUMENT.sections) {
  assert.equal(MOCK_DOCUMENT.raw.slice(s.start, s.end), s.text);
}
for (const q of ["meals", "hotel", "airfare", "receipt", "deadline", "taxi"]) {
  const { trace } = mockAsk(q);
  assert.ok(trace.spans.length > 0, `no span for "${q}"`);
  for (const s of trace.spans) {
    assert.equal(MOCK_DOCUMENT.raw.slice(s.start, s.end), s.text);
  }
}

// An unanswerable question abstains rather than inventing a citation.
const refused = mockAsk("does the policy cover pet boarding");
assert.equal(refused.citation, null);
assert.equal(refused.trace.abstained_at, "retrieval");

console.log("spans.check: ok");
