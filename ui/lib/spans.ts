import type { Span } from "./types";

export type Piece = { text: string; spanIndex: number | null };

/**
 * Cuts `raw` into plain and highlighted pieces using the API's character
 * offsets. Out-of-range spans are dropped and overlapping spans are clipped,
 * so concatenating every piece always reproduces `raw` exactly.
 */
export function cut(raw: string, spans: Span[]): Piece[] {
  const ordered = spans
    .map((s, i) => ({ ...s, i }))
    .filter((s) => s.end > s.start && s.start >= 0 && s.end <= raw.length)
    .sort((a, b) => a.start - b.start);

  const pieces: Piece[] = [];
  let at = 0;
  for (const s of ordered) {
    const start = Math.max(s.start, at);
    if (start >= s.end) continue;
    if (start > at) pieces.push({ text: raw.slice(at, start), spanIndex: null });
    pieces.push({ text: raw.slice(start, s.end), spanIndex: s.i });
    at = s.end;
  }
  if (at < raw.length) pieces.push({ text: raw.slice(at), spanIndex: null });
  return pieces;
}
