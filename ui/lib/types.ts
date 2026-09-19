export type Citation = {
  document: string;
  version: string;
  section: string;
};

export type RetrievedChunk = {
  section: string;
  distance: number;
};

export type Span = {
  chunk_id: string;
  start: number;
  end: number;
  text: string;
};

export type Candidate = {
  chunk_id: string;
  section: string;
  distance: number;
  score: number;
};

export type Stage = {
  name: string;
  ms: number;
};

/** What the backend is doing after retrieval, announced as it starts. */
export type Phase = "generating" | "verifying" | "cached";

/** The gates the pipeline can stop at, in pipeline order. */
export const ABSTENTION_GATES = [
  "retrieval",
  "quote",
  "model",
  "verification",
  "generation",
  "span",
] as const;

export type AbstentionGate = (typeof ABSTENTION_GATES)[number];

export type Trace = {
  abstained_at: string | null;
  confidence: number;
  retrieval_score: number;
  margin: number;
  governing_rule: string;
  spans: Span[];
  candidates: Candidate[];
  stages: Stage[];
  total_ms: number;
  generate_ms: number;
  verify_ms: number;
};

export type AskResponse = {
  answer: string;
  citation: Citation | null;
  retrieved_chunks: RetrievedChunk[];
  trace: Trace;
};

export type DocumentSection = {
  chunk_id: string;
  section: string;
  section_title: string;
  text: string;
  start: number;
  end: number;
};

export type PolicyDoc = {
  raw: string;
  document: string;
  version: string;
  sections: DocumentSection[];
};

export type EvalMetrics = {
  recall_at_3: number;
  recall_at_1: number;
  mrr: number;
  citation_accuracy: number;
  answer_correctness: number;
  false_answer_rate: number;
  false_refusal_rate: number;
  decision_accuracy: number;
  p50_ms: number;
  p95_ms: number;
};

export type EvalReport = Record<string, EvalMetrics>;
