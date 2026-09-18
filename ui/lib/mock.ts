import type {
  AskResponse,
  Candidate,
  DocumentSection,
  EvalReport,
  PolicyDoc,
} from "./types";

/**
 * Offline fixtures. Used only when the backend at NEXT_PUBLIC_API_BASE is
 * unreachable, so the UI still renders something truthful in shape.
 */

const RAW = `# Employee Expense Policy — Version 2.0

## 1. Meals
Employees may claim up to $65 per day for meals while traveling overnight.
Alcohol is not reimbursable.

## 2. Hotels
Hotels are reimbursable up to $225 per night.
A manager must approve higher rates before booking.

## 3. Airfare
Employees must purchase economy airfare.
Business-class airfare requires written approval from a vice president.

## 4. Ground Transportation
Taxi, rideshare, train, and public-transit expenses are reimbursable.
Luxury vehicle upgrades are not reimbursable.

## 5. Receipts
Receipts are required for individual expenses of $25 or more.

## 6. Submission Deadline
Expense reports must be submitted within 30 days after travel ends.
`;

/** Offsets are derived from RAW so raw.slice(start, end) === text, always. */
function sliceSections(raw: string): DocumentSection[] {
  const heads = [...raw.matchAll(/^## (.+)$/gm)];
  return heads.map((m, i) => {
    const start = m.index ?? 0;
    const end = i + 1 < heads.length ? (heads[i + 1].index ?? raw.length) : raw.length;
    const heading = m[1].trim();
    return {
      chunk_id: `chunk-${i + 1}`,
      section: heading,
      section_title: heading.replace(/^\d+\.\s*/, ""),
      text: raw.slice(start, end),
      start,
      end,
    };
  });
}

export const MOCK_DOCUMENT: PolicyDoc = {
  raw: RAW,
  document: "Employee Expense Policy",
  version: "v2.0",
  sections: sliceSections(RAW),
};

function sectionByIndex(i: number) {
  return MOCK_DOCUMENT.sections[i];
}

/** Locate a sentence inside RAW and return its character span. */
function spanOf(sectionIndex: number, needle: string) {
  const s = sectionByIndex(sectionIndex);
  const at = RAW.indexOf(needle, s.start);
  return { chunk_id: s.chunk_id, start: at, end: at + needle.length, text: needle };
}

/**
 * Candidates are deliberately ordered so `distance` rank differs from `score`
 * rank — that disagreement is what the rerank animation is showing.
 */
function candidates(winner: number): Candidate[] {
  const order = [winner, (winner + 3) % 6, (winner + 1) % 6, (winner + 4) % 6];
  return order.map((idx, rank) => {
    const s = sectionByIndex(idx);
    return {
      chunk_id: s.chunk_id,
      section: s.section,
      distance: 0.21 + rank * 0.07 + (rank === 1 ? -0.03 : 0),
      score: rank === 0 ? 0.94 : 0.62 - rank * 0.17,
    };
  });
}

const STAGES = [
  { name: "dense", ms: 41.2 },
  { name: "bm25", ms: 12.8 },
  { name: "fuse", ms: 3.1 },
  { name: "rerank", ms: 88.6 },
  { name: "expand", ms: 9.4 },
];

function grounded(
  sectionIndex: number,
  answer: string,
  quote: string,
): AskResponse {
  const s = sectionByIndex(sectionIndex);
  const cands = candidates(sectionIndex);
  return {
    answer,
    citation: {
      document: MOCK_DOCUMENT.document,
      version: MOCK_DOCUMENT.version,
      section: s.section,
    },
    retrieved_chunks: cands.map((c) => ({
      section: c.section,
      distance: c.distance,
    })),
    trace: {
      abstained_at: null,
      confidence: 0.91,
      retrieval_score: 0.79,
      margin: 0.32,
      governing_rule: quote,
      spans: [spanOf(sectionIndex, quote)],
      candidates: cands,
      stages: STAGES,
      total_ms: STAGES.reduce((a, b) => a + b.ms, 0),
    },
  };
}

function abstained(gate: string): AskResponse {
  const cands = candidates(0).map((c, i) => ({
    ...c,
    distance: 0.63 + i * 0.05,
    score: 0.19 - i * 0.04,
  }));
  return {
    answer: "",
    citation: null,
    retrieved_chunks: cands.map((c) => ({
      section: c.section,
      distance: c.distance,
    })),
    trace: {
      abstained_at: gate,
      confidence: 0.12,
      retrieval_score: 0.21,
      margin: 0.03,
      governing_rule: "",
      spans: [],
      candidates: cands,
      stages: STAGES.slice(0, 4),
      total_ms: STAGES.slice(0, 4).reduce((a, b) => a + b.ms, 0),
    },
  };
}

const CANNED: Array<[RegExp, AskResponse]> = [
  [
    /meal|food|dinner|lunch|per diem|alcohol|drink/i,
    grounded(
      0,
      "Up to $65 per day for meals, and only when the travel includes an overnight stay. Alcohol is excluded.",
      "Employees may claim up to $65 per day for meals while traveling overnight.",
    ),
  ],
  [
    /hotel|lodging|night|room/i,
    grounded(
      1,
      "Hotels are reimbursable up to $225 per night. Anything above that needs manager approval before booking.",
      "Hotels are reimbursable up to $225 per night.",
    ),
  ],
  [
    /air|flight|fly|business class|economy/i,
    grounded(
      2,
      "Economy only. Business class requires written approval from a vice president.",
      "Employees must purchase economy airfare.",
    ),
  ],
  [
    /taxi|uber|lyft|rideshare|train|transit|car|ground/i,
    grounded(
      3,
      "Taxi, rideshare, train and public transit are reimbursable. Luxury vehicle upgrades are not.",
      "Taxi, rideshare, train, and public-transit expenses are reimbursable.",
    ),
  ],
  [
    /receipt/i,
    grounded(
      4,
      "A receipt is required for any individual expense of $25 or more.",
      "Receipts are required for individual expenses of $25 or more.",
    ),
  ],
  [
    /deadline|submit|late|30 day|report/i,
    grounded(
      5,
      "Expense reports are due within 30 days after the travel ends.",
      "Expense reports must be submitted within 30 days after travel ends.",
    ),
  ],
];

export function mockAsk(question: string): AskResponse {
  const hit = CANNED.find(([re]) => re.test(question));
  return hit ? hit[1] : abstained("retrieval");
}

export const MOCK_EVALS: EvalReport = {
  dense: {
    recall_at_3: 1.0,
    recall_at_1: 0.8667,
    mrr: 0.9333,
    citation_accuracy: 0.9667,
    answer_correctness: 0.9,
    false_answer_rate: 0.3333,
    false_refusal_rate: 0.0,
    decision_accuracy: 0.875,
    p50_ms: 2793.6,
    p95_ms: 3708.8,
  },
  "dense+bm25": {
    recall_at_3: 1.0,
    recall_at_1: 0.8333,
    mrr: 0.9111,
    citation_accuracy: 0.9667,
    answer_correctness: 0.9,
    false_answer_rate: 0.3889,
    false_refusal_rate: 0.0,
    decision_accuracy: 0.8542,
    p50_ms: 2801.3,
    p95_ms: 3632.1,
  },
  "dense+bm25+rerank": {
    recall_at_3: 1.0,
    recall_at_1: 0.9,
    mrr: 0.95,
    citation_accuracy: 0.9667,
    answer_correctness: 0.9667,
    false_answer_rate: 0.2222,
    false_refusal_rate: 0.0,
    decision_accuracy: 0.9167,
    p50_ms: 7837.7,
    p95_ms: 25476.6,
  },
  full: {
    recall_at_3: 1.0,
    recall_at_1: 0.9,
    mrr: 0.95,
    citation_accuracy: 0.9667,
    answer_correctness: 0.9667,
    false_answer_rate: 0.1111,
    false_refusal_rate: 0.0,
    decision_accuracy: 0.9583,
    p50_ms: 4071.6,
    p95_ms: 8426.7,
  },
  "full-no-bm25": {
    recall_at_3: 1.0,
    recall_at_1: 0.9,
    mrr: 0.95,
    citation_accuracy: 0.9333,
    answer_correctness: 0.9333,
    false_answer_rate: 0.1111,
    false_refusal_rate: 0.0333,
    decision_accuracy: 0.9375,
    p50_ms: 3815.1,
    p95_ms: 4935.5,
  },
  "bm25-only": {
    recall_at_3: 0.9667,
    recall_at_1: 0.8667,
    mrr: 0.9111,
    citation_accuracy: 0.9667,
    answer_correctness: 0.9,
    false_answer_rate: 0.4444,
    false_refusal_rate: 0.0,
    decision_accuracy: 0.8333,
    p50_ms: 2950.2,
    p95_ms: 3726.6,
  },
};

export const EXAMPLE_QUESTIONS = [
  "Can I expense a $70 dinner if I stayed overnight?",
  "My hotel was $280 a night — is that covered?",
  "Do I need a receipt for a $22 taxi?",
  "Can I fly business class to Berlin?",
  "Does the policy cover pet boarding while I travel?",
];
