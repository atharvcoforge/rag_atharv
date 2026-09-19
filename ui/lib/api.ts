import { MOCK_DOCUMENT, MOCK_EVALS, mockAsk } from "./mock";
import type { AskResponse, EvalReport, Phase, PolicyDoc, Stage } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

/** Everything the UI gets back is tagged with whether it came from fixtures. */
export type Sourced<T> = { data: T; offline: boolean };

const TIMEOUT_MS = 8000;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    signal: AbortSignal.timeout(TIMEOUT_MS),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export async function fetchDocument(): Promise<Sourced<PolicyDoc>> {
  try {
    return { data: await getJSON<PolicyDoc>("/api/document"), offline: false };
  } catch {
    return { data: MOCK_DOCUMENT, offline: true };
  }
}

export async function fetchEvals(): Promise<Sourced<EvalReport>> {
  try {
    return { data: await getJSON<EvalReport>("/api/eval/latest"), offline: false };
  } catch {
    return { data: MOCK_EVALS, offline: true };
  }
}

export async function ask(question: string): Promise<Sourced<AskResponse>> {
  try {
    const res = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(TIMEOUT_MS * 4),
    });
    if (!res.ok) throw new Error(`/api/ask -> ${res.status}`);
    return { data: (await res.json()) as AskResponse, offline: false };
  } catch {
    return { data: mockAsk(question), offline: true };
  }
}

export type StreamHandlers = {
  onStage: (stage: Stage) => void;
  onPhase: (phase: Phase) => void;
  /** Only the offline fixture replay produces tokens; the live stream does not. */
  onToken: (token: string) => void;
  onDone: (result: AskResponse, offline: boolean) => void;
};

/** Tokens may arrive as a JSON string, a JSON object, or bare text. */
function readToken(data: string): string {
  try {
    const parsed: unknown = JSON.parse(data);
    if (typeof parsed === "string") return parsed;
    if (parsed && typeof parsed === "object") {
      const o = parsed as Record<string, unknown>;
      for (const k of ["token", "text", "delta", "content"]) {
        if (typeof o[k] === "string") return o[k];
      }
    }
  } catch {
    /* not JSON — treat as literal text */
  }
  return data;
}

/**
 * Subscribe to the SSE stream. Falls back to POST /api/ask, and then to
 * fixtures, so the pane always resolves to a terminal state.
 *
 * The live stream reports progress, not prose: retrieval stages land in
 * milliseconds and the generation and verification phases are announced as they
 * start, but no answer text arrives until the gates have accepted it.
 * Returns a cancel function.
 */
export function askStream(question: string, h: StreamHandlers): () => void {
  let cancelled = false;
  let settled = false;
  let es: EventSource | null = null;

  const url = `${API_BASE}/api/ask/stream?question=${encodeURIComponent(question)}`;

  const fallback = async () => {
    if (cancelled || settled) return;
    settled = true;
    const { data, offline } = await ask(question);
    if (cancelled) return;
    if (offline) {
      await replayMock(data, h, () => cancelled);
      if (cancelled) return;
      h.onDone(data, true);
    } else {
      h.onDone(data, false);
    }
  };

  try {
    es = new EventSource(url);
    es.addEventListener("stage", (e) => {
      try {
        h.onStage(JSON.parse((e as MessageEvent<string>).data) as Stage);
      } catch {
        /* ignore malformed stage frames */
      }
    });
    es.addEventListener("status", (e) => {
      try {
        const { phase } = JSON.parse((e as MessageEvent<string>).data) as {
          phase: Phase;
        };
        h.onPhase(phase);
      } catch {
        /* ignore malformed status frames */
      }
    });
    es.addEventListener("token", (e) => {
      h.onToken(readToken((e as MessageEvent<string>).data));
    });
    // Catches both a dropped connection and the server's own `error` event: either
    // way the pane owes the user a terminal state, and POST /api/ask can still give one.
    es.addEventListener("error", () => {
      es?.close();
      void fallback();
    });
    es.addEventListener("done", (e) => {
      settled = true;
      es?.close();
      try {
        h.onDone(JSON.parse((e as MessageEvent<string>).data) as AskResponse, false);
      } catch {
        void fallback();
      }
    });
  } catch {
    void fallback();
  }

  return () => {
    cancelled = true;
    es?.close();
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Paces the fixture response so offline mode still shows the pipeline run. */
async function replayMock(
  result: AskResponse,
  h: StreamHandlers,
  isCancelled: () => boolean,
) {
  for (const stage of result.trace.stages) {
    await sleep(Math.min(stage.ms, 120));
    if (isCancelled()) return;
    h.onStage(stage);
  }
  h.onPhase("generating");
  for (const word of result.answer.match(/\S+\s*/g) ?? []) {
    await sleep(22);
    if (isCancelled()) return;
    h.onToken(word);
  }
}
