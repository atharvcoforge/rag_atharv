# Perceived Latency — Design

Date: 2026-09-18
Status: approved

## Problem

`full` answers a question in about 4.1 seconds at p50 and 8.4 at p95
(`reports/ablation.json`). The retrieval half is not the cost. Dense search, BM25,
fusion, and section expansion together measure under 20ms; pgvector with an HNSW index
over a 250-word policy has nothing left to give.

The cost is the 8B model, paid three times per question:

| Where | Cost | Source |
|---|---|---|
| Rerank, six serial yes/no calls | ~1.0s | `rerank` stage p50 |
| Generation, JSON-schema constrained | ~2.5s | `dense` p50 minus a 5ms retrieve |
| Entailment verification | ~0.2-1s | residual, worse when recovery runs |

Two things follow. First, no amount of index tuning moves this number. Second, a user
watching a spinner for four seconds is not told any of the above: the current stream
endpoint runs `ask()` to completion and *then* replays stage timings and word-by-word
tokens, so the first byte and the last byte arrive together.

This design optimises **perceived** latency. The goal is not a sub-second answer; it is
that the interface stops lying about what it is doing, and that the one stage paying an
unnecessary tax stops paying it.

## What is explicitly not the goal

Millisecond end-to-end answers. That would mean deleting verification or moving to a
small generator, and the ablation says what those cost: without the quote and entailment
gates, false answers go from 11% to 22% or 39%. The abstention behaviour is the product.
Latency work that spends it is a regression sold as a feature.

## Success criteria

| Metric | Target |
|---|---|
| Time to first `stage` event (warm) | ≤ 200ms |
| Time to `done` (warm p50) | ~1-2s |
| `rerank` stage p50 | well under 1s, if a dedicated reranker passes its probe |
| Golden-set quality | within the epsilon below, or the change does not ship |

## 1. Streaming contract

`GET /api/ask/stream` becomes a live stream over a running pipeline rather than a replay
of a finished one. Four event types:

| Event | Emitted | Payload |
|---|---|---|
| `stage` | as each retrieval stage completes | `{name, ms}` |
| `status` | entering generation, then verification | `{phase}` |
| `done` | after the gates decide | the full `/api/ask` payload |
| `error` | the pipeline raised | `{message}` |

Two rules constrain this.

**No answer text before the gates.** Streaming generated tokens as they arrive would
show the user a sentence that quote-location or entailment may still refuse, and
retracting a visible answer is worse than waiting for it. The generated JSON is also not
readable mid-stream. So `done` stays atomic, and honest progress is what streams. The
fake per-word `token` replay is deleted from the live path; the UI's offline fixture
mode keeps its own pacing, because there nothing is at stake.

**`POST /api/ask` does not change.** The eval harness, the CLI, and the lab contract all
read it. It stays a single blocking JSON response.

Emission needs the pipeline to publish progress while it runs. `Pipeline.ask` takes an
optional `on_event(name, data)` listener; `retrieve` reports each `Stage` as it closes,
and `Answerer.answer` reports `generating` before the generate call and `verifying`
before the first entailment call. The API runs `ask` on a worker thread and drains a
queue, which is what turns those callbacks into flushed SSE frames.

## 2. Model specialisation

The reranker and the generator are the same 8B weights today, which costs twice.

*Capacity*: six serial cross-encoder calls at ~153ms each. The ablation already shows a
thread pool does not help — Ollama serialises requests to one model, and eight
candidates took 1190ms sequentially against 1288ms across four threads.

*Contention*: with `-np 1`, a rerank call and a generate call queue behind each other on
the same slot, which is why the `dense+bm25+rerank` sweep row reads 7.8s.

The fix is a genuinely separate scorer:

- `RERANK_MODEL` — the cross-encoder, defaulting to `qwen3:8b` until something beats it.
- `RERANK_OLLAMA_URL` — the endpoint it runs on, defaulting to the primary. Pointing it
  at a second Ollama process keeps a small reranker and an 8B generator resident at once
  instead of thrashing one slot.
- `OLLAMA_KEEP_ALIVE` — sent with every request so both models stay loaded between
  questions. This is the keep-alive; there is no separate warming daemon.
- `POST /api/warm` — loads both models on demand, so a deploy can pay the cold start
  instead of the first user.

Routing is by role, not by caller. `yes_probability(..., role="rerank")` goes to the
rerank endpoint with the rerank model; `role="entailment"` stays on the primary endpoint
with the *generation* model. Verification must not inherit a small reranker's judgement:
the entailment gate is what holds false answers at 11%, and it is graded on the hardest
distinctions in the corpus.

### The probe gate

This project has already tried and rejected three cheap rerankers:
`dengcao/Qwen3-Reranker-0.6B:Q8_0` returned uniform -11.93 logprobs for every input
including "The capital of France is"; `qwen3:0.6b` and `qwen3:1.7b` were fast but scored
"rental car" at 0.945 against a sentence about luxury vehicle upgrades.

So a candidate reranker is not adopted because it is small. `rag rerank-probe` scores a
fixed set of question / governing-sentence / near-miss-sentence triples drawn from the
policy's known traps and fails the model unless all three hold:

1. **Not degenerate** — the spread across all scores exceeds 0.05. This is the check the
   0.6B GGUF failed.
2. **Separated** — the lowest governing score exceeds the highest near-miss score by at
   least 0.2. This is the check the 1.7B failed.
3. **Faster** — mean latency per call is below the 153ms the 8B costs. A reranker that
   separates but is not faster has no reason to exist here.

## 3. Quality gate

Latency changes are measured against the current `full` row, on the same 48 goldens with
the same fitted thresholds. A change ships only if:

| Metric | Baseline | Ships if |
|---|---|---|
| False answer rate | 0.1111 | ≤ 0.1311 |
| False refusal rate | 0.0 | ≤ 0.02 |
| Decision accuracy | 0.9583 | ≥ 0.9383 |
| Answer correctness | 0.9667 | ≥ 0.9367 |
| Citation accuracy | 0.9667 | ≥ 0.9367 |

Streaming and keep-alive cannot move these — they change when bytes are sent, not what
is computed — so the gate exists for the reranker swap and for any reduction in
`rerank_candidates`. If a candidate fails, the default stays `qwen3:8b` and the latency
win is whatever streaming and keep-alive delivered.

Because the answer half of the budget was previously invisible (stage timings covered
retrieval only), `trace` gains `generate_ms` and `verify_ms`. A regression in the
generate/verify split is now readable from any single response rather than inferred by
subtracting stages from a total.

## Testing

Hermetic, as the rest of the suite is: the SSE ordering test drives a fake pipeline, the
endpoint-routing tests use `httpx.MockTransport` with two recorded base URLs, and the
probe's pass/fail logic is a pure function tested on synthetic scores. CI continues to
run without Ollama or Postgres. The reranker promotion itself is not a test; it is a
measurement, and it happens against live models before any default changes.
