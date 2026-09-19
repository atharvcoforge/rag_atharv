# RAG Perceived Latency Implementation Plan

**Goal:** Cut perceived wait (first event in milliseconds) and make the answer half of
the latency budget measurable, without weakening the abstention gates.

**Spec:** [`../specs/2026-09-18-rag-latency-design.md`](../specs/2026-09-18-rag-latency-design.md)

**Outcome:** shipped. First event went from 4s (the whole answer) to 7.9ms at p50. Time
to the answer is unchanged at 4.0s, because no candidate reranker passed its probe.
Measurements in [`../../../reports/latency.md`](../../../reports/latency.md).

## Tasks

1. **Design doc and answer-phase timers.** `trace.generate_ms` and `trace.verify_ms` in
   `answer.py`, charged in one place each so verification cost stays attributable.
2. **Live SSE.** `Pipeline.ask(question, on_event=...)` publishes each retrieval stage
   as it closes and each answer phase as it starts; `api.py` runs the pipeline on a
   worker thread and drains a queue into `stage` / `status` / `done` / `error` frames.
   `done` stays atomic — no answer text before the gates. `POST /api/ask` unchanged.
3. **Dual Ollama client.** `RERANK_OLLAMA_URL` and `OLLAMA_KEEP_ALIVE`; scoring routes
   by role, so rerank can live on its own endpoint while entailment stays on the
   generation model. `POST /api/warm` loads both.
4. **Reranker probe.** `rag rerank-probe` fails a candidate that inverts any case, is
   confident on an unanswerable question, or is not faster than the 8B.
5. **UI.** Consumes `status`; tokens are optional and only the offline fixture produces
   them; the answer pane says what the backend is doing while it waits.
6. **Gate.** Golden set run on this branch and on the previous commit: identical to the
   digit. Probe run on four models: no promotion, `RERANK_MODEL` stays `qwen3:8b`.

## What this did not do

Sub-second answers. That would mean cutting verification or the 8B, and both are load
bearing for the false-answer rate. The rerank second and the generate 2.5s are still
there, now visible in the trace rather than inferred.
