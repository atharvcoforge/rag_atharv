# Grounded RAG Policy Assistant

Span-grounded answers over a six-section employee expense policy. Local models only.
The system cites the sentence it used, and refuses when the policy does not govern the
question.

Shipping retrieval config is **`full`**: dense + BM25 + RRF + cross-encoder rerank +
quote verification. HyDE was measured and deleted. Numbers live in
[`reports/ablation.md`](reports/ablation.md); streaming latency in
[`reports/latency.md`](reports/latency.md).

## What it does

```
policy.md
    -> heading parse (6 sections, never split a sentence)
    -> proposition split (atomic rules, char spans into the file)
    -> contextual prefix + qwen3-embedding:0.6b (1024-d, L2-normalised)
    -> Postgres/pgvector HNSW  or  NumPy fallback

question
    -> instruct-prefixed query embedding
    -> dense top-k  +  BM25  ->  RRF  ->  qwen3:8b yes/no rerank
    -> Gate 1: retrieval score vs calibrated tau
    -> parent-section expansion
    -> qwen3:8b JSON answer that must quote a governing sentence
    -> Gate 2: quote is a whitespace-tolerant substring of retrieved evidence
    -> exact-match answer cache (.cache/answers.sqlite)
    -> {answer, citation, retrieved_chunks, trace}
```

`GET /api/ask/stream` reports that run live over SSE: each retrieval stage as it closes
(first one at ~8ms), then `generating` / `verifying` / `cached`, then the payload. No
answer text is streamed before the gates have accepted it.

Lab contract is the top-level JSON. Distances are numbers, at most three chunks,
sorted ascending. Everything extra is under `trace`.

## Results (48 golden questions)

| config | Answer | False answer | False refusal | Decision | p50 |
|---|---|---|---|---|---|
| dense | 0.90 | 0.33 | 0.00 | 0.88 | 2.8s |
| dense+bm25+rerank | 0.97 | 0.22 | 0.00 | 0.92 | 7.8s* |
| **full** | **0.97** | **0.11** | **0.00** | **0.96** | **4.1s** |
| full+hyde (rejected) | 0.93 | 0.17 | 0.00 | 0.94 | 5.7s |

\*rerank-only pass ran under Ollama contention; `full` is the fairer latency.

Recall@3 is 1.0 for every dense config. The hard metric is false answers on
questions the policy cannot settle (rental car, parking, parental leave).

> Re-running `full` later reproduced retrieval rows but not answer rows (0.80
> correctness / 0.17 false answers). That drift is recorded in
> [`reports/latency.md`](reports/latency.md). Re-run `rag eval --ablate` and
> `rag calibrate` before treating the answer columns as current.

## Run it

Prerequisites: Python 3.12 (`uv`), Ollama with `qwen3-embedding:0.6b` and `qwen3:8b`,
Docker Desktop for Postgres. Node 22+ is only needed for the UI.

```bash
cp .env.example .env
docker compose up -d
uv sync
uv run rag index                 # full corpus (sections + propositions, contextual embeds)
uv run rag ask "Can I expense wine with dinner?"
uv run rag eval --ablate         # writes reports/ablation.json + .md
uv run rag calibrate             # fits tau into config/thresholds.json
uv run uvicorn ragpolicy.api:app --reload --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/api/warm   # load both models before the first user
```

Identical questions (case/whitespace folded) are answered from `.cache/answers.sqlite`
instead of re-running the 8B — expect milliseconds on a retry. Set `ANSWER_CACHE=off`
for cold timing. Re-index clears the cache.

### CLI

| command | purpose |
|---|---|
| `rag index [--lab]` | embed + store corpus (`--lab` = six bare sections) |
| `rag ask [--config NAME] [--json] QUESTION` | answer one question |
| `rag eval [--ablate] [--config NAME] [--limit N]` | golden set / ablation sweep |
| `rag calibrate [--config NAME]` | fit abstention thresholds |
| `rag lab-six [--out PATH]` | Mini RAG Lab six questions → JSON report |
| `rag rerank-probe [--model M] [--baseline-ms MS]` | gate a cheaper reranker |

Configs: `dense`, `dense+bm25`, `dense+bm25+rerank`, `full`, `full-no-bm25`,
`bm25-only`, `lab`.

### Swapping the reranker

Six serial cross-encoder calls are about a second of every answer. Three smaller
scorers have already been rejected for scoring near misses as highly as the
governing rule. `rag rerank-probe` exits non-zero when a candidate inverts a case,
stays confident on an unanswerable question, or is not actually faster:

```bash
uv run rag rerank-probe --model qwen3:1.7b
```

Only after it passes is `RERANK_MODEL` worth changing — and then only if the golden set
holds. Point `RERANK_OLLAMA_URL` at a second Ollama process to keep the scorer and the
generator resident at the same time.

### Lab contract path

Production default is `full`. The Mini RAG Lab brief wants six bare section chunks and
plain cosine `LIMIT 3`. That lives behind `--config lab` / `--lab` and does not replace
the measured pipeline:

```bash
uv run rag index --lab
uv run rag ask --config lab "Can I book first-class airfare?"
uv run rag lab-six               # writes reports/lab-six-questions.json
```

`lab` ask builds an ephemeral six-section store so it does not wipe a full Postgres index.

### UI

```bash
cd ui && npm install && npm run dev
```

Open http://localhost:3000. Split view: ask + live pipeline trace on the left, policy
document on the right with the cited span highlighted. `/evals` renders the ablation
table from `reports/ablation.json`.

### HTTP API

| method | path | role |
|---|---|---|
| `POST` | `/api/ask` | `{question, config?}` → lab response contract |
| `GET` | `/api/ask/stream` | SSE: `stage` / `status` / `done` (or `error`) |
| `GET` | `/api/document` | raw policy + section offsets |
| `POST` | `/api/index` | re-embed and store |
| `POST` | `/api/warm` | load rerank + gen models |
| `GET` | `/api/health` | store backend + model names |
| `GET` | `/api/eval/latest` | last ablation JSON (or `{}`) |

### Tests

Hermetic suite: stubs Ollama with `httpx.MockTransport`, uses NumPy store (Postgres
tests skip when Docker is down). Coverage gate is **95%**.

```bash
uv run pytest                # includes --cov-fail-under=95
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

CI on GitHub Actions runs the same hermetic checks.

## Layout

| path | role |
|---|---|
| `policy.md` | the corpus |
| `src/ragpolicy/ingest.py` | section + proposition parse, span invariant |
| `src/ragpolicy/models.py` | Ollama client, embedding cache, yes/no logprobs |
| `src/ragpolicy/store.py` | pgvector + NumPy, shared test suite |
| `src/ragpolicy/retrieve.py` | BM25, RRF, rerank, parent expansion |
| `src/ragpolicy/answer.py` | generation, quote location, two gates |
| `src/ragpolicy/pipeline.py` | wiring, ablation flags, ask/index |
| `src/ragpolicy/response_cache.py` | exact-match ask cache |
| `src/ragpolicy/evaluate.py` | golden set, metrics, calibration, lab-six |
| `src/ragpolicy/rerank_probe.py` | separation gate for candidate rerankers |
| `src/ragpolicy/api.py` | FastAPI + live SSE |
| `src/ragpolicy/cli.py` | `rag` entry point |
| `src/ragpolicy/config.py` | env → `Settings` |
| `eval/golden.yaml` | 48 questions, six buckets |
| `config/thresholds.json` | fitted `tau` |
| `reports/` | ablation, latency, lab-six outputs |
| `ui/` | Next.js ask UI + `/evals` |
| `.env.example` | Ollama, store, cache knobs |

Design notes:
[`docs/superpowers/specs/2026-09-18-grounded-rag-design.md`](docs/superpowers/specs/2026-09-18-grounded-rag-design.md),
[`docs/superpowers/specs/2026-09-18-rag-latency-design.md`](docs/superpowers/specs/2026-09-18-rag-latency-design.md).
