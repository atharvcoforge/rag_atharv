# Grounded RAG Policy Assistant

Span-grounded answers over a six-section employee expense policy. Local models only.
The system cites the sentence it used, and refuses when the policy does not govern the
question.

Shipping retrieval config is `full`: dense + BM25 + RRF + cross-encoder rerank + quote
verification. HyDE was measured and deleted. Numbers live in
[`reports/ablation.md`](reports/ablation.md).

## What it does

```
policy.md
    -> heading parse (6 sections, never split a sentence)
    -> proposition split (10 atomic rules, char spans into the file)
    -> contextual prefix + qwen3-embedding:0.6b (1024-d, L2-normalised)
    -> Postgres/pgvector HNSW  or  NumPy fallback

question
    -> instruct-prefixed query embedding
    -> dense top-k  +  BM25  ->  RRF  ->  qwen3:8b yes/no rerank
    -> Gate 1: retrieval score vs calibrated tau
    -> parent-section expansion
    -> qwen3:8b JSON answer that must quote a governing sentence
    -> Gate 2: quote is a whitespace-tolerant substring of retrieved evidence
    -> {answer, citation, retrieved_chunks, trace}
```

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

## Run it

Prerequisites: Python 3.12 (`uv`), Ollama with `qwen3-embedding:0.6b` and `qwen3:8b`,
Docker Desktop for Postgres. Node 22 is only needed for the UI
(`export PATH="$HOME/.local/node/bin:$PATH"`).

```bash
cp .env.example .env
docker compose up -d
uv sync
uv run rag index
uv run rag ask "Can I expense wine with dinner?"
uv run rag eval --ablate          # writes reports/ablation.json
uv run uvicorn ragpolicy.api:app --reload --port 8000
```

UI, in another terminal:

```bash
export PATH="$HOME/.local/node/bin:$PATH"
cd ui && npm install && npm run dev
```

Open http://localhost:3000. The right pane is the policy; the cited sentence
highlights in place. `/evals` renders the ablation table.

Tests (no Ollama, no Docker):

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

CI on GitHub Actions is that same hermetic suite.

## Layout

| path | role |
|---|---|
| `policy.md` | the corpus |
| `src/ragpolicy/ingest.py` | section + proposition parse, span invariant |
| `src/ragpolicy/models.py` | Ollama client, embedding cache, yes/no logprobs |
| `src/ragpolicy/store.py` | pgvector + NumPy, shared test suite |
| `src/ragpolicy/retrieve.py` | BM25, RRF, rerank, parent expansion |
| `src/ragpolicy/answer.py` | generation, quote location, two gates |
| `src/ragpolicy/pipeline.py` | wiring and ablation flags |
| `src/ragpolicy/evaluate.py` | golden set, metrics, calibration |
| `src/ragpolicy/api.py` | FastAPI + SSE |
| `eval/golden.yaml` | 48 questions, six buckets |
| `config/thresholds.json` | fitted `tau` |
| `ui/` | Next.js split view |

Design notes: [`docs/superpowers/specs/2026-09-18-grounded-rag-design.md`](docs/superpowers/specs/2026-09-18-grounded-rag-design.md).
