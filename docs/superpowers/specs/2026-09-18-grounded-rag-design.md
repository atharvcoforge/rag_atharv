# Grounded RAG Policy Assistant — Design

Date: 2026-09-18
Status: approved

## Problem

Answer employee questions about a six-section expense policy, citing the section that
supports each answer, and refuse when the policy does not contain the answer.

The corpus is roughly 250 words. Retrieval quality is not the hard part; **knowing when
not to answer** is. This design optimises for that.

## The three failure modes worth engineering against

1. **Near-miss abstention.** "Can I expense a rental car?" Section 4 enumerates taxi,
   rideshare, train, and public transit, and bans luxury vehicle upgrades. It never
   mentions rental cars. A dense retriever returns Section 4 at a very small cosine
   distance, and an ungated generator invents a rule. This is the dominant failure.
2. **Numeric thresholds.** "$20 lunch, do I need a receipt?" requires Section 5's $25
   threshold. Embedding models compress exact numerals poorly; lexical matching does not.
3. **Unfalsifiable citations.** Section-level citation cannot be mechanically checked. A
   model can cite Section 2 for a claim that is not in Section 2 and nothing catches it.

Every component below targets one of these, and ships only if the ablation harness shows
it moves a metric. Rejected components are recorded in the report rather than deleted
silently.

## Architecture

### Ingestion

`policy.md` is parsed on `## N. Title` headings into exactly six **section chunks**, which
are the unit of citation and satisfy the lab requirement of one chunk per numbered section.
No sentence is ever split across chunks.

Each section is additionally split into **propositions**, one per sentence, roughly twelve
in total. A proposition carries its parent section id and a `(start, end)` character span
into the raw `policy.md` bytes. Propositions are the unit of *retrieval*; sections remain
the unit of *citation*.

This matters for failure mode 1. As a single chunk, Section 4 averages "taxi, rideshare,
train, transit, luxury upgrades" into one vector that sits moderately close to almost any
transport question, including rental cars. Split into four propositions, no individual
proposition is close to "rental car", so the retrieval score gate fires instead of
returning false confidence.

Both units are embedded with a deterministic contextual prefix, the Contextual Retrieval
idea from Anthropic's 2024 work with the LLM-generated context replaced by a template
(the corpus is small enough that a template carries the same information without the
nondeterminism and cost):

```
Employee Expense Policy v2.0, Section 4 (Ground Transportation): Luxury vehicle upgrades are not reimbursable.
```

Queries are embedded with Qwen3-Embedding's instruction prefix; documents are not. The
model is trained for this asymmetry.

All vectors are L2-normalised at write time, so cosine distance reduces to `1 - dot` and
the NumPy and pgvector backends agree to floating-point tolerance.

### Storage

Postgres 17 with pgvector. One `chunks` table holding text, `vector(1024)`, and metadata
(document, version, section, section title, chunk id, kind, parent id, span offsets). An
HNSW index with `vector_cosine_ops` serves the dense query exactly as the lab specifies:

```sql
ORDER BY embedding <=> %(query_vector)s ASC LIMIT 3
```

A NumPy in-memory store implements the same interface with exact search. Both are driven
by one shared test suite, so "the fallback behaves differently" cannot silently become
true. CI uses NumPy; local development uses Postgres.

### Retrieval

1. **Dense**: pgvector HNSW, top 12.
2. **BM25**: hand-rolled, about 30 lines, over the same rows. This is what retrieves on
   "$25", "30 days", and "receipts". Implemented rather than depended on because it is
   small, exactly testable, and genuinely BM25 rather than Postgres `ts_rank_cd`'s
   approximation. Ceiling noted in code: above ~100k chunks it belongs in Postgres.
3. **Reciprocal Rank Fusion** with `k=60`, which fuses two rankings without needing to
   normalise incomparable score scales.
4. **Cross-encoder rerank**. Ollama exposes `logprobs` and `top_logprobs`, so
   Qwen3-Reranker-0.6B runs as a true cross-encoder: one forward pass per candidate,
   generating a single token, reading `P("yes")` out of the top-logprob distribution.
   This yields calibrated relevance probabilities with no PyTorch dependency.
5. **Parent expansion**: winning propositions map up to their parent sections, deduped,
   preserving best rank.

### Abstention

Two gates.

**Gate 1, pre-generation.** Abstain if the top rerank probability is below `tau`, or if
the margin between the top and second candidate is below `delta`. This catches the rental
car case before a single generation token is spent, which is also the cheapest possible
path for out-of-scope questions.

**Gate 2, post-generation.** The answer is decomposed into atomic claims. Each claim is
checked for entailment against the retrieved spans, again via `P("yes")` from the reranker.
Independently, every citation span is asserted to be a byte-exact substring of `policy.md`
at the recorded offsets. A fabricated citation is therefore structurally impossible rather
than merely discouraged.

`tau` and `delta` are fitted on a development split of the golden set to maximise F1 on
the answerable/unanswerable decision, and written to `config/thresholds.json` together
with the run that produced them. They are not hand-tuned constants.

### Generation

`qwen3:8b`, temperature 0, thinking disabled, constrained to a JSON schema by Ollama's
`format` parameter. The prompt supplies numbered evidence blocks and instructs the model
to use only those excerpts, to name the supporting section, and to emit the exact refusal
string when the excerpts are insufficient.

### Output contract

The lab's literal response shape, with all additional data namespaced under `trace` so the
deliverable stays clean:

```json
{
  "answer": "...",
  "citation": { "document": "...", "version": "...", "section": "1. Meals" },
  "retrieved_chunks": [{ "section": "1. Meals", "distance": 0.08 }],
  "trace": { "...": "..." }
}
```

A contract test validates against the lab's schema verbatim: `distance` is a JSON number,
at most three chunks, sorted ascending by distance.

### Evaluation

`eval/golden.yaml` holds roughly 45 questions in six buckets: answerable, in-domain
unanswerable, out-of-domain, adversarial (prompt injection inside the question), numeric
boundary (exactly $25, exactly 30 days, day 31), and multi-section.

Metrics: Recall@1, Recall@3, MRR, nDCG@3, citation exactness, answer correctness,
abstention precision and recall, false-answer rate on unanswerable questions, hallucinated
citation rate, and p50/p95 latency per pipeline stage.

`rag eval --ablate` sweeps configurations (dense only, plus BM25, plus rerank, plus HyDE,
plus verifier, full) and writes a markdown comparison to `reports/`. Model calls are
recorded to cassettes so tests and CI run hermetically without Ollama.

## Non-goals

- Multi-document corpora, incremental re-indexing, authentication, persistence of chat
  history. None are required and all would add surface area without moving a metric.
- LLM-generated per-chunk context. The template carries the same signal here.

## Testing strategy

Pure functions (parsing, spans, BM25, RRF, metrics) are unit tested directly. Model-facing
code is tested against recorded cassettes. Both stores run the same suite. The lab's output
schema has a dedicated contract test. Live tests are marked and excluded from CI.
