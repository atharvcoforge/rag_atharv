# Latency

Measured 2026-09-18 on the shipping `full` config: Postgres/pgvector, `qwen3:8b` for
rerank and generation, models warmed through `POST /api/warm` first.

## What the user waits for

Five questions over `GET /api/ask/stream`, timed from the request to each event:

| question | first `stage` | first `status` | `done` | retrieve | generate | verify |
|---|---|---|---|---|---|---|
| wine with dinner | 7.9ms | 993ms | 4000ms | 988ms | 2827ms | 180ms |
| meal allowance | 937ms | 1929ms | 4624ms | 1924ms | 2512ms | 182ms |
| receipt for $20 lunch | 9.2ms | 993ms | 4086ms | 986ms | 2920ms | 174ms |
| first-class airfare | 7.5ms | 983ms | 3739ms | 978ms | 2581ms | 175ms |
| rental car | 7.2ms | 978ms | 3928ms | 975ms | 2781ms | 169ms |

**Time to first event: 7.9ms at p50.** It was previously equal to the total, because the
endpoint ran `ask()` to completion and then replayed the timings it had already
collected. The target was 200ms.

**Time to the answer: 4.0s at p50**, which is unchanged, and honestly so — see the
reranker section. The generate and verify columns are new (`trace.generate_ms`,
`trace.verify_ms`); previously only retrieval was instrumented and the other three
seconds could only be inferred by subtraction.

The 937ms outlier on the second question is a model swap inside Ollama: the embedding
model had been evicted while the 8B answered the first question. `OLLAMA_KEEP_ALIVE`
(30m by default now) is what holds that down, and a second endpoint via
`RERANK_OLLAMA_URL` is what would remove it.

## Reranker candidates

`rag rerank-probe --model X`. Every case is a policy question scored against the
sentence that governs it and the sentence most likely to be mistaken for it. A
candidate must not invert any case, must stay unconfident on questions the policy
cannot answer, and must be cheaper than the 153ms/call the 8B costs.

| model | worst margin | unanswerable high | ms/call | verdict |
|---|---|---|---|---|
| **qwen3:8b** (incumbent) | +0.169 | 0.032 | 167 | passes on quality; it *is* the baseline |
| qwen3:1.7b | -0.000 | 0.945 | 42 | FAIL — ties first-class against business-class |
| qwen3:0.6b | +0.068 | 0.956 | 21 | FAIL — 0.956 on "can I expense a rental car?" |
| mistral:7b | -0.001 | 0.000 | 165 | FAIL — inverts, and no faster |

The small models are four to eight times faster and would have taken roughly 900ms off
every answer. They are also the reason the abstention numbers exist: both score the
rental-car question above 0.94 against a sentence about luxury vehicle upgrades, which
is precisely the fabrication the calibrated `tau` gate is there to stop. **No promotion.
`RERANK_MODEL` stays `qwen3:8b`**, and the rerank stage keeps costing its second.

`rerank_candidates` was left at 6. Cutting it to 4 would save about 330ms of a 4000ms
answer, against a documented note that 6 is what covers every golden question surviving
fusion. Not worth the recall risk for 8%.

## Quality gate

The latency work changes when bytes are sent, not what is computed, and the golden set
confirms it. The same 48 questions, same fitted thresholds, run twice — once on this
branch and once on a clean worktree of the previous commit:

| metric | previous commit | this branch |
|---|---|---|
| recall@1 / recall@3 / MRR | 0.9 / 1.0 / 0.95 | 0.9 / 1.0 / 0.95 |
| citation accuracy | 0.9333 | 0.9333 |
| answer correctness | 0.80 | 0.80 |
| decision accuracy | 0.9167 | 0.9167 |
| false answer | 0.1667 | 0.1667 |
| false refusal | 0.0333 | 0.0333 |

Identical to the digit, so the change ships.

**These are not the numbers in `ablation.md`.** That report recorded 0.9667 answer
correctness and 0.1111 false answers for `full`; today the same code on the same
questions produces 0.80 and 0.1667. The regression predates this branch — it reproduces
exactly on the previous commit — so it is drift in the local model stack rather than
anything the streaming work did. It is recorded here rather than quietly overwritten in
`ablation.md`, because a sweep re-run under a changed environment is a different
experiment, not a correction. Worth re-running `rag eval --ablate` and re-fitting `tau`
before trusting the old table again.
