# Ablation

48 golden questions. Thresholds fitted on a 24/24 split (`tau = 2e-05`). Lower is
better for false answer, false refusal, and both latencies.

> Re-running `full` on 2026-09-18 reproduced the retrieval rows but not the answer rows
> (0.80 correctness, 0.167 false answers against the 0.967 and 0.111 below). The drift
> predates the streaming work and reproduces on the commit before it; the measurements
> are in [`latency.md`](latency.md). Re-run the sweep and re-fit `tau` before quoting
> the answer columns here.

The shipping config is **`full`**: dense + BM25 + RRF + cross-encoder rerank + quote
verification. It is the only row that cuts false answers below 12% without raising
false refusals.

| config | Recall@3 | Recall@1 | MRR | Citation | Answer | False answer | False refusal | Decision | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | 1.0 | 0.8667 | 0.9333 | 0.9667 | 0.9 | 0.3333 | 0.0 | 0.875 | 2794 | 3709 |
| `dense+bm25` | 1.0 | 0.8333 | 0.9111 | 0.9667 | 0.9 | 0.3889 | 0.0 | 0.8542 | 2801 | 3632 |
| `dense+bm25+rerank` | 1.0 | 0.9 | 0.95 | 0.9667 | 0.9667 | 0.2222 | 0.0 | 0.9167 | 7838 | 25477 |
| **`full`** | 1.0 | 0.9 | 0.95 | 0.9667 | 0.9667 | **0.1111** | 0.0 | **0.9583** | 4072 | 8427 |
| `full-no-bm25` | 1.0 | 0.9 | 0.95 | 0.9333 | 0.9333 | 0.1111 | 0.0333 | 0.9375 | 3815 | 4936 |
| `full+hyde` | 1.0 | 0.9 | 0.95 | 0.9667 | 0.9333 | 0.1667 | 0.0 | 0.9375 | 5653 | 6677 |
| `bm25-only` | 0.9667 | 0.8667 | 0.9111 | 0.9667 | 0.9 | 0.4444 | 0.0 | 0.8333 | 2950 | 3727 |

`dense+bm25+rerank` p50/p95 is inflated: that pass ran while other work was sharing
Ollama (`-np 1`). `full` is the cleaner latency read for the same stages plus
verification.

## What shipped, and why

- **Dense retrieval.** Recall@3 is already 1.0. It is not optional; BM25 alone drops
  recall@3 to 0.97 and false-answers 44% of unanswerable questions.
- **BM25.** Alone it is worse than dense. Inside `full` it buys answer correctness
  (0.967 vs 0.933) and kills the false refusals that `full-no-bm25` introduces
  (0.033). Kept for those two deltas, not for recall.
- **Cross-encoder rerank.** Cuts false answers 0.389 to 0.222. Costs about a second
  of serial yes/no scoring. Paid on purpose.
- **Quote verification (gate 2).** Cuts false answers 0.222 to 0.111 with no extra
  false refusals. That is the cheapest remaining win.

## Tried and rejected

- **HyDE.** `full+hyde` raised false answers (0.111 to 0.167), dropped answer
  correctness (0.967 to 0.933), and added ~1.6s of dense-stage latency. Deleted from
  the pipeline. The numbers stay in the table so the rejection is reproducible.
- **Qwen3-Reranker-0.6B GGUF** (`dengcao/Qwen3-Reranker-0.6B:Q8_0`). Uniform
  logprobs at -11.93 for every prompt, including "The capital of France is".
  Unusable. `qwen3:8b` is the cross-encoder instead.
- **qwen3:0.6b / 1.7b as rerankers.** 21ms and 51ms per call, negative class
  separation. The 1.7B model scored "rental car" against "Luxury vehicle upgrades
  are not reimbursable" at 0.945. Capacity is the product here, not a preference.
- **Thread-pool rerank.** Eight candidates: 1190ms sequential, 1288ms with four
  threads. Ollama serialises same-model requests. Pool removed.
- **Asking whether an excerpt "contains the answer".** Positive min 0.0002 against
  negative max 0.0003. Asking which excerpt *governs* the question is what made
  rerank and generation actually work.
- **Fitting tau on retrieval scores alone.** Credits refusals the quote check would
  have made anyway. End-to-end calibration on a held-out split is what is in
  `config/thresholds.json`.
