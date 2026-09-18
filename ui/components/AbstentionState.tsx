"use client";

import { motion } from "framer-motion";
import { ABSTENTION_GATES, type Trace } from "@/lib/types";
import { cn } from "@/lib/cn";
import { SPRING } from "@/lib/motion";

/** What each gate actually checked, phrased as a decision rather than a fault. */
const GATE_COPY: Record<string, { gate: string; reason: string }> = {
  retrieval: {
    gate: "retrieval",
    reason: "Nothing in the document scored above the retrieval floor.",
  },
  quote: {
    gate: "quote",
    reason: "No verbatim sentence in the policy supports an answer.",
  },
  model: {
    gate: "model",
    reason: "The model declined to answer from the retrieved text alone.",
  },
  verification: {
    gate: "verification",
    reason: "The draft answer did not verify against the sentence it cited.",
  },
  generation: {
    gate: "generation",
    reason: "Generation produced nothing that could be grounded in the source.",
  },
  span: {
    gate: "span",
    reason: "The cited span could not be located in the source document.",
  },
};

export default function AbstentionState({
  at,
  trace,
}: {
  at: string;
  trace: Trace;
}) {
  const copy = GATE_COPY[at] ?? {
    gate: at,
    reason: "The pipeline stopped before it could ground an answer.",
  };
  const stoppedIndex = ABSTENTION_GATES.indexOf(
    at as (typeof ABSTENTION_GATES)[number],
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={SPRING}
      className="border border-hairline bg-surface"
    >
      <div className="border-b border-hairline px-5 py-3">
        <span className="label">Abstained</span>
      </div>

      <div className="px-5 py-6">
        <p className="max-w-[42ch] text-[19px] leading-[1.5] text-ink">
          No rule in this policy governs that question.
        </p>
        <p className="mt-3 max-w-[54ch] text-[13px] leading-[1.7] text-ink-dim">
          {copy.reason} Answering anyway would mean inventing a rule, so the
          system stopped.
        </p>
      </div>

      {/* Which gate stopped it, in pipeline order. */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-2 border-t border-hairline px-5 py-3.5">
        {ABSTENTION_GATES.map((g, i) => {
          const passed = stoppedIndex >= 0 && i < stoppedIndex;
          const stopped = g === at;
          return (
            <span key={g} className="flex items-center gap-2">
              {i > 0 && <span className="text-[10px] text-ink-faint">/</span>}
              <span
                className={cn(
                  "num text-[11px] tracking-[0.06em] uppercase",
                  stopped && "text-ink",
                  passed && "text-ink-faint line-through decoration-white/20",
                  !stopped && !passed && "text-white/15",
                )}
              >
                {g}
              </span>
            </span>
          );
        })}
        <span className="ml-auto num text-[11px] text-ink-faint">
          stopped at {copy.gate}
        </span>
      </div>

      <dl className="grid grid-cols-3 border-t border-hairline">
        {[
          ["Retrieval score", trace.retrieval_score],
          ["Margin", trace.margin],
          ["Confidence", trace.confidence],
        ].map(([k, v], i) => (
          <div
            key={k as string}
            className={cn("px-5 py-3.5", i > 0 && "border-l border-hairline")}
          >
            <dt className="label">{k as string}</dt>
            <dd className="num mt-2 text-[15px] leading-none text-ink-dim">
              {(v as number).toFixed(2)}
            </dd>
          </div>
        ))}
      </dl>
    </motion.div>
  );
}
