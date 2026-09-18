"use client";

import { AnimatePresence, LayoutGroup, motion, useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";
import { SPRING } from "@/lib/motion";
import type { Candidate, Stage } from "@/lib/types";

type Order = "distance" | "score";

type Props = {
  stages: Stage[];
  candidates: Candidate[];
  totalMs: number | null;
  running: boolean;
  /** Increments once per question, to replay the rerank reorder. */
  runId: number;
};

export default function PipelineTrace({
  stages,
  candidates,
  totalMs,
  running,
  runId,
}: Props) {
  const reduce = useReducedMotion();
  const [order, setOrder] = useState<Order>("score");
  const [seenRun, setSeenRun] = useState(runId);

  // On each new result, snap back to retrieval order during render...
  if (seenRun !== runId) {
    setSeenRun(runId);
    setOrder(candidates.length && !reduce ? "distance" : "score");
  }

  // ...then let the rerank physically move the rows into their final places.
  useEffect(() => {
    if (order !== "distance" || reduce) return;
    const t = setTimeout(() => setOrder("score"), 700);
    return () => clearTimeout(t);
  }, [order, reduce, runId]);

  const sum = stages.reduce((a, s) => a + s.ms, 0) || 1;
  const rows = [...candidates].sort((a, b) =>
    order === "distance" ? a.distance - b.distance : b.score - a.score,
  );
  const topScore = Math.max(...candidates.map((c) => c.score), 0.0001);

  return (
    <section className="border border-hairline bg-surface">
      <header className="flex items-center gap-3 border-b border-hairline px-5 py-3">
        <span className="label">Pipeline</span>
        {running && (
          <motion.span
            className="num text-[11px] text-ink-faint"
            animate={{ opacity: [0.35, 1, 0.35] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
          >
            running
          </motion.span>
        )}
        <span className="num ml-auto text-[11px] text-ink-dim">
          {totalMs !== null ? `${totalMs.toFixed(1)} ms` : `${sum.toFixed(1)} ms`}
        </span>
      </header>

      {/* Proportional stage bar. */}
      <div className="px-5 pt-5 pb-4">
        <div className="flex h-[7px] w-full gap-[2px] overflow-hidden bg-white/[0.03]">
          <AnimatePresence initial={false}>
            {stages.map((s, i) => (
              <motion.div
                key={s.name}
                initial={{ flexGrow: 0.0001, opacity: 0 }}
                animate={{ flexGrow: s.ms, opacity: 1 }}
                exit={{ flexGrow: 0.0001, opacity: 0 }}
                transition={reduce ? { duration: 0 } : SPRING}
                style={{
                  flexBasis: 0,
                  backgroundColor: `rgba(255,255,255,${0.14 + i * 0.09})`,
                }}
                title={`${s.name} · ${s.ms.toFixed(1)} ms`}
              />
            ))}
          </AnimatePresence>
          {stages.length === 0 && (
            <div className="flex-1 bg-white/[0.04]" />
          )}
        </div>

        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2">
          {stages.length === 0 ? (
            <span className="label">awaiting a question</span>
          ) : (
            stages.map((s) => (
              <motion.div
                key={s.name}
                initial={reduce ? false : { opacity: 0, y: 3 }}
                animate={{ opacity: 1, y: 0 }}
                transition={SPRING}
                className="flex items-baseline gap-2"
              >
                <span className="label">{s.name}</span>
                <span className="num text-[12px] text-ink">
                  {s.ms.toFixed(1)}
                  <span className="text-ink-faint">ms</span>
                </span>
              </motion.div>
            ))
          )}
        </div>
      </div>

      {/* Candidates — the rows move when the ordering changes. */}
      <div className="flex items-center gap-3 border-t border-hairline px-5 py-2.5">
        <span className="label">Candidates</span>
        <span className="num text-[11px] text-ink-faint">{candidates.length}</span>
        <div className="ml-auto flex border border-hairline">
          {(["distance", "score"] as Order[]).map((o) => (
            <button
              key={o}
              onClick={() => setOrder(o)}
              disabled={!candidates.length}
              className={cn(
                "px-2.5 py-1 text-[10px] tracking-[0.08em] uppercase transition-colors",
                order === o
                  ? "bg-white/[0.07] text-ink"
                  : "text-ink-faint hover:text-ink-dim",
              )}
            >
              {o === "distance" ? "retrieval" : "reranked"}
            </button>
          ))}
        </div>
      </div>

      <LayoutGroup id="candidates">
        <ol className="min-h-[52px]">
          {rows.length === 0 && (
            <li className="px-5 py-4">
              <span className="text-[12px] text-ink-faint">
                No candidates yet.
              </span>
            </li>
          )}
          {rows.map((c, rank) => (
            <motion.li
              key={c.chunk_id}
              layout
              layoutId={`candidate-${c.chunk_id}`}
              transition={reduce ? { duration: 0 } : SPRING}
              className="grid grid-cols-[22px_1fr_68px_120px] items-center gap-3 border-t border-hairline bg-surface px-5 py-2.5"
            >
              <span className="num text-[11px] text-ink-faint">
                {String(rank + 1).padStart(2, "0")}
              </span>
              <span className="truncate text-[12.5px] text-ink">{c.section}</span>
              <span
                className="num text-right text-[12px] text-ink-dim"
                title="vector distance"
              >
                {c.distance.toFixed(3)}
              </span>
              <span className="flex items-center gap-2">
                <span className="relative h-[3px] flex-1 bg-white/[0.06]">
                  <motion.span
                    className="absolute inset-y-0 left-0 bg-white/40"
                    initial={false}
                    animate={{ width: `${(c.score / topScore) * 100}%` }}
                    transition={reduce ? { duration: 0 } : SPRING}
                  />
                </span>
                <span className="num w-[38px] text-right text-[12px] text-ink">
                  {c.score.toFixed(2)}
                </span>
              </span>
            </motion.li>
          ))}
        </ol>
      </LayoutGroup>

      <div className="flex items-center justify-between border-t border-hairline px-5 py-2.5">
        <span className="label">rank by</span>
        <span className="num text-[11px] text-ink-faint">
          {order === "distance"
            ? "ascending vector distance"
            : "descending cross-encoder score"}
        </span>
      </div>
    </section>
  );
}
