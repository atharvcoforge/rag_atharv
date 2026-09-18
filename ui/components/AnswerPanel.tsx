"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Quote } from "lucide-react";
import AbstentionState from "./AbstentionState";
import ConfidenceDial from "./ConfidenceDial";
import { SPRING } from "@/lib/motion";
import type { AskResponse } from "@/lib/types";

type Props = {
  status: "idle" | "running" | "done";
  /** Tokens received so far; the full answer once done. */
  text: string;
  result: AskResponse | null;
  offline: boolean;
  onCitationClick: () => void;
};

export default function AnswerPanel({
  status,
  text,
  result,
  offline,
  onCitationClick,
}: Props) {
  const reduce = useReducedMotion();
  const abstainedAt = result?.trace.abstained_at ?? null;

  if (status === "idle") {
    return (
      <div className="border border-hairline bg-surface px-5 py-10">
        <p className="max-w-[44ch] text-[13px] leading-[1.7] text-ink-faint">
          Every answer is quoted from the policy on the right, or the system
          declines. There is no third option.
        </p>
      </div>
    );
  }

  if (abstainedAt && result) {
    return <AbstentionState at={abstainedAt} trace={result.trace} />;
  }

  const citation = result?.citation ?? null;

  return (
    <div className="border border-hairline bg-surface">
      <header className="flex items-center gap-3 border-b border-hairline px-5 py-3">
        <span className="label">Answer</span>
        {offline && (
          <span className="num border border-hairline px-1.5 py-0.5 text-[10px] tracking-[0.06em] text-ink-faint uppercase">
            fixture
          </span>
        )}
        {result && (
          <span className="num ml-auto text-[11px] text-ink-faint">
            {result.trace.total_ms.toFixed(1)} ms
          </span>
        )}
      </header>

      <div className="px-5 py-6">
        <p className="max-w-[58ch] text-[17px] leading-[1.6] text-ink">
          {text}
          {status === "running" && (
            <motion.span
              className="ml-0.5 inline-block h-[15px] w-[7px] translate-y-[1px] bg-ink align-baseline"
              animate={reduce ? {} : { opacity: [1, 0.15, 1] }}
              transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
            />
          )}
        </p>

        <AnimatePresence>
          {citation && (
            <motion.button
              initial={reduce ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={SPRING}
              onClick={onCitationClick}
              className="group mt-6 flex items-center gap-2 border border-hairline px-3 py-2 text-left transition-colors hover:border-accent/60"
            >
              <span className="h-[13px] w-[2px] bg-accent" aria-hidden />
              <span className="text-[12.5px] text-ink-dim group-hover:text-ink">
                {citation.document}{" "}
                <span className="num">{citation.version}</span>
                <span className="mx-1.5 text-ink-faint">·</span>
                <span className="num">§{citation.section}</span>
              </span>
            </motion.button>
          )}
        </AnimatePresence>

        {result?.trace.governing_rule && (
          <motion.figure
            initial={reduce ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={SPRING}
            className="mt-4 flex gap-3 border-l border-hairline-strong pl-3"
          >
            <Quote size={13} strokeWidth={1.5} className="mt-1 shrink-0 text-ink-faint" />
            <blockquote className="num max-w-[62ch] text-[12px] leading-[1.75] text-ink-dim">
              {result.trace.governing_rule}
            </blockquote>
          </motion.figure>
        )}
      </div>

      {result && (
        <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-6 border-t border-hairline px-5 py-4">
          <ConfidenceDial value={result.trace.confidence} />
          <div>
            <div className="label">Retrieval score</div>
            <div className="num mt-2 text-[15px] leading-none text-ink-dim">
              {result.trace.retrieval_score.toFixed(3)}
            </div>
          </div>
          <div>
            <div className="label">Margin</div>
            <div className="num mt-2 text-[15px] leading-none text-ink-dim">
              {result.trace.margin.toFixed(3)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
