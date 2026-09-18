"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useRef } from "react";
import { cn } from "@/lib/cn";
import { SPRING } from "@/lib/motion";
import { cut } from "@/lib/spans";
import type { PolicyDoc, Span } from "@/lib/types";

type Props = {
  doc: PolicyDoc | null;
  spans: Span[];
  /** Index into `spans` that is currently the cited one. */
  activeIndex: number;
  /** Bump to re-flash + re-scroll the active span without changing it. */
  flashKey: number;
};

export default function PolicyDocument({
  doc,
  spans,
  activeIndex,
  flashKey,
}: Props) {
  const reduce = useReducedMotion();
  const activeRef = useRef<HTMLElement>(null);
  const pieces = useMemo(() => (doc ? cut(doc.raw, spans) : []), [doc, spans]);

  useEffect(() => {
    const el = activeRef.current;
    if (!el) return;
    el.scrollIntoView({
      block: "center",
      behavior: reduce ? "auto" : "smooth",
    });
  }, [activeIndex, flashKey, reduce, pieces]);

  if (!doc) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="label">Loading document</span>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex items-baseline gap-3 border-b border-hairline px-6 py-3.5">
        <span className="label">Source</span>
        <span className="text-[13px] text-ink">{doc.document}</span>
        <span className="num text-[11px] text-ink-faint">{doc.version}</span>
        <span className="num ml-auto text-[11px] text-ink-faint">
          {doc.raw.length} chars · {doc.sections.length} sections
        </span>
      </header>

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <pre className="num max-w-[76ch] text-[12.5px] leading-[1.85] whitespace-pre-wrap text-ink-dim">
          {pieces.map((p, i) =>
            p.spanIndex === null ? (
              <span key={i}>{p.text}</span>
            ) : (
              <motion.mark
                key={`${i}-${p.spanIndex === activeIndex ? flashKey : "s"}`}
                ref={p.spanIndex === activeIndex ? activeRef : undefined}
                initial={
                  reduce
                    ? false
                    : { backgroundColor: "rgba(229,160,13,0.45)", opacity: 0.6 }
                }
                animate={{
                  backgroundColor:
                    p.spanIndex === activeIndex
                      ? "rgba(229,160,13,0.18)"
                      : "rgba(255,255,255,0.05)",
                  opacity: 1,
                }}
                transition={reduce ? { duration: 0 } : SPRING}
                className={cn(
                  "box-decoration-clone px-0.5 py-[3px]",
                  p.spanIndex === activeIndex
                    ? "text-ink shadow-[inset_0_-1px_0_0_var(--color-accent)]"
                    : "text-ink-dim",
                )}
              >
                {p.text}
              </motion.mark>
            ),
          )}
        </pre>
      </div>
    </div>
  );
}
