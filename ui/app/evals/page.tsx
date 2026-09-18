"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";
import { fetchEvals } from "@/lib/api";
import { cn } from "@/lib/cn";
import { SPRING } from "@/lib/motion";
import type { EvalMetrics, EvalReport } from "@/lib/types";

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

export default function EvalsPage() {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [offline, setOffline] = useState(false);
  const reduce = useReducedMotion();

  useEffect(() => {
    let live = true;
    void fetchEvals().then(({ data, offline }) => {
      if (!live) return;
      setReport(data);
      setOffline(offline);
    });
    return () => {
      live = false;
    };
  }, []);

  if (!report) {
    return (
      <div className="px-6 py-8">
        <span className="label">Loading ablation</span>
      </div>
    );
  }

  const rows: Array<[string, EvalMetrics]> = Object.entries(report).sort(
    (a, b) =>
      a[1].false_answer_rate - b[1].false_answer_rate ||
      b[1].answer_correctness - a[1].answer_correctness,
  );
  const winner = rows[0]?.[0];
  const worstFar = Math.max(...rows.map(([, m]) => m.false_answer_rate), 0.0001);

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-[1100px] px-6 py-8">
        <div className="flex items-baseline gap-3">
          <h1 className="text-[15px] text-ink">Ablation</h1>
          <span className="num text-[11px] text-ink-faint">
            {rows.length} configurations
          </span>
          {offline && (
            <span className="num border border-hairline px-1.5 py-0.5 text-[10px] tracking-[0.06em] text-ink-faint uppercase">
              fixture
            </span>
          )}
        </div>
        <p className="mt-3 max-w-[64ch] text-[13px] leading-[1.7] text-ink-dim">
          False answer rate is the headline number: the share of questions the
          system answered when it had no grounds to. Everything else is a
          tradeoff against it.
        </p>

        <div className="mt-8 border border-hairline bg-surface">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-hairline">
                <th className="label px-5 py-3 font-normal">Config</th>
                <th className="label px-4 py-3 text-right font-normal">
                  Recall@3
                </th>
                <th className="label px-4 py-3 text-right font-normal">
                  Answer correctness
                </th>
                <th className="label border-x border-hairline px-4 py-3 font-normal">
                  False answer rate
                </th>
                <th className="label px-4 py-3 text-right font-normal">
                  False refusal
                </th>
                <th className="label px-5 py-3 text-right font-normal">p50</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([name, m], i) => {
                const best = name === winner;
                return (
                  <motion.tr
                    key={name}
                    initial={reduce ? false : { opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ ...SPRING, delay: reduce ? 0 : i * 0.03 }}
                    className={cn(
                      "border-b border-hairline last:border-b-0",
                      best && "bg-white/[0.04]",
                    )}
                  >
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-2.5">
                        <span className="num text-[13px] text-ink">{name}</span>
                        {best && (
                          <span className="num border border-hairline-strong px-1.5 py-0.5 text-[9px] tracking-[0.1em] text-ink-dim uppercase">
                            best
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="num px-4 py-4 text-right text-[13px] text-ink-dim">
                      {pct(m.recall_at_3)}
                    </td>
                    <td className="num px-4 py-4 text-right text-[13px] text-ink-dim">
                      {pct(m.answer_correctness)}
                    </td>

                    {/* Headline metric: bigger type plus a proportional bar. */}
                    <td className="border-x border-hairline px-4 py-4">
                      <div className="flex items-center gap-3">
                        <span
                          className={cn(
                            "num w-[62px] text-[18px] leading-none",
                            best ? "text-ink" : "text-ink-dim",
                          )}
                        >
                          {pct(m.false_answer_rate)}
                        </span>
                        <span className="relative h-[4px] flex-1 bg-white/[0.05]">
                          <motion.span
                            className={cn(
                              "absolute inset-y-0 left-0",
                              best ? "bg-white/55" : "bg-white/25",
                            )}
                            initial={reduce ? false : { width: 0 }}
                            animate={{
                              width: `${(m.false_answer_rate / worstFar) * 100}%`,
                            }}
                            transition={reduce ? { duration: 0 } : SPRING}
                          />
                        </span>
                      </div>
                    </td>

                    <td className="num px-4 py-4 text-right text-[13px] text-ink-dim">
                      {pct(m.false_refusal_rate)}
                    </td>
                    <td className="num px-5 py-4 text-right text-[13px] text-ink-dim">
                      {m.p50_ms.toFixed(0)}
                      <span className="text-ink-faint">ms</span>
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <p className="num mt-4 text-[11px] leading-[1.8] text-ink-faint">
          sorted by false answer rate, ascending · ties broken by answer
          correctness
        </p>
      </div>
    </div>
  );
}
