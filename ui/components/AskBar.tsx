"use client";

import { motion } from "framer-motion";
import { CornerDownLeft } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/cn";
import { SPRING } from "@/lib/motion";

type Props = {
  onAsk: (question: string) => void;
  busy: boolean;
  examples: string[];
};

export default function AskBar({ onAsk, busy, examples }: Props) {
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        input.current?.focus();
        input.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const submit = () => {
    const q = value.trim();
    if (q && !busy) onAsk(q);
  };

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-3 border bg-surface px-4 transition-colors",
          focused ? "border-hairline-strong" : "border-hairline",
        )}
      >
        <span className="num select-none text-[11px] text-ink-faint">ASK</span>
        <input
          ref={input}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="What does the policy say about…"
          aria-label="Ask a question about the policy"
          spellCheck={false}
          className="h-14 flex-1 bg-transparent text-[15px] leading-none text-ink outline-none placeholder:text-ink-faint focus-visible:outline-none"
        />
        {value.trim() ? (
          <motion.button
            initial={{ opacity: 0, x: 4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={SPRING}
            onClick={submit}
            disabled={busy}
            className="flex items-center gap-1.5 border border-hairline px-2.5 py-1.5 text-[11px] tracking-[0.08em] text-ink-dim uppercase hover:border-hairline-strong hover:text-ink disabled:opacity-40"
          >
            Run <CornerDownLeft size={12} strokeWidth={1.75} />
          </motion.button>
        ) : (
          <kbd className="num select-none border border-hairline px-1.5 py-1 text-[10px] text-ink-faint">
            ⌘K
          </kbd>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {examples.map((q) => (
          <button
            key={q}
            onClick={() => {
              setValue(q);
              input.current?.focus();
            }}
            className="border border-hairline px-2.5 py-1.5 text-left text-[12px] leading-none text-ink-dim transition-colors hover:border-hairline-strong hover:text-ink"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
