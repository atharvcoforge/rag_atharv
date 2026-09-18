"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnswerPanel from "@/components/AnswerPanel";
import AskBar from "@/components/AskBar";
import PipelineTrace from "@/components/PipelineTrace";
import PolicyDocument from "@/components/PolicyDocument";
import { askStream, fetchDocument } from "@/lib/api";
import { EXAMPLE_QUESTIONS } from "@/lib/mock";
import type { AskResponse, PolicyDoc, Stage } from "@/lib/types";

type Status = "idle" | "running" | "done";

export default function Home() {
  const [doc, setDoc] = useState<PolicyDoc | null>(null);
  const [docOffline, setDocOffline] = useState(false);

  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [text, setText] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [stages, setStages] = useState<Stage[]>([]);
  const [offline, setOffline] = useState(false);

  const [runId, setRunId] = useState(0);
  const [activeSpan, setActiveSpan] = useState(0);
  const [flashKey, setFlashKey] = useState(0);

  const cancel = useRef<(() => void) | null>(null);

  useEffect(() => {
    let live = true;
    void fetchDocument().then(({ data, offline }) => {
      if (!live) return;
      setDoc(data);
      setDocOffline(offline);
    });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => () => cancel.current?.(), []);

  const onAsk = useCallback((q: string) => {
    cancel.current?.();
    setQuestion(q);
    setStatus("running");
    setText("");
    setResult(null);
    setStages([]);
    setActiveSpan(0);

    cancel.current = askStream(q, {
      onStage: (s) =>
        setStages((prev) =>
          prev.some((p) => p.name === s.name) ? prev : [...prev, s],
        ),
      onToken: (t) => setText((prev) => prev + t),
      onDone: (r, isOffline) => {
        setResult(r);
        setText(r.answer);
        setStages(r.trace.stages);
        setOffline(isOffline);
        setStatus("done");
        setActiveSpan(0);
        setRunId((n) => n + 1);
        setFlashKey((n) => n + 1);
      },
    });
  }, []);

  const spans = result?.trace.spans ?? [];
  const degraded = docOffline || (status === "done" && offline);

  return (
    <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,47fr)_minmax(0,53fr)]">
      {/* Left: question and answer. */}
      <div className="scroll-thin min-h-0 overflow-y-auto border-hairline lg:border-r">
        {degraded && (
          <div className="border-b border-hairline bg-raised px-6 py-2">
            <span className="num text-[11px] text-ink-faint">
              backend unreachable — rendering local fixtures
            </span>
          </div>
        )}

        <div className="flex flex-col gap-6 px-6 py-6">
          <AskBar
            onAsk={onAsk}
            busy={status === "running"}
            examples={EXAMPLE_QUESTIONS}
          />

          {question && (
            <div>
              <span className="label">Question</span>
              <p className="mt-2 text-[13px] leading-[1.6] text-ink-dim">
                {question}
              </p>
            </div>
          )}

          <AnswerPanel
            status={status}
            text={text}
            result={result}
            offline={offline}
            onCitationClick={() => {
              setActiveSpan(0);
              setFlashKey((n) => n + 1);
            }}
          />

          <PipelineTrace
            stages={stages}
            candidates={result?.trace.candidates ?? []}
            totalMs={result?.trace.total_ms ?? null}
            running={status === "running"}
            runId={runId}
          />
        </div>
      </div>

      {/* Right: the source of truth. */}
      <div className="min-h-0 overflow-hidden">
        <PolicyDocument
          doc={doc}
          spans={spans}
          activeIndex={activeSpan}
          flashKey={flashKey}
        />
      </div>
    </div>
  );
}
