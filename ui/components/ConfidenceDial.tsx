"use client";

import { motion, useReducedMotion } from "framer-motion";
import { SPRING } from "@/lib/motion";

/** 270° arc, opening at the bottom. */
const SWEEP = 270;
const R = 21;
const C = 26;

function arc(fromDeg: number, toDeg: number) {
  const pt = (deg: number) => {
    const rad = ((deg - 90) * Math.PI) / 180;
    return [C + R * Math.cos(rad), C + R * Math.sin(rad)];
  };
  const [x0, y0] = pt(fromDeg);
  const [x1, y1] = pt(toDeg);
  const large = Math.abs(toDeg - fromDeg) > 180 ? 1 : 0;
  return `M ${x0} ${y0} A ${R} ${R} 0 ${large} 1 ${x1} ${y1}`;
}

const TRACK = arc(-135, -135 + SWEEP);

export default function ConfidenceDial({
  value,
  label = "Confidence",
}: {
  value: number;
  label?: string;
}) {
  const reduce = useReducedMotion();
  const v = Math.max(0, Math.min(1, value));

  return (
    <div className="flex items-center gap-3">
      <svg width={52} height={52} viewBox="0 0 52 52" aria-hidden className="shrink-0">
        <path
          d={TRACK}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={3}
          strokeLinecap="butt"
        />
        <motion.path
          d={TRACK}
          fill="none"
          stroke="var(--color-ink)"
          strokeWidth={3}
          strokeLinecap="butt"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: v }}
          transition={reduce ? { duration: 0 } : SPRING}
        />
      </svg>
      <div>
        <div className="label">{label}</div>
        <div className="num mt-1.5 text-[19px] leading-none text-ink">
          {v.toFixed(2)}
        </div>
      </div>
    </div>
  );
}
