import type { Transition } from "framer-motion";

/** One spring for the whole app, so every state change feels the same. */
export const SPRING: Transition = {
  type: "spring",
  stiffness: 400,
  damping: 35,
};

export const FADE: Transition = { duration: 0.18, ease: [0.2, 0, 0, 1] };
