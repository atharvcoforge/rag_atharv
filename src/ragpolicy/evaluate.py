"""Golden-set evaluation, ablation sweep, and threshold calibration.

This is the module that decides what ships. A stage stays in the pipeline when the
ablation shows it moving a number, and gets deleted when it does not.

The metric that matters most is ``false_answer_rate``: the share of questions the policy
cannot answer where the system answered anyway. Retrieval metrics can look excellent
while that number is terrible, which is exactly how confidently wrong RAG demos happen.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ragpolicy.answer import REFUSAL, Thresholds
from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.pipeline import ABLATIONS, Pipeline, RetrievalConfig

GOLDEN_PATH = REPO_ROOT / "eval" / "golden.yaml"
REPORTS = REPO_ROOT / "reports"
THRESHOLDS_PATH = REPO_ROOT / "config" / "thresholds.json"


def _cold(settings: Settings) -> Settings:
    """Eval timings must not read a warm answer cache from a previous UI session."""
    return replace(settings, answer_cache=False)


#: Buckets where the correct behaviour is to refuse.
REFUSAL_BUCKETS = frozenset({"unanswerable", "out_of_domain", "adversarial"})


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    bucket: str
    question: str
    section: str | None = None
    must_include: tuple[str, ...] = ()
    must_exclude: tuple[str, ...] = ()

    @property
    def should_refuse(self) -> bool:
        return self.bucket in REFUSAL_BUCKETS


@dataclass
class CaseResult:
    case: Case
    answer: str
    refused: bool
    cited_section: str | None
    retrieved_sections: list[str]
    retrieval_score: float
    latency_ms: float
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def correct_decision(self) -> bool:
        """Did the system make the right answer-or-refuse call?"""
        return self.refused == self.case.should_refuse

    @property
    def retrieval_hit(self) -> bool | None:
        """Was the gold section retrieved at all? ``None`` when there is no gold."""
        if self.case.section is None:
            return None
        return self.case.section in self.retrieved_sections

    @property
    def rank_of_gold(self) -> int | None:
        if self.case.section is None or self.case.section not in self.retrieved_sections:
            return None
        return self.retrieved_sections.index(self.case.section) + 1

    @property
    def citation_correct(self) -> bool | None:
        if self.case.should_refuse or self.case.section is None:
            return None
        return self.cited_section == self.case.section

    @property
    def content_correct(self) -> bool | None:
        """Did the answer contain what it must, and avoid what it must not?"""
        if self.case.should_refuse:
            return not self.case.must_exclude or not any(
                bad.lower() in self.answer.lower() for bad in self.case.must_exclude
            )
        if self.refused:
            return False
        lowered = self.answer.lower()
        if any(bad.lower() in lowered for bad in self.case.must_exclude):
            return False
        return all(good.lower() in lowered for good in self.case.must_include)


def load_cases(path: Path = GOLDEN_PATH) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Case(
            id=entry["id"],
            bucket=entry["bucket"],
            question=entry["question"],
            section=str(entry["section"]) if entry.get("section") else None,
            must_include=tuple(entry.get("must_include", ())),
            must_exclude=tuple(entry.get("must_exclude", ())),
        )
        for entry in raw
    ]


def evaluate_case(pipeline: Pipeline, case: Case) -> CaseResult:
    started = time.perf_counter()
    response = pipeline.ask(case.question)
    elapsed = (time.perf_counter() - started) * 1000

    return CaseResult(
        case=case,
        answer=response.answer,
        refused=response.answer.strip() == REFUSAL,
        cited_section=(response.citation or {}).get("section", "").split(".")[0] or None,
        retrieved_sections=[c["section"].split(".")[0] for c in response.retrieved_chunks],
        retrieval_score=float(response.trace.get("retrieval_score", 0.0)),
        latency_ms=elapsed,
        stages=list(response.trace.get("stages", [])),
    )


def summarise(results: Sequence[CaseResult]) -> dict[str, Any]:
    answerable = [r for r in results if not r.case.should_refuse]
    refusable = [r for r in results if r.case.should_refuse]

    hits = [r for r in answerable if r.retrieval_hit is not None]
    ranks = [r.rank_of_gold for r in hits if r.rank_of_gold is not None]

    def share(values: Sequence[bool]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    latencies = sorted(r.latency_ms for r in results)
    stage_totals: dict[str, list[float]] = {}
    for result in results:
        for stage in result.stages:
            stage_totals.setdefault(stage["name"], []).append(stage["ms"])

    return {
        "n": len(results),
        "recall_at_1": share([r.rank_of_gold == 1 for r in hits]),
        "recall_at_3": share([r.retrieval_hit is True for r in hits]),
        "mrr": round(sum(1 / r for r in ranks) / len(hits), 4) if hits else 0.0,
        "citation_accuracy": share(
            [r.citation_correct for r in answerable if r.citation_correct is not None]
        ),
        "answer_correctness": share(
            [r.content_correct for r in answerable if r.content_correct is not None]
        ),
        "decision_accuracy": share([r.correct_decision for r in results]),
        "abstention_recall": share([r.refused for r in refusable]),
        "false_answer_rate": share([not r.refused for r in refusable]),
        "false_refusal_rate": share([r.refused for r in answerable]),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1) if latencies else 0.0,
        "stage_p50_ms": {
            name: round(statistics.median(values), 1)
            for name, values in sorted(stage_totals.items())
        },
    }


def run_suite(
    settings: Settings,
    config: RetrievalConfig,
    name: str,
    thresholds: Thresholds | None = None,
    limit: int = 0,
    quiet: bool = False,
) -> tuple[dict[str, Any], list[CaseResult]]:
    cases = load_cases()
    if limit:
        cases = cases[:limit]

    pipeline = Pipeline(_cold(settings), config=config, thresholds=thresholds or _load_thresholds())
    results = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        result = evaluate_case(pipeline, case)
        results.append(result)
        # Always emit progress. A sweep takes minutes and a silent process is
        # indistinguishable from a hung one.
        mark = "ok  " if result.correct_decision else "MISS"
        elapsed = time.perf_counter() - started
        eta = (elapsed / index) * (len(cases) - index)
        print(
            f"  [{index:>2}/{len(cases)}] {mark} {case.id:<26} "
            f"{result.latency_ms:6.0f}ms  eta {eta:4.0f}s",
            flush=True,
        )

    metrics = summarise(results)
    if not quiet:
        print(f"\n{name}")
        for key, value in metrics.items():
            if key != "stage_p50_ms":
                print(f"  {key:22} {value}")
        print(f"  stages {metrics['stage_p50_ms']}")
    return metrics, results


def run_ablation(settings: Settings, limit: int = 0) -> dict[str, dict[str, Any]]:
    # Resume from whatever a previous run already finished. A full sweep is ~17 minutes
    # of model calls and should never have to start over.
    table: dict[str, dict[str, Any]] = {}
    checkpoint = REPORTS / "ablation.json"
    if checkpoint.exists():
        table = json.loads(checkpoint.read_text(encoding="utf-8"))

    for position, (name, config) in enumerate(ABLATIONS.items(), start=1):
        if name == "lab":
            # Lab is a contract adapter, not an ablation row.
            continue
        if name in table:
            print(f"=== [{position}/{len(ABLATIONS)}] {name} (cached)", flush=True)
            continue
        print(f"\n=== [{position}/{len(ABLATIONS)}] {name}", flush=True)
        metrics, _ = run_suite(settings, config, name, limit=limit, quiet=True)
        table[name] = metrics
        print(
            f"  recall@3 {metrics['recall_at_3']:.2f}  "
            f"answer {metrics['answer_correctness']:.2f}  "
            f"false-answer {metrics['false_answer_rate']:.2f}  "
            f"false-refusal {metrics['false_refusal_rate']:.2f}  "
            f"p50 {metrics['p50_ms']:.0f}ms",
            flush=True,
        )
        # Checkpoint after every config so a long sweep is never lost.
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "ablation.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "ablation.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (REPORTS / "ablation.md").write_text(_markdown_table(table), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'ablation.md'}")
    return table


def run_calibration(settings: Settings, config: RetrievalConfig) -> Thresholds:
    """Fit tau against the *end-to-end* decision, on a dev split, in a single pass.

    Fitting on the retrieval score alone is the obvious move and it is wrong here, since
    the retrieval gate is not the only thing that can refuse: the quote check and the
    entailment verifier both refuse downstream. A tau tuned against retrieval scores in
    isolation takes credit for refusals the pipeline would have made anyway.

    So run every dev case once with the gate wide open, recording the retrieval score
    and whether the rest of the pipeline refused. Any candidate tau can then be scored
    exactly, because raising tau only ever turns an answer into a refusal, never the
    reverse. One pass, exact sweep, holdout untouched until the end.
    """
    cases = load_cases()
    dev = [case for index, case in enumerate(cases) if index % 2 == 0]
    holdout = [case for index, case in enumerate(cases) if index % 2 == 1]

    def observe(subset: Sequence[Case], label: str) -> list[tuple[bool, float, bool]]:
        pipeline = Pipeline(
            _cold(settings), config=config, thresholds=Thresholds(tau=0.0, delta=0.0)
        )
        rows = []
        for index, case in enumerate(subset, start=1):
            response = pipeline.ask(case.question)
            rows.append(
                (
                    case.should_refuse,
                    float(response.trace.get("retrieval_score", 0.0)),
                    response.answer.strip() == REFUSAL,
                )
            )
            print(f"  {label} [{index:>2}/{len(subset)}] {case.id}", flush=True)
        return rows

    def score_tau(rows: Sequence[tuple[bool, float, bool]], tau: float) -> dict[str, float]:
        decisions = [(want, score < tau or downstream) for want, score, downstream in rows]
        answerable = [(w, r) for w, r in decisions if not w]
        refusable = [(w, r) for w, r in decisions if w]
        return {
            "decision_accuracy": sum(w == r for w, r in decisions) / len(decisions),
            "false_answer_rate": (
                sum(not r for _, r in refusable) / len(refusable) if refusable else 0.0
            ),
            "false_refusal_rate": (
                sum(r for _, r in answerable) / len(answerable) if answerable else 0.0
            ),
        }

    print(f"calibrating on {len(dev)} dev cases")
    dev_rows = observe(dev, "dev")
    grid = sorted({round(score, 5) for _, score, _ in dev_rows} | {0.0, 1.0})

    # Tie-break toward fewer false answers: stating a policy rule that does not exist is
    # worse than declining to answer.
    best_tau = 0.0
    best = {"decision_accuracy": -1.0, "false_answer_rate": 1.0, "false_refusal_rate": 1.0}
    for candidate in grid:
        metrics = score_tau(dev_rows, candidate)
        if (metrics["decision_accuracy"], -metrics["false_answer_rate"]) > (
            best["decision_accuracy"],
            -best["false_answer_rate"],
        ):
            best_tau, best = candidate, metrics

    print(f"\nverifying tau={best_tau} on {len(holdout)} held-out cases")
    holdout_metrics = score_tau(observe(holdout, "hold"), best_tau)

    thresholds = Thresholds(tau=best_tau, delta=0.0, entailment=0.5)
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(
        json.dumps(
            {
                "tau": thresholds.tau,
                "delta": thresholds.delta,
                "entailment": thresholds.entailment,
                "fitted_on": {
                    "dev_cases": len(dev),
                    "holdout_cases": len(holdout),
                    "dev": {k: round(v, 4) for k, v in best.items()},
                    "holdout": {k: round(v, 4) for k, v in holdout_metrics.items()},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ntau={best_tau}")
    print(f"  dev     { {k: round(v, 3) for k, v in best.items()} }")
    print(f"  holdout { {k: round(v, 3) for k, v in holdout_metrics.items()} }")
    return thresholds


def _load_thresholds() -> Thresholds:
    if not THRESHOLDS_PATH.exists():
        return Thresholds()
    data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return Thresholds(
        tau=float(data["tau"]),
        delta=float(data.get("delta", 0.0)),
        entailment=float(data.get("entailment", 0.5)),
    )


_COLUMNS = [
    ("recall_at_3", "Recall@3"),
    ("recall_at_1", "Recall@1"),
    ("mrr", "MRR"),
    ("citation_accuracy", "Citation"),
    ("answer_correctness", "Answer"),
    ("false_answer_rate", "False answer"),
    ("false_refusal_rate", "False refusal"),
    ("decision_accuracy", "Decision"),
    ("p50_ms", "p50 ms"),
    ("p95_ms", "p95 ms"),
]


def _markdown_table(table: dict[str, dict[str, Any]]) -> str:
    header = "| config | " + " | ".join(label for _, label in _COLUMNS) + " |"
    divider = "|---" * (len(_COLUMNS) + 1) + "|"
    rows = [
        "| `" + name + "` | " + " | ".join(_format(metrics[key]) for key, _ in _COLUMNS) + " |"
        for name, metrics in table.items()
    ]
    return "\n".join(
        [
            "# Ablation",
            "",
            "Lower is better for false answer, false refusal, and both latencies.",
            "",
            header,
            divider,
            *rows,
            "",
        ]
    )


def _format(value: Any) -> str:
    return f"{value:.0f}" if isinstance(value, float) and value > 10 else f"{value}"


#: The six required questions from the Mini RAG Lab brief.
LAB_SIX: tuple[dict[str, Any], ...] = (
    {
        "q": "How much can I spend on food each day?",
        "expected": "1",
        "must_include": ("65",),
    },
    {
        "q": "Can I book first-class airfare?",
        "expected": "3",
        "must_include": ("economy",),
        "must_exclude": ("vice president",),
    },
    {
        "q": "My hotel costs $250. What do I need?",
        "expected": "2",
        "must_include": ("manager",),
    },
    {
        "q": "Do I need a receipt for a $20 taxi?",
        "expected": "5",
        "must_include": ("receipt",),
    },
    {
        "q": "Can I claim a limousine upgrade?",
        "expected": "4",
        "must_include": ("luxury",),
    },
    {
        "q": "Does the company reimburse gym memberships?",
        "expected": None,
        "must_include": (),
    },
)


def run_lab_six(settings: Settings, out: Path | None = None) -> list[dict[str, Any]]:
    """Run the six lab questions under ``--config lab`` and write the report JSON."""
    pipeline = Pipeline(_cold(settings), config=ABLATIONS["lab"])
    rows: list[dict[str, Any]] = []

    for item in LAB_SIX:
        response = pipeline.ask(item["q"])
        payload = response.to_dict()
        citation = payload.get("citation")
        cited = None if citation is None else str(citation.get("section", "")).split(".", 1)[0]
        retrieved = payload.get("retrieved_chunks") or []
        distances = [float(c["distance"]) for c in retrieved]
        expected = item["expected"]
        refused = payload["answer"].strip() == REFUSAL

        if expected is None:
            ok = refused and citation is None
            detail = f"refuse={refused} citation_null={citation is None}"
        else:
            top_ok = bool(retrieved) and str(retrieved[0]["section"]).startswith(f"{expected}.")
            cite_ok = cited == expected
            includes = all(
                needle.lower() in payload["answer"].lower() for needle in item["must_include"]
            )
            excludes = not any(
                bad.lower() in payload["answer"].lower() for bad in item.get("must_exclude", ())
            )
            ok = cite_ok and includes and excludes and not refused
            detail = f"cite_ok={cite_ok} top_ok={top_ok} includes={includes} excludes={excludes}"

        rows.append(
            {
                "q": item["q"],
                "expected": expected,
                "pass": ok,
                "detail": detail,
                "answer": payload["answer"],
                "citation": citation,
                "retrieved_chunks": retrieved,
                "sorted_asc": distances == sorted(distances),
                "n_chunks": len(retrieved),
            }
        )
        print(f"{'PASS' if ok else 'FAIL'}  {item['q']}")

    destination = out or (REPORTS / "lab-six-questions.json")
    REPORTS.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {destination}")
    return rows
