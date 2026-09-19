"""Command line entry point: ``rag index``, ``rag ask``, ``rag eval``, ``rag calibrate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragpolicy.config import Settings
from ragpolicy.pipeline import ABLATIONS, Pipeline, RetrievalConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="embed the policy and write it to the vector store")
    index.add_argument(
        "--lab",
        action="store_true",
        help="lab contract: index exactly six bare section chunks (does not replace full)",
    )

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--config", choices=sorted(ABLATIONS), default="full")
    ask.add_argument("--json", action="store_true", help="print the full response with trace")

    evaluate = sub.add_parser("eval", help="run the golden set")
    evaluate.add_argument("--ablate", action="store_true", help="sweep every configuration")
    evaluate.add_argument("--config", choices=sorted(ABLATIONS), default="full")
    evaluate.add_argument("--limit", type=int, default=0, help="only the first N questions")

    calibrate = sub.add_parser("calibrate", help="fit abstention thresholds on the dev split")
    calibrate.add_argument("--config", choices=sorted(ABLATIONS), default="full")

    lab_six = sub.add_parser(
        "lab-six",
        help="run the six Mini RAG Lab questions under --config lab and write the report",
    )
    lab_six.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON path (default: reports/lab-six-questions.json)",
    )

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "index":
        config = ABLATIONS["lab"] if args.lab else RetrievalConfig()
        count = Pipeline(settings, config=config).index()
        mode = "lab (6 bare sections)" if args.lab else "full"
        print(f"indexed {count} chunks ({mode}) from {settings.policy_path.name}")
        return 0

    if args.command == "ask":
        pipeline = Pipeline(settings, config=_config_for(args.config))
        response = pipeline.ask(args.question)
        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
        else:
            _print_human(response)
        return 0

    from ragpolicy.evaluate import run_ablation, run_calibration, run_lab_six, run_suite

    if args.command == "eval":
        if args.ablate:
            run_ablation(settings, limit=args.limit)
        else:
            run_suite(settings, _config_for(args.config), args.config, limit=args.limit)
        return 0

    if args.command == "calibrate":
        run_calibration(settings, _config_for(args.config))
        return 0

    if args.command == "lab-six":
        run_lab_six(settings, out=args.out)
        return 0

    return 1


def _config_for(name: str) -> RetrievalConfig:
    return ABLATIONS[name]


def _print_human(response: object) -> None:
    payload = response.to_dict()  # type: ignore[attr-defined]
    trace = payload["trace"]

    print(f"\n{payload['answer']}\n")
    if payload["citation"]:
        citation = payload["citation"]
        print(f"  cited: {citation['document']} v{citation['version']}, §{citation['section']}")
    else:
        print(f"  abstained at: {trace.get('abstained_at')}")

    print(f"  confidence: {trace.get('confidence', 0):.3f}")
    print("\n  retrieved:")
    for chunk in payload["retrieved_chunks"]:
        print(f"    {chunk['distance']:.4f}  {chunk['section']}")

    stages = trace.get("stages", [])
    if stages:
        timings = "  ".join(f"{s['name']} {s['ms']:.0f}ms" for s in stages)
        print(f"\n  {timings}  |  total {trace.get('total_ms', 0):.0f}ms")


if __name__ == "__main__":
    sys.exit(main())
