"""Golden-set helpers, metrics, ablation, calibration — no live models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragpolicy.answer import REFUSAL, Response, Thresholds
from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.evaluate import (
    GOLDEN_PATH,
    LAB_SIX,
    Case,
    CaseResult,
    _cold,
    _format,
    _load_thresholds,
    _markdown_table,
    evaluate_case,
    load_cases,
    run_ablation,
    run_calibration,
    run_lab_six,
    run_suite,
    summarise,
)
from ragpolicy.pipeline import ABLATIONS, RetrievalConfig


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        ollama_base_url="http://localhost:0",
        embed_model="stub",
        rerank_model="stub",
        gen_model="stub",
        vector_store="numpy",
        postgres_dsn="postgresql://unused",
        policy_path=REPO_ROOT / "policy.md",
        embed_cache_path=tmp_path / "emb.sqlite",
        answer_cache_path=tmp_path / "ans.sqlite",
        cassette_dir=tmp_path / "cassettes",
        cassette_mode="off",
        answer_cache=True,
    )


def _result(
    *,
    bucket: str = "answerable",
    section: str | None = "1",
    answer: str = "Employees may claim up to $65 per day.",
    cited: str | None = "1. Meals",
    retrieved: list[str] | None = None,
    must_include: tuple[str, ...] = ("$65",),
    must_exclude: tuple[str, ...] = (),
    score: float = 0.9,
) -> CaseResult:
    case = Case(
        id="x",
        bucket=bucket,
        question="q",
        section=section,
        must_include=must_include,
        must_exclude=must_exclude,
    )
    return CaseResult(
        case=case,
        answer=answer,
        refused=answer.strip() == REFUSAL,
        cited_section=None if cited is None else cited.split(".")[0],
        retrieved_sections=retrieved if retrieved is not None else ["1", "2"],
        retrieval_score=score,
        latency_ms=100.0,
        stages=[{"name": "dense", "ms": 5.0}],
    )


def test_load_cases_reads_the_golden_yaml() -> None:
    cases = load_cases(GOLDEN_PATH)
    assert len(cases) == 48
    assert cases[0].id == "meals-cap"
    assert cases[0].should_refuse is False
    refuse = next(c for c in cases if c.bucket == "unanswerable")
    assert refuse.should_refuse is True


def test_case_result_metrics_for_answerable_and_refusal() -> None:
    hit = _result()
    assert hit.correct_decision is True
    assert hit.retrieval_hit is True
    assert hit.rank_of_gold == 1
    assert hit.citation_correct is True
    assert hit.content_correct is True

    miss = _result(retrieved=["2", "3"], cited="2. Hotels", answer="No idea")
    assert miss.retrieval_hit is False
    assert miss.rank_of_gold is None
    assert miss.citation_correct is False
    assert miss.content_correct is False

    refused = _result(bucket="unanswerable", section=None, answer=REFUSAL, cited=None)
    assert refused.correct_decision is True
    assert refused.retrieval_hit is None
    assert refused.citation_correct is None
    assert refused.content_correct is True


def test_content_correct_rejects_forbidden_strings_on_refusals() -> None:
    bad = _result(
        bucket="unanswerable",
        section=None,
        answer=REFUSAL + " but also $65",
        cited=None,
        must_exclude=("$65",),
    )
    # refused is True because answer equals REFUSAL only when strip-equal; here it does not.
    assert bad.refused is False
    assert bad.content_correct is False


def test_content_correct_for_answerable_refusals_and_exclusions() -> None:
    refused = _result(answer=REFUSAL, cited=None)
    assert refused.refused is True
    assert refused.content_correct is False

    polluted = _result(answer="Up to $65 but also a limousine", must_exclude=("limousine",))
    assert polluted.content_correct is False


def test_summarise_handles_empty_and_populated_results() -> None:
    empty = summarise([])
    assert empty["n"] == 0
    assert empty["recall_at_3"] == 0.0
    assert empty["p50_ms"] == 0.0

    metrics = summarise([_result(), _result(bucket="unanswerable", section=None, answer=REFUSAL)])
    assert metrics["n"] == 2
    assert metrics["recall_at_3"] == 1.0
    assert "dense" in metrics["stage_p50_ms"]


def test_evaluate_case_records_pipeline_response(monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(id="c", bucket="answerable", question="q", section="1", must_include=("$65",))

    class FakePipeline:
        def ask(self, question: str) -> Response:
            return Response(
                answer="Employees may claim up to $65.",
                citation={"section": "1. Meals"},
                retrieved_chunks=[{"section": "1. Meals", "distance": 0.1}],
                trace={"retrieval_score": 0.8, "stages": [{"name": "dense", "ms": 1.0}]},
            )

    result = evaluate_case(FakePipeline(), case)  # type: ignore[arg-type]
    assert result.cited_section == "1"
    assert result.retrieval_score == 0.8
    assert result.retrieved_sections == ["1"]


def test_cold_disables_the_answer_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.answer_cache is True
    assert _cold(settings).answer_cache is False


def test_load_thresholds_defaults_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr("ragpolicy.evaluate.THRESHOLDS_PATH", missing)
    assert _load_thresholds() == Thresholds()

    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"tau": 0.01, "delta": 0.0, "entailment": 0.4}), encoding="utf-8")
    monkeypatch.setattr("ragpolicy.evaluate.THRESHOLDS_PATH", path)
    loaded = _load_thresholds()
    assert loaded.tau == 0.01
    assert loaded.entailment == 0.4


def test_format_and_markdown_table() -> None:
    assert _format(12.3) == "12"
    assert _format(0.95) == "0.95"
    table = {
        "full": {
            "recall_at_3": 1.0,
            "recall_at_1": 0.9,
            "mrr": 0.95,
            "citation_accuracy": 0.9,
            "answer_correctness": 0.9,
            "false_answer_rate": 0.1,
            "false_refusal_rate": 0.0,
            "decision_accuracy": 0.95,
            "p50_ms": 4000.0,
            "p95_ms": 8000.0,
        }
    }
    md = _markdown_table(table)
    assert md.startswith("# Ablation")
    assert "`full`" in md
    assert "4000" in md


def test_run_suite_prints_progress_and_returns_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = [
        Case(id="a", bucket="answerable", question="q1", section="1", must_include=("$65",)),
        Case(id="b", bucket="unanswerable", question="q2"),
    ]
    monkeypatch.setattr("ragpolicy.evaluate.load_cases", lambda: cases)
    monkeypatch.setattr(
        "ragpolicy.evaluate._load_thresholds", lambda: Thresholds(tau=0.0, delta=0.0)
    )

    class StubPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def ask(self, question: str) -> Response:
            if question == "q2":
                return Response(
                    answer=REFUSAL,
                    citation=None,
                    retrieved_chunks=[],
                    trace={"retrieval_score": 0.0, "stages": []},
                )
            return Response(
                answer="Employees may claim up to $65.",
                citation={"section": "1"},
                retrieved_chunks=[{"section": "1", "distance": 0.1}],
                trace={"retrieval_score": 0.9, "stages": [{"name": "dense", "ms": 2.0}]},
            )

    monkeypatch.setattr("ragpolicy.evaluate.Pipeline", StubPipeline)
    metrics, results = run_suite(_settings(tmp_path), RetrievalConfig(), "full", limit=2)
    assert metrics["n"] == 2
    assert len(results) == 2
    out = capsys.readouterr().out
    assert "ok" in out
    assert "full" in out


def test_run_ablation_skips_lab_and_resumes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    checkpoint = reports / "ablation.json"
    cached_metrics = {
        "recall_at_3": 1.0,
        "recall_at_1": 0.9,
        "mrr": 0.9,
        "citation_accuracy": 0.9,
        "answer_correctness": 0.9,
        "false_answer_rate": 0.1,
        "false_refusal_rate": 0.0,
        "decision_accuracy": 0.9,
        "p50_ms": 100.0,
        "p95_ms": 200.0,
        "stage_p50_ms": {},
    }
    checkpoint.write_text(json.dumps({"dense": cached_metrics}), encoding="utf-8")
    monkeypatch.setattr("ragpolicy.evaluate.REPORTS", reports)
    monkeypatch.setattr(
        "ragpolicy.evaluate.ABLATIONS",
        {
            "dense": ABLATIONS["dense"],
            "lab": ABLATIONS["lab"],
            "full": ABLATIONS["full"],
        },
    )
    calls: list[str] = []

    def fake_suite(
        settings: Settings,
        config: RetrievalConfig,
        name: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[CaseResult]]:
        calls.append(name)
        return (
            {
                "recall_at_3": 1.0,
                "recall_at_1": 0.9,
                "mrr": 0.9,
                "citation_accuracy": 0.9,
                "answer_correctness": 0.9,
                "false_answer_rate": 0.1,
                "false_refusal_rate": 0.0,
                "decision_accuracy": 0.9,
                "p50_ms": 100.0,
                "p95_ms": 200.0,
                "stage_p50_ms": {},
            },
            [],
        )

    monkeypatch.setattr("ragpolicy.evaluate.run_suite", fake_suite)
    table = run_ablation(_settings(tmp_path))
    assert "dense" in table
    assert "full" in table
    assert "lab" not in table
    assert calls == ["full"]
    assert (reports / "ablation.md").exists()


def test_run_calibration_fits_tau_and_writes_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        Case(id=f"c{i}", bucket="answerable" if i % 2 == 0 else "unanswerable", question=f"q{i}")
        for i in range(4)
    ]
    monkeypatch.setattr("ragpolicy.evaluate.load_cases", lambda: cases)
    path = tmp_path / "thresholds.json"
    monkeypatch.setattr("ragpolicy.evaluate.THRESHOLDS_PATH", path)

    scores = {"q0": 0.8, "q1": 0.1, "q2": 0.7, "q3": 0.05}

    class StubPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def ask(self, question: str) -> Response:
            refuse = scores[question] < 0.3
            return Response(
                answer=REFUSAL if refuse else "ok $65",
                citation=None if refuse else {"section": "1"},
                retrieved_chunks=[],
                trace={"retrieval_score": scores[question]},
            )

    monkeypatch.setattr("ragpolicy.evaluate.Pipeline", StubPipeline)
    fitted = run_calibration(_settings(tmp_path), RetrievalConfig())
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tau"] == fitted.tau
    assert "fitted_on" in data


def test_run_lab_six_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "lab-six.json"
    monkeypatch.setattr("ragpolicy.evaluate.REPORTS", tmp_path)

    class StubPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._i = 0

        def ask(self, question: str) -> Response:
            item = LAB_SIX[self._i]
            self._i += 1
            if item["expected"] is None:
                return Response(
                    answer=REFUSAL,
                    citation=None,
                    retrieved_chunks=[],
                    trace={},
                )
            section = f"{item['expected']}. Title"
            answer = " ".join(item["must_include"]) or "ok"
            return Response(
                answer=answer,
                citation={"section": section},
                retrieved_chunks=[{"section": section, "distance": 0.1}],
                trace={},
            )

    monkeypatch.setattr("ragpolicy.evaluate.Pipeline", StubPipeline)
    rows = run_lab_six(_settings(tmp_path), out=out)
    assert len(rows) == 6
    assert all(row["pass"] for row in rows)
    assert out.exists()
