"""CLI entry points, with Pipeline and eval harness stubbed out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragpolicy import cli
from ragpolicy.answer import REFUSAL, Response
from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.pipeline import ABLATIONS, RetrievalConfig
from ragpolicy.rerank_probe import CaseScore, ProbeResult


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
        answer_cache=False,
    )


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "Settings",
        type("S", (), {"from_env": staticmethod(lambda: _settings(tmp_path))}),
    )


def test_index_full_and_lab(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[RetrievalConfig] = []

    class StubPipeline:
        def __init__(self, settings: Settings, config: RetrievalConfig | None = None) -> None:
            calls.append(config or RetrievalConfig())

        def index(self) -> int:
            return 6 if calls[-1].lab else 16

    monkeypatch.setattr(cli, "Pipeline", StubPipeline)
    assert cli.main(["index"]) == 0
    assert "full" in capsys.readouterr().out
    assert cli.main(["index", "--lab"]) == 0
    assert "lab" in capsys.readouterr().out
    assert calls[0].lab is False
    assert calls[1].lab is True


def test_ask_human_and_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = Response(
        answer="Employees may claim up to $65 per day for meals.",
        citation={"document": "Employee Expense Policy", "version": "2.0", "section": "1. Meals"},
        retrieved_chunks=[{"section": "1. Meals", "distance": 0.12}],
        trace={
            "confidence": 0.9,
            "stages": [{"name": "dense", "ms": 8.0}, {"name": "rerank", "ms": 900.0}],
            "total_ms": 1200.0,
        },
    )

    class StubPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def ask(self, question: str) -> Response:
            return response

    monkeypatch.setattr(cli, "Pipeline", StubPipeline)
    assert cli.main(["ask", "How much for meals?"]) == 0
    human = capsys.readouterr().out
    assert "$65" in human
    assert "cited:" in human
    assert "dense" in human

    assert cli.main(["ask", "--json", "How much for meals?"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"].startswith("Employees")


def test_ask_prints_abstention(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = Response(
        answer=REFUSAL,
        citation=None,
        retrieved_chunks=[{"section": "4. Ground", "distance": 0.4}],
        trace={"abstained_at": "retrieval", "confidence": 0.1, "stages": [], "total_ms": 50.0},
    )

    class StubPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def ask(self, question: str) -> Response:
            return response

    monkeypatch.setattr(cli, "Pipeline", StubPipeline)
    assert cli.main(["ask", "rental car?"]) == 0
    out = capsys.readouterr().out
    assert "abstained at: retrieval" in out


def test_eval_ablate_and_single(monkeypatch: pytest.MonkeyPatch) -> None:
    import ragpolicy.evaluate as evaluate

    called: dict[str, Any] = {}

    def ablate(settings: Settings, limit: int = 0) -> dict[str, Any]:
        called["ablate"] = limit
        return {}

    def suite(
        settings: Settings, config: RetrievalConfig, name: str, limit: int = 0
    ) -> tuple[dict[str, Any], list[Any]]:
        called["suite"] = (name, limit, config)
        return {}, []

    monkeypatch.setattr(evaluate, "run_ablation", ablate)
    monkeypatch.setattr(evaluate, "run_suite", suite)

    assert cli.main(["eval", "--ablate", "--limit", "3"]) == 0
    assert called["ablate"] == 3
    assert cli.main(["eval", "--config", "dense", "--limit", "1"]) == 0
    assert called["suite"][0] == "dense"
    assert called["suite"][1] == 1


def test_calibrate_and_lab_six(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ragpolicy.evaluate as evaluate

    called: dict[str, Any] = {}

    def calibrate(settings: Settings, config: RetrievalConfig) -> Any:
        called["calibrate"] = config
        return None

    def lab_six(settings: Settings, out: Path | None = None) -> list[Any]:
        called["lab"] = out
        return []

    monkeypatch.setattr(evaluate, "run_calibration", calibrate)
    monkeypatch.setattr(evaluate, "run_lab_six", lab_six)

    assert cli.main(["calibrate", "--config", "full"]) == 0
    assert called["calibrate"] == ABLATIONS["full"]

    out = tmp_path / "six.json"
    assert cli.main(["lab-six", "--out", str(out)]) == 0
    assert called["lab"] == out


def test_rerank_probe_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class StubClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            return None

    def pass_result(*args: Any, **kwargs: Any) -> ProbeResult:
        return ProbeResult(
            model="m",
            scores=(CaseScore(question="q", governing=0.9, near_miss=0.1),),
            ms_per_call=40.0,
            baseline_ms=150.0,
        )

    def fail_result(*args: Any, **kwargs: Any) -> ProbeResult:
        return ProbeResult(
            model="m",
            scores=(CaseScore(question="q", governing=0.1, near_miss=0.9),),
            ms_per_call=40.0,
            baseline_ms=150.0,
        )

    monkeypatch.setattr(cli, "OllamaClient", StubClient)
    monkeypatch.setattr(cli, "run_probe", pass_result)
    assert cli.main(["rerank-probe", "--model", "qwen3:1.7b"]) == 0
    assert "PASS" in capsys.readouterr().out

    monkeypatch.setattr(cli, "run_probe", fail_result)
    assert cli.main(["rerank-probe"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_config_for_looks_up_ablations() -> None:
    assert cli._config_for("lab") is ABLATIONS["lab"]
