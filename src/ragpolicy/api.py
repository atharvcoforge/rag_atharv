"""HTTP layer over :class:`~ragpolicy.pipeline.Pipeline`.

``POST /api/ask`` returns the lab's response contract byte for byte; everything this
system knows beyond that contract stays under ``trace``. The streaming endpoint replays
the same payload as Server-Sent Events so the UI can render stages and tokens as they
arrive.

The pipeline is built on first use, never at import time: constructing one opens a
Postgres connection and loads the corpus, which neither the tests nor CI have.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ragpolicy.config import REPO_ROOT, Settings
from ragpolicy.ingest import parse_document
from ragpolicy.pipeline import ABLATIONS, Pipeline

ABLATION_REPORT = REPO_ROOT / "reports" / "ablation.json"

_PIPELINES: dict[str, Pipeline] = {}


def _cached_pipeline(config: str) -> Pipeline:
    """One pipeline per retrieval config, built on first request and kept."""
    if config not in _PIPELINES:
        from ragpolicy.evaluate import _load_thresholds

        _PIPELINES[config] = Pipeline(
            Settings.from_env(), config=ABLATIONS[config], thresholds=_load_thresholds()
        )
    return _PIPELINES[config]


def pipeline_factory() -> Callable[[str], Pipeline]:
    """Injection seam: tests override this so no Ollama or Postgres call happens."""
    return _cached_pipeline


PipelineFor = Annotated[Callable[[str], Pipeline], Depends(pipeline_factory)]

app = FastAPI(title="ragpolicy", description=__doc__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    config: str = "full"


def _checked(config: str) -> str:
    if config not in ABLATIONS:
        raise HTTPException(status_code=422, detail=f"unknown config: {config}")
    return config


@app.post("/api/ask")
def ask(body: AskRequest, build: PipelineFor) -> dict[str, Any]:
    return build(_checked(body.config)).ask(body.question).to_dict()


@app.get("/api/ask/stream")
def ask_stream(question: str, build: PipelineFor, config: str = "full") -> StreamingResponse:
    payload = build(_checked(config)).ask(question).to_dict()
    return StreamingResponse(_events(payload), media_type="text/event-stream")


@app.get("/api/document")
def document(build: PipelineFor) -> dict[str, Any]:
    pipeline = build("full")
    parsed = parse_document(pipeline.raw)
    return {
        "raw": pipeline.raw,
        "document": parsed.title,
        "version": parsed.version,
        "sections": [
            {
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
            }
            for chunk in pipeline.corpus
            if chunk.kind == "section"
        ],
    }


@app.post("/api/index")
def index(build: PipelineFor) -> dict[str, int]:
    return {"indexed": build("full").index()}


@app.get("/api/health")
def health(build: PipelineFor) -> dict[str, Any]:
    pipeline = build("full")
    settings = pipeline.settings
    return {
        "ok": True,
        "store": type(pipeline.store).__name__,
        "models": {
            "embed": settings.embed_model,
            "rerank": settings.rerank_model,
            "gen": settings.gen_model,
        },
        "chunks": len(pipeline.corpus),
    }


@app.get("/api/eval/latest")
def eval_latest() -> dict[str, Any]:
    """The last ablation sweep, or an empty table when nobody has run one yet."""
    if not ABLATION_REPORT.exists():
        return {}
    loaded: dict[str, Any] = json.loads(ABLATION_REPORT.read_text(encoding="utf-8"))
    return loaded


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _events(payload: dict[str, Any]) -> Iterator[str]:
    """Stages first, then the answer a token at a time, then the whole response."""
    for stage in payload["trace"].get("stages", []):
        yield _sse("stage", {"name": stage["name"], "ms": stage["ms"]})
    for token in re.findall(r"\S+\s*", payload["answer"]):
        yield _sse("token", token)
    yield _sse("done", payload)
