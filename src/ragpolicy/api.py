"""HTTP layer over :class:`~ragpolicy.pipeline.Pipeline`.

``POST /api/ask`` returns the lab's response contract byte for byte; everything this
system knows beyond that contract stays under ``trace``. The streaming endpoint reports
the same run live over Server-Sent Events: retrieval stages land in milliseconds, the
generation and verification phases are announced as they start, and the answer itself
arrives once the abstention gates have passed on it.

The pipeline is built on first use, never at import time: constructing one opens a
Postgres connection and loads the corpus, which neither the tests nor CI have.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from queue import Queue
from threading import Thread
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
    pipeline = build(_checked(config))
    return StreamingResponse(_events(pipeline, question), media_type="text/event-stream")


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


@app.post("/api/warm")
def warm(build: PipelineFor) -> dict[str, float]:
    """Load both models on demand, so a deploy pays the cold start instead of a user."""
    return build("full").client.warm()


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
        "endpoints": {
            "ollama": settings.ollama_base_url,
            "rerank": settings.rerank_ollama_url or settings.ollama_base_url,
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


def _events(pipeline: Pipeline, question: str) -> Iterator[str]:
    """Progress as it happens: each retrieval stage, then the answer phases, then the payload.

    The pipeline runs on a worker thread and reports through a queue, which is what lets
    a stage that finished in 5ms reach the browser while generation is still running.

    ``done`` stays atomic on purpose. Generated text is only an answer once the quote and
    entailment gates have accepted it, and showing a sentence that verification may still
    refuse would be a worse experience than the wait it saves.
    """
    events: Queue[tuple[str, Any] | None] = Queue()
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["done"] = pipeline.ask(
                question, on_event=lambda name, data: events.put((name, data))
            ).to_dict()
        except Exception as error:  # a dead stream tells the UI nothing; an event does
            outcome["error"] = {"message": str(error)}
        finally:
            events.put(None)

    worker = Thread(target=run, daemon=True)
    worker.start()

    while (event := events.get()) is not None:
        name, data = event
        yield _sse(name, data)

    worker.join()
    if "error" in outcome:
        yield _sse("error", outcome["error"])
    else:
        yield _sse("done", outcome["done"])
