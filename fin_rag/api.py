"""FastAPI service layer.

Two endpoints matter in production: a JSON one for server-to-server callers and
an SSE one, because a RAG answer takes seconds and users abandon a spinner.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .compliance import REFUSAL_MESSAGE, Verdict, check_query
from .config import load_settings
from .graph import ANSWER_PROMPT, RagPipeline, build_context
from .ingest import build_from_corpus
from .llm import build_llm


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    index: int
    doc_id: str
    chunk_id: str
    source: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    verdict: str
    reasons: list[str]
    disclaimer: str
    trace: list[str]


class AppState:
    """Holds the indexed retriever so it is built once at startup, not per request."""

    pipeline: RagPipeline | None = None


state = AppState()


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(
        title="Financial Knowledge Base (RAG)",
        version="0.1.0",
        description=(
            "LangGraph-orchestrated hybrid-retrieval RAG service with compliance guardrails."
        ),
    )
    if pipeline is not None:
        state.pipeline = pipeline

    @app.on_event("startup")
    def _startup() -> None:
        if state.pipeline is not None:
            return
        settings = load_settings()
        retriever, _ = build_from_corpus(settings=settings)
        llm = build_llm(
            settings.llm_provider, model=settings.llm_model, base_url=settings.llm_base_url
        )
        state.pipeline = RagPipeline(retriever=retriever, llm=llm, settings=settings)

    def _require_pipeline() -> RagPipeline:
        if state.pipeline is None:
            raise HTTPException(status_code=503, detail="index not ready")
        return state.pipeline

    @app.get("/health")
    def health() -> dict[str, Any]:
        pipeline = state.pipeline
        return {
            "status": "ok" if pipeline else "starting",
            "indexed_chunks": len(pipeline.retriever) if pipeline else 0,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        pipeline = _require_pipeline()
        result = pipeline.answer(request.question)
        return AskResponse(**result.to_dict())

    @app.post("/ask/stream")
    def ask_stream(request: AskRequest) -> StreamingResponse:
        pipeline = _require_pipeline()
        return StreamingResponse(
            _sse(pipeline, request.question),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _sse(pipeline: RagPipeline, question: str) -> Iterator[str]:
    """Emit citations *before* the tokens.

    Deliberate ordering: the UI can render sources while the answer is still
    being written, and the client never has to wait for the stream to close to
    know what the answer was grounded in.
    """

    def event(kind: str, payload: dict[str, Any]) -> str:
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    guard = check_query(question)
    if guard.verdict is Verdict.REFUSE:
        yield event("refusal", {"reason": guard.reasons, "answer": REFUSAL_MESSAGE})
        yield event("done", {})
        return

    hits = pipeline.retriever.search(
        question,
        top_k=pipeline.settings.retrieval.top_k,
        candidate_k=pipeline.settings.retrieval.candidate_k,
    )
    if pipeline.settings.retrieval.use_rerank and hits:
        hits = pipeline.reranker.rerank(question, hits, top_k=pipeline.settings.retrieval.top_k)
    hits = [h for h in hits if h.score >= pipeline.settings.retrieval.min_score]
    context, citations = build_context(hits)
    yield event("citations", {"citations": citations})

    if not hits:
        yield event("token", {"text": "根据现有资料无法回答该问题。"})
        yield event("done", {})
        return

    prompt = ANSWER_PROMPT.format(context=context, question=question)
    for piece in pipeline.llm.stream(prompt):
        yield event("token", {"text": piece})
    if guard.disclaimer:
        yield event("disclaimer", {"text": guard.disclaimer})
    yield event("done", {})


app = create_app()
