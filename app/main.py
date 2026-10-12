"""FastAPI service for the vehicle safety agent (Swagger UI at /docs).

    uvicorn app.main:app --port 8000

Needs the artifacts written by ``scripts/run_pipeline.py`` (DuckDB database, recall embeddings) and a chat model:
local Ollama by default, or OpenAI / Azure OpenAI through ``SAFETYAGENT_LLM`` (see safetyagent/llm.py).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from safetyagent.agent import AgentResult, SafetyAgent
from safetyagent.tools import RecallSearch, SafetyDB, StatsQuery, validation_messages

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "How many complaints about the brakes of the 2024 Toyota Tacoma has NHTSA received, and is there a recall that covers this problem?"


class Engine(Protocol):
    db: SafetyDB

    def ask(self, question: str) -> AgentResult: ...

    def search(self, q: RecallSearch, rid: str) -> Any: ...


class AskRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500, examples=[EXAMPLE])


class ToolCallOut(BaseModel):
    tool: str
    arguments: dict[str, Any]


class EvidenceOut(BaseModel):
    id: str
    kind: str
    text: str


class AskResponse(BaseModel):
    answer: str
    route: str
    citations: list[str]
    grounded: bool = Field(description="the final answer passed the citation and number checks")
    problems: list[str]
    tool_calls: list[ToolCallOut]
    sql: list[str]
    evidence: list[EvidenceOut]
    plan_attempts: int
    answer_attempts: int
    latency_ms: float


class StatsOut(BaseModel):
    sql: str
    params: list[Any]
    columns: list[str]
    rows: list[list[Any]]
    rendered: str


class SearchOut(BaseModel):
    campaign: str
    score: float
    report_date: str
    components: str
    defect: str
    consequence: str
    remedy: str


def build_default_engine() -> SafetyAgent:
    """Agent over the pipeline's artifacts; the query encoder runs on CPU unless SAFETYAGENT_DEVICE says otherwise."""
    import numpy as np

    from safetyagent.data import coverage
    from safetyagent.llm import make_llm_from_env
    from safetyagent.search import Encoder, RecallIndex

    cfg = json.loads((ROOT / "data" / "cache" / "api" / "config.json").read_text())
    db = SafetyDB(ROOT / cfg["db"])
    ids = [r[0] for r in db.con.execute("SELECT campaign FROM recalls ORDER BY campaign").fetchall()]
    enc = Encoder(cfg["embed_model"], device=os.getenv("SAFETYAGENT_DEVICE", "cpu"))
    index = RecallIndex(ids, np.load(ROOT / cfg["embeddings"]), enc.encode_query)
    llm = make_llm_from_env(ROOT / cfg["llm_cache"])
    return SafetyAgent(llm, db, index, coverage(db.con))


def create_app(engine_factory: Callable[[], Engine]) -> FastAPI:
    """Build the app; the engine is created once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine_factory()
        yield

    app = FastAPI(title="Vehicle Safety Intelligence Agent", version="1.0.0", lifespan=lifespan,
                  description="LangGraph agent over NHTSA complaints and recalls: SQL tools, recall search, cited and "
                              "grounding-checked answers.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        t0 = time.perf_counter()
        res = request.app.state.engine.ask(req.question)
        calls = [ToolCallOut(tool="stats_query", arguments={k: v for k, v in c.items() if v is not None}) for c in res.stats_calls]
        calls += [ToolCallOut(tool="recall_search", arguments={k: v for k, v in c.items() if v is not None}) for c in res.search_calls]
        return AskResponse(answer=res.answer, route=res.route, citations=res.citations, grounded=res.grounded, problems=res.problems,
                           tool_calls=calls, sql=res.sql, evidence=[EvidenceOut(id=e["id"], kind=e["kind"], text=e["text"]) for e in res.evidence],
                           plan_attempts=res.plan_attempts, answer_attempts=res.answer_attempts,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    @app.post("/tools/stats", response_model=StatsOut)
    def stats(body: dict[str, Any], request: Request) -> StatsOut:
        """Run the SQL tool directly (no LLM); arguments are validated like the agent's tool calls."""
        db = request.app.state.engine.db
        try:
            q = StatsQuery.model_validate(body, context={"vocab": db.vocab})
        except ValidationError as err:
            raise HTTPException(status_code=422, detail=validation_messages(err, "stats_query"))
        res = db.run_stats(q)
        return StatsOut(sql=res.sql, params=res.params, columns=res.columns, rows=[list(r) for r in res.rows], rendered=res.render())

    @app.post("/tools/search", response_model=list[SearchOut])
    def search(body: dict[str, Any], request: Request) -> list[SearchOut]:
        """Run the recall search directly (no LLM)."""
        engine = request.app.state.engine
        try:
            q = RecallSearch.model_validate(body, context={"vocab": engine.db.vocab})
        except ValidationError as err:
            raise HTTPException(status_code=422, detail=validation_messages(err, "recall_search"))
        res = engine.search(q, "X1")
        return [SearchOut(campaign=h.campaign, score=round(h.score, 4), report_date=h.report_date, components=h.components,
                          defect=h.defect, consequence=h.consequence, remedy=h.remedy) for h in res.hits]

    return app


app = create_app(build_default_engine)
