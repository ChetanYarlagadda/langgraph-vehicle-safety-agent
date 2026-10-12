"""The LangGraph state machine.

    classify -> plan -(invalid calls, retries left)-> plan
                     -> execute -> answer -> check_grounding -(problems, retries left)-> answer
                                                             -> END

* ``classify`` asks the LLM for a route (sql / semantic / both); the keyword router is the fallback.
* ``plan`` asks for tool calls in a JSON schema that only offers the route's tools; each call is validated with
  Pydantic against the database vocabulary, and validation messages go back to the LLM on a retry.
* ``execute`` runs the SQL tool (parameterised DuckDB) and the recall search (FAISS) and builds cited evidence.
* ``answer`` drafts a cited answer from the evidence; ``check_grounding`` verifies citations and numbers and sends
  the problems back for a rewrite.
"""

from __future__ import annotations

import json
import operator
import time
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from safetyagent import prompts
from safetyagent.grounding import Evidence, check_grounding
from safetyagent.llm import ChatModel, ChatReply
from safetyagent.router import keyword_route
from safetyagent.tools import RecallHit, RecallSearch, SafetyDB, SearchResult, StatsQuery, ungrounded_arguments, validation_messages


class AgentState(TypedDict, total=False):
    question: str
    route: str
    route_source: str
    plan_attempts: int
    plan_raw: str
    plan_errors: list[str]
    stats_calls: list[dict]
    search_calls: list[dict]
    call_log: Annotated[list[dict], operator.add]
    evidence: list[dict]
    sql: list[str]
    answer: str
    citations: list[str]
    answer_attempts: int
    problems: list[str]
    answers: Annotated[list[dict], operator.add]
    declined: bool
    events: Annotated[list[dict], operator.add]


@dataclass
class AgentResult:
    question: str
    route: str
    route_source: str
    answer: str
    citations: list[str]
    grounded: bool
    problems: list[str]
    stats_calls: list[dict]
    search_calls: list[dict]
    sql: list[str]
    evidence: list[dict]
    plan_attempts: int
    answer_attempts: int
    call_log: list[dict]
    answers: list[dict]
    declined: bool = False
    events: list[dict] = field(default_factory=list)

    @property
    def llm_seconds(self) -> float:
        return sum(e.get("llm_s", 0.0) for e in self.events)

    @property
    def tool_seconds(self) -> float:
        return sum(e.get("tool_s", 0.0) for e in self.events)

    @property
    def latency_s(self) -> float:
        """Model time of each LLM call (as originally measured, also for cache hits) plus tool time."""
        return self.llm_seconds + self.tool_seconds

    @property
    def llm_calls(self) -> int:
        return sum(1 for e in self.events if "llm_s" in e)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(latency_s=self.latency_s, llm_seconds=self.llm_seconds, tool_seconds=self.tool_seconds, llm_calls=self.llm_calls)
        return d


def _event(node: str, reply: ChatReply | None = None, tool_s: float | None = None) -> dict:
    e: dict[str, Any] = {"node": node}
    if reply is not None:
        e.update(llm_s=reply.latency_s, cached=reply.cached, prompt_tokens=reply.prompt_tokens, output_tokens=reply.output_tokens)
    if tool_s is not None:
        e["tool_s"] = tool_s
    return e


def _parse_json(text: str) -> dict | None:
    try:
        out = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return out if isinstance(out, dict) else None


class SafetyAgent:
    """Question -> route -> validated tool calls -> evidence -> cited, grounding-checked answer."""

    def __init__(self, llm: ChatModel, db: SafetyDB, index, coverage: dict, max_plan_attempts: int = 3,
                 max_answer_attempts: int = 3) -> None:
        self.llm = llm
        self.db = db
        self.index = index
        self.coverage = coverage
        self.max_plan_attempts = max_plan_attempts
        self.max_answer_attempts = max_answer_attempts
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("classify", self.classify)
        g.add_node("plan", self.plan)
        g.add_node("execute", self.execute)
        g.add_node("answer", self.answer)
        g.add_node("check_grounding", self.check)
        g.add_node("decline", self.decline)
        g.add_edge(START, "classify")
        g.add_edge("classify", "plan")
        g.add_conditional_edges("plan", self.after_plan, {"plan": "plan", "execute": "execute", "decline": "decline"})
        g.add_edge("decline", END)
        g.add_edge("execute", "answer")
        g.add_edge("answer", "check_grounding")
        g.add_conditional_edges("check_grounding", self.after_check, {"answer": "answer", "end": END})
        return g.compile()

    def ask(self, question: str) -> AgentResult:
        s = self.graph.invoke({"question": question}, config={"recursion_limit": 30})
        return AgentResult(
            question=question, route=s["route"], route_source=s["route_source"], answer=s.get("answer", ""),
            citations=s.get("citations", []), grounded=not s.get("problems"), problems=s.get("problems", []),
            stats_calls=s.get("stats_calls", []), search_calls=s.get("search_calls", []), sql=s.get("sql", []),
            evidence=s.get("evidence", []), plan_attempts=s.get("plan_attempts", 0), answer_attempts=s.get("answer_attempts", 0),
            call_log=s.get("call_log", []), answers=s.get("answers", []), declined=s.get("declined", False), events=s.get("events", []),
        )

    # ------------------------------------------------------------------ nodes
    def classify(self, state: AgentState) -> dict:
        msgs = [{"role": "system", "content": prompts.classify_system()}, {"role": "user", "content": state["question"]}]
        reply = self.llm.chat(msgs, prompts.ROUTE_SCHEMA, max_tokens=20)
        out = _parse_json(reply.text)
        route = out.get("route") if out else None
        if route in prompts.ROUTES:
            return {"route": route, "route_source": "llm", "events": [_event("classify", reply)]}
        return {"route": keyword_route(state["question"]), "route_source": "keyword fallback", "events": [_event("classify", reply)]}

    def plan(self, state: AgentState) -> dict:
        route = state["route"]
        attempt = state.get("plan_attempts", 0) + 1
        user = (prompts.plan_user(state["question"], route) if attempt == 1 else
                prompts.plan_retry_user(state["question"], route, state.get("plan_raw", ""), state.get("plan_errors", [])))
        msgs = [{"role": "system", "content": prompts.plan_system(self.coverage)}, {"role": "user", "content": user}]
        reply = self.llm.chat(msgs, prompts.plan_schema(route), max_tokens=500)
        data = _parse_json(reply.text)
        errors: list[str] = []
        stats: list[dict] = []
        searches: list[dict] = []
        if data is None:
            errors.append("the output is not a JSON object")
            data = {}
        ctx = {"vocab": self.db.vocab}
        raw_stats = data.get("stats_queries") or []
        raw_search = data.get("recall_searches") or []
        for name, raws, model, out in (("stats_queries", raw_stats if route in ("sql", "both") else [], StatsQuery, stats),
                                       ("recall_searches", raw_search if route in ("semantic", "both") else [], RecallSearch, searches)):
            for i, raw in enumerate(raws):
                try:
                    call = model.model_validate(raw, context=ctx)
                except ValidationError as err:
                    errors += validation_messages(err, f"{name}[{i}]")
                    continue
                problems = ungrounded_arguments(call, state["question"], f"{name}[{i}]")
                if problems:
                    errors += problems
                else:
                    out.append(call.model_dump())
        if route in ("sql", "both") and not raw_stats:
            errors.append(f"route '{route}' needs at least one stats_query")
        if route in ("semantic", "both") and not raw_search:
            errors.append(f"route '{route}' needs at least one recall_search")
        log = {"attempt": attempt, "n_calls": len(raw_stats) + len(raw_search), "n_valid": len(stats) + len(searches),
               "errors": errors, "raw": reply.text}
        return {"plan_attempts": attempt, "plan_raw": reply.text, "plan_errors": errors, "stats_calls": stats,
                "search_calls": searches, "call_log": [log], "events": [_event("plan", reply)]}

    def after_plan(self, state: AgentState) -> str:
        if state.get("plan_errors") and state["plan_attempts"] < self.max_plan_attempts:
            return "plan"
        return "execute" if state.get("stats_calls") or state.get("search_calls") else "decline"

    def decline(self, state: AgentState) -> dict:
        """No valid tool call after the retries: abstain instead of answering without evidence."""
        text = ("The question could not be turned into a valid query of the NHTSA data: "
                + "; ".join(state.get("plan_errors", [])) + ".")
        return {"answer": text, "citations": [], "problems": [], "declined": True,
                "answers": [{"attempt": 0, "answer": text, "citations": [], "problems": []}]}

    def execute(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        evidence: list[Evidence] = []
        sql: list[str] = []
        for i, call in enumerate(state.get("stats_calls", []), start=1):
            res = self.db.run_stats(StatsQuery(**call), f"S{i}")
            sql.append(res.sql)
            if res.query.group_by is None:
                must = [str(res.value or 0)]
            else:
                must = [str(g) for g, _ in res.rows] or ["0"]
            evidence.append(Evidence(res.id, "stats", res.render(), must))
        seen: set[str] = set()
        for j, call in enumerate(state.get("search_calls", []), start=1):
            q = RecallSearch(**call)
            res = self.search(q, f"X{j}")
            new = [h for h in res.hits if h.campaign not in seen]
            seen |= {h.campaign for h in new}
            if not res.hits:
                evidence.append(Evidence(res.id, "empty", res.render()))
            evidence += [Evidence(h.campaign, "recall", h.render()) for h in new]
        return {"evidence": [asdict(e) for e in evidence], "sql": sql, "events": [_event("execute", tool_s=time.perf_counter() - t0)]}

    def search(self, q: RecallSearch, rid: str) -> SearchResult:
        """Semantic recall search, restricted to the vehicle's campaigns when a vehicle is given."""
        allowed = None
        if q.make or q.model or q.model_year:
            allowed = self.db.vehicle_campaigns(q.make, q.model, q.model_year)
        hits = self.index.search(q.query, q.k, allowed)
        details = self.db.recall_details([c for c, _ in hits])
        out = []
        for c, score in hits:
            d = details[c]
            out.append(RecallHit(c, score, d["report_date"], d["manufacturer"], d["components"] or "", d["potential_units"],
                                 d["defect"], d["consequence"], d["remedy"], bool(d["do_not_drive"])))
        return SearchResult(rid, q, out)

    def answer(self, state: AgentState) -> dict:
        attempt = state.get("answer_attempts", 0) + 1
        evidence_text = "\n".join(e["text"] for e in state.get("evidence", []))
        msgs = [{"role": "system", "content": prompts.answer_system()},
                {"role": "user", "content": prompts.answer_user(state["question"], evidence_text)}]
        if attempt > 1:
            prev = json.dumps({"answer": state.get("answer", ""), "citations": state.get("citations", [])})
            msgs += [{"role": "assistant", "content": prev},
                     {"role": "user", "content": "Your answer failed these checks:\n- " + "\n- ".join(state.get("problems", []))
                      + "\nRewrite the answer so that it passes, using only the evidence."}]
        reply = self.llm.chat(msgs, prompts.ANSWER_SCHEMA, max_tokens=400)
        out = _parse_json(reply.text) or {}
        answer = out.get("answer") if isinstance(out.get("answer"), str) else ""
        cites = [str(c) for c in out.get("citations", []) if isinstance(c, (str, int))] if isinstance(out.get("citations"), list) else []
        return {"answer_attempts": attempt, "answer": answer, "citations": cites, "events": [_event("answer", reply)]}

    def check(self, state: AgentState) -> dict:
        evidence = [Evidence(**e) for e in state.get("evidence", [])]
        problems = check_grounding(state.get("answer", ""), state.get("citations", []), evidence, state["question"])
        record = {"attempt": state["answer_attempts"], "answer": state.get("answer", ""), "citations": state.get("citations", []),
                  "problems": problems}
        return {"problems": problems, "answers": [record]}

    def after_check(self, state: AgentState) -> str:
        if state.get("problems") and state["answer_attempts"] < self.max_answer_attempts:
            return "answer"
        return "end"


def closed_book_answer(llm: ChatModel, question: str, coverage: dict) -> tuple[str, ChatReply]:
    """Baseline without tools: the same LLM answers from its own knowledge."""
    msgs = [{"role": "system", "content": prompts.closed_book_system(coverage)}, {"role": "user", "content": question}]
    reply = llm.chat(msgs, prompts.CLOSED_BOOK_SCHEMA, max_tokens=200)
    out = _parse_json(reply.text) or {}
    return (out.get("answer") if isinstance(out.get("answer"), str) else ""), reply


def classify_only(llm: ChatModel, question: str) -> tuple[str | None, ChatReply]:
    """Run only the classify prompt (used for the separately written routing probes)."""
    msgs = [{"role": "system", "content": prompts.classify_system()}, {"role": "user", "content": question}]
    reply = llm.chat(msgs, prompts.ROUTE_SCHEMA, max_tokens=20)
    out = _parse_json(reply.text)
    route = out.get("route") if out else None
    return (route if route in prompts.ROUTES else None), reply
