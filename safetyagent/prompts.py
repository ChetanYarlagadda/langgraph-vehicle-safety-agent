"""Prompts and JSON schemas for the classify, plan and answer nodes (and the closed-book baseline)."""

from __future__ import annotations

import json

from safetyagent.data import COMPONENT_FAMILIES

ROUTES = ("sql", "semantic", "both")

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": list(ROUTES)}},
    "required": ["route"],
}

_STATS_ITEM = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string", "enum": ["complaints", "recalls"]},
        "measure": {"type": "string", "enum": ["count", "potential_units", "injuries", "deaths"]},
        "make": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"]},
        "model_year": {"type": ["integer", "null"]},
        "component": {"type": ["string", "null"]},
        "year": {"type": ["integer", "null"]},
        "crash": {"type": ["boolean", "null"]},
        "fire": {"type": ["boolean", "null"]},
        "do_not_drive": {"type": ["boolean", "null"]},
        "group_by": {"type": ["string", "null"], "enum": ["component", "make", "model", "model_year", "year", None]},
        "limit": {"type": "integer"},
    },
}
_SEARCH_ITEM = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "make": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"]},
        "model_year": {"type": ["integer", "null"]},
        "k": {"type": "integer"},
    },
}
# Every field is required (nullable): Ollama's grammar emits optional properties only in schema order, so a model that
# skips model_year and writes component next can never go back to add it.
_STATS_ITEM["required"] = list(_STATS_ITEM["properties"])
_SEARCH_ITEM["required"] = list(_SEARCH_ITEM["properties"])


def plan_schema(route: str) -> dict:
    """JSON schema for the planner's output; the route decides which tools may be called."""
    props: dict = {}
    if route in ("sql", "both"):
        props["stats_queries"] = {"type": "array", "items": _STATS_ITEM, "minItems": 1, "maxItems": 3}
    if route in ("semantic", "both"):
        props["recall_searches"] = {"type": "array", "items": _SEARCH_ITEM, "minItems": 1, "maxItems": 2}
    return {"type": "object", "properties": props, "required": list(props)}


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "citations"],
}
CLOSED_BOOK_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def classify_system() -> str:
    return (
        "You route questions about U.S. NHTSA vehicle safety data to the right tools.\n"
        "Routes:\n"
        '- "sql": the question asks for a number, a total or a ranking: how many complaints or recalls, how many vehicles '
        "were affected, which component or model year is most common. Answered by SQL over complaint and recall tables.\n"
        '- "semantic": the question asks what a recall is about: why a vehicle was recalled, what the defect is, the '
        "safety risk or the remedy. Answered by searching recall summaries.\n"
        '- "both": the question needs a number AND a recall description, e.g. a complaint count together with whether '
        "a recall covers the problem.\n"
        "Examples:\n"
        "Q: Which model years of the Chevrolet Equinox drew the most complaints? -> sql\n"
        "Q: What problem led to the recall of the 2020 Kia Telluride's seat belts? -> semantic\n"
        "Q: How many owners of the 2018 Nissan Altima reported crashes, and has it been recalled for steering problems? -> both\n"
        'Return JSON {"route": "sql" | "semantic" | "both"}.'
    )


EXAMPLES = [
    ("Which model years of the Chevrolet Equinox drew the most complaints?", "sql",
     {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "CHEVROLET", "model": "EQUINOX", "group_by": "model_year", "limit": 5}]}),
    ("What problem led to the recall of the 2020 Kia Telluride's seat belts?", "semantic",
     {"recall_searches": [{"query": "seat belt problem", "make": "KIA", "model": "TELLURIDE", "model_year": 2020, "k": 5}]}),
    ("How many owners of the 2018 Nissan Altima reported crashes, and has it been recalled for steering problems?", "both",
     {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "NISSAN", "model": "ALTIMA", "model_year": 2018, "crash": True}],
      "recall_searches": [{"query": "steering problem", "make": "NISSAN", "model": "ALTIMA", "model_year": 2018, "k": 5}]}),
]

ROUTE_RULE = {
    "sql": "Use only stats_queries.",
    "semantic": "Use only recall_searches.",
    "both": "Use at least one stats_query AND at least one recall_search.",
}


def plan_system(coverage: dict) -> str:
    families = "\n".join(f"  {name} ({gloss})" for name, gloss in COMPONENT_FAMILIES.items())
    examples = "\n".join(f"Q: {q} (route {r})\nA: {json.dumps(a)}" for q, r, a in EXAMPLES)
    return (
        "You turn a question about U.S. NHTSA vehicle safety data into tool calls. Return JSON only.\n\n"
        "Data:\n"
        f"- complaints: vehicle owner complaints received by NHTSA from {coverage['complaints_from']} to "
        f"{coverage['complaints_to']} (make, model, model_year, received date, crash, fire, injuries, deaths, components).\n"
        f"- recalls: vehicle recall campaigns reported from {coverage['recalls_from']} to {coverage['recalls_to']} "
        "(report date, potentially affected vehicles, do-not-drive advisory, components, and the vehicles covered).\n\n"
        "Tool stats_queries (SQL; one number or a ranked list):\n"
        '  dataset: "complaints" or "recalls"\n'
        '  measure: "count" for how many complaints or recall campaigns; "potential_units" (recalls only) only when the '
        'question asks how many vehicles were affected; "injuries" or "deaths" (complaints only)\n'
        '  make, model: NHTSA spelling in capitals, e.g. "FORD" "F-150", "HONDA" "CR-V", "TESLA" "MODEL Y"\n'
        "  model_year: the vehicle's model year, e.g. 'the 2019 Honda Civic' -> 2019\n"
        "  component: one component family from the list below\n"
        "  year: only when the question restricts the calendar year a complaint was received or a recall was reported "
        "(e.g. 'received in 2025'); never the vehicle's model year\n"
        "  crash / fire: true keeps only complaints that report a crash / a fire; only when the question asks about it\n"
        "  do_not_drive: true keeps only recalls with a do-not-drive advisory\n"
        '  group_by: "component", "make", "model", "model_year" or "year" to rank groups (which ... is most common)\n'
        "  limit: number of groups to return\n"
        "  Set a filter to null when the question does not restrict it.\n"
        "Tool recall_searches (semantic search over recall defect summaries):\n"
        "  query: the problem to look for, in plain words\n"
        "  make, model, model_year: restrict to recalls covering this vehicle\n"
        "  k: number of recalls to return (5)\n\n"
        f"Component families:\n{families}\n\n"
        f"Examples:\n{examples}"
    )


def plan_user(question: str, route: str) -> str:
    return f"Route: {route}. {ROUTE_RULE[route]}\nQuestion: {question}"


def plan_retry_user(question: str, route: str, previous: str, errors: list[str]) -> str:
    """Single-turn retry message: the rejected calls and the validator's messages (small models tend to copy a prior turn)."""
    return (
        f"{plan_user(question, route)}\n\nA previous attempt returned these tool calls:\n{previous}\n"
        "They were rejected by the validator:\n- " + "\n- ".join(errors) + "\n"
        "Return new tool calls for the question that fix every problem listed: change the rejected fields as the messages "
        "say and keep the other fields."
    )


def answer_system() -> str:
    return (
        "You answer questions about U.S. NHTSA vehicle safety data for a safety analyst, using ONLY the evidence given.\n"
        "Rules:\n"
        "1. Cite the evidence id in square brackets right after each fact, e.g. [S1] or [25V123000].\n"
        "2. Copy numbers exactly as they appear in the evidence; never estimate, add up or round.\n"
        "3. If the evidence does not contain the answer, say that the data does not show it.\n"
        "4. When describing a recall, give its campaign number, the defect, the risk and the remedy briefly.\n"
        "5. Keep the answer under 120 words.\n"
        'Return JSON {"answer": "...", "citations": ["S1", "25V123000"]}.'
    )


def answer_user(question: str, evidence_text: str) -> str:
    return f"Question: {question}\n\nEvidence:\n{evidence_text}"


def closed_book_system(coverage: dict) -> str:
    return (
        "You answer questions about U.S. NHTSA vehicle safety data from your own knowledge, without tools. The "
        f"complaint data covers complaints received {coverage['complaints_from']} to {coverage['complaints_to']}; the "
        f"recall data covers campaigns reported {coverage['recalls_from']} to {coverage['recalls_to']}. Give a direct "
        "answer with a specific number, component or recall campaign number when the question asks for one, in 1-3 "
        'sentences. Return JSON {"answer": "..."}.'
    )
