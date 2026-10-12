"""Shared fixtures: a tiny synthetic NHTSA database, a fake recall index and a scripted fake LLM."""

from __future__ import annotations

import json

import duckdb
import pytest

from safetyagent.llm import ChatReply
from safetyagent.tools import SafetyDB

SCHEMA_SQL = """
CREATE TABLE complaints (odino BIGINT, make VARCHAR, model VARCHAR, model_year INTEGER, received DATE, incident_date DATE,
    crash BOOLEAN, fire BOOLEAN, injured INTEGER, deaths INTEGER, medical_attention BOOLEAN, towed BOOLEAN, state VARCHAR, miles INTEGER);
CREATE TABLE complaint_components (odino BIGINT, component VARCHAR, family VARCHAR);
CREATE TABLE recalls (campaign VARCHAR, manufacturer VARCHAR, report_date DATE, potential_units BIGINT, initiated_by VARCHAR,
    do_not_drive BOOLEAN, park_outside BOOLEAN, defect VARCHAR, consequence VARCHAR, remedy VARCHAR);
CREATE TABLE recall_vehicles (campaign VARCHAR, make VARCHAR, model VARCHAR, model_year INTEGER);
CREATE TABLE recall_components (campaign VARCHAR, component VARCHAR, family VARCHAR);
INSERT INTO complaints VALUES
 (1, 'FORD', 'F-150', 2020, '2025-02-01', NULL, true,  false, 1, 0, false, false, 'TX', 1000),
 (2, 'FORD', 'F-150', 2020, '2025-03-01', NULL, false, false, 0, 0, false, false, 'TX', 2000),
 (3, 'FORD', 'F-150', 2021, '2026-01-05', NULL, false, true,  0, 0, false, false, 'CA', 3000),
 (4, 'HONDA', 'CR-V', 2020, '2025-05-01', NULL, false, false, 0, 0, false, false, 'OH', 4000),
 (5, 'HONDA', 'CR-V', 2020, '2026-02-01', NULL, true,  false, 2, 1, true,  true,  'OH', 5000);
INSERT INTO complaint_components VALUES
 (1, 'SERVICE BRAKES: HYDRAULIC', 'SERVICE BRAKES'), (1, 'ENGINE', 'ENGINE'),
 (2, 'SERVICE BRAKES', 'SERVICE BRAKES'), (2, 'SERVICE BRAKES, HYDRAULIC', 'SERVICE BRAKES'),
 (3, 'ELECTRICAL SYSTEM', 'ELECTRICAL SYSTEM'), (4, 'AIR BAGS', 'AIR BAGS'),
 (5, 'AIR BAGS: FRONTAL', 'AIR BAGS'), (5, 'STEERING', 'STEERING');
INSERT INTO recalls VALUES
 ('20V001000', 'Ford Motor Company', '2020-03-01', 1000, 'MFR', false, false, 'A brake hose may leak fluid.', 'Reduced braking increases the risk of a crash.', 'Dealers will replace the hose.'),
 ('25V002000', 'Ford Motor Company', '2025-06-10', 5000, 'MFR', false, false, 'The rearview camera image may not display.', 'Reduced rear visibility increases the risk of a crash.', 'Dealers will update the software.'),
 ('25V003000', 'American Honda Motor Co.', '2025-08-20', 300, 'MFR', true, false, 'The airbag inflator may rupture.', 'Metal fragments may strike occupants.', 'Dealers will replace the inflator.');
INSERT INTO recall_vehicles VALUES
 ('20V001000', 'FORD', 'F-150', 2020), ('25V002000', 'FORD', 'F-150', 2020), ('25V002000', 'FORD', 'F-150', 2021),
 ('25V002000', 'FORD', 'EXPLORER', 2021), ('25V002000', 'FORD', 'F-150 LIGHTNING', 2022), ('25V003000', 'HONDA', 'CR-V', 2020);
INSERT INTO recall_components VALUES
 ('20V001000', 'SERVICE BRAKES, HYDRAULIC', 'SERVICE BRAKES'), ('25V002000', 'BACK OVER PREVENTION: DISPLAY', 'BACK OVER PREVENTION'),
 ('25V003000', 'AIR BAGS: FRONTAL', 'AIR BAGS');
"""

COVERAGE = {"complaints_from": "2025-01-01", "complaints_to": "2026-10-07", "recalls_from": "2010-01-01", "recalls_to": "2026-10-06"}


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(SCHEMA_SQL)
    return c


@pytest.fixture
def db(con) -> SafetyDB:
    return SafetyDB(":memory:", con=con)


class FakeIndex:
    """Returns the allowed campaigns in id order (or all known campaigns) with decreasing scores."""

    campaigns = ["20V001000", "25V002000", "25V003000"]

    def search(self, query: str, k: int = 5, allowed: list[str] | None = None) -> list[tuple[str, float]]:
        pool = self.campaigns if allowed is None else [c for c in self.campaigns if c in allowed]
        return [(c, 1.0 - i / 10) for i, c in enumerate(pool[:k])]


class FakeLLM:
    """Scripted chat model: routes, plans and answers are popped from queues according to the requested schema."""

    name = "fake"

    def __init__(self, routes: list, plans: list, answers: list) -> None:
        self.routes, self.plans, self.answers = list(routes), list(plans), list(answers)
        self.calls: list[tuple[str, list[dict]]] = []

    def chat(self, messages: list[dict], schema: dict, max_tokens: int) -> ChatReply:
        props = schema.get("properties", {})
        if "route" in props:
            kind, queue = "route", self.routes
        elif "stats_queries" in props or "recall_searches" in props:
            kind, queue = "plan", self.plans
        else:
            kind, queue = "answer", self.answers
        self.calls.append((kind, messages))
        item = queue.pop(0)
        text = item if isinstance(item, str) else json.dumps(item)
        return ChatReply(text, 0.1, False, 10, 5)
