from conftest import COVERAGE, FakeIndex, FakeLLM
from fastapi.testclient import TestClient

from app.main import create_app
from safetyagent.agent import SafetyAgent


def client(db) -> TestClient:
    llm = FakeLLM(routes=[{"route": "semantic"}],
                  plans=[{"recall_searches": [{"query": "camera", "make": "FORD", "model": "F-150", "model_year": 2021}]}],
                  answers=[{"answer": "Recall [25V002000]: the camera image may not display; dealers update the software.",
                            "citations": ["25V002000"]}])
    return TestClient(create_app(lambda: SafetyAgent(llm, db, FakeIndex(), COVERAGE)))


def test_ask_returns_cited_answer(db):
    with client(db) as c:
        r = c.post("/ask", json={"question": "Why was the 2021 Ford F-150 recalled over its rearview camera?"})
        assert r.status_code == 200
        body = r.json()
        assert body["route"] == "semantic" and body["grounded"] and body["citations"] == ["25V002000"]
        assert body["tool_calls"][0]["tool"] == "recall_search" and body["evidence"][0]["id"] == "25V002000"


def test_tools_endpoints_validate_and_run(db):
    with client(db) as c:
        r = c.post("/tools/stats", json={"dataset": "complaints", "make": "honda", "model": "crv", "model_year": 2020})
        assert r.status_code == 200 and r.json()["rows"] == [[2]]
        bad = c.post("/tools/stats", json={"dataset": "complaints", "make": "HONDAA"})
        assert bad.status_code == 422 and "closest valid values: HONDA" in bad.json()["detail"][0]
        hits = c.post("/tools/search", json={"query": "airbag", "make": "HONDA"}).json()
        assert [h["campaign"] for h in hits] == ["25V003000"]
        assert c.post("/ask", json={"question": "short"}).status_code == 422
