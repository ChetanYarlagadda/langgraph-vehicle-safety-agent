from conftest import COVERAGE, FakeIndex, FakeLLM

from safetyagent.agent import SafetyAgent

Q_SQL = "How many complaints has NHTSA received about the 2020 Ford F-150?"
Q_BOTH = "How many complaints about the airbags of the 2020 Honda CR-V has NHTSA received, and is there a recall that covers this problem?"


def agent(db, llm: FakeLLM) -> SafetyAgent:
    return SafetyAgent(llm, db, FakeIndex(), COVERAGE)


def test_sql_route_happy_path(db):
    llm = FakeLLM(routes=[{"route": "sql"}],
                  plans=[{"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-150", "model_year": 2020}]}],
                  answers=[{"answer": "NHTSA received 2 complaints about the 2020 Ford F-150 [S1].", "citations": ["S1"]}])
    res = agent(db, llm).ask(Q_SQL)
    assert res.route == "sql" and res.grounded and res.plan_attempts == 1 and res.answer_attempts == 1
    assert [k for k, _ in llm.calls] == ["route", "plan", "answer"]
    assert res.evidence[0]["must_mention"] == ["2"]


def test_invalid_tool_call_is_retried_with_the_validation_message(db):
    bad = {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-1500", "model_year": 2020}]}
    good = {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-150", "model_year": 2020}]}
    llm = FakeLLM(routes=[{"route": "sql"}], plans=[bad, good],
                  answers=[{"answer": "There are 2 complaints [S1].", "citations": ["S1"]}])
    res = agent(db, llm).ask(Q_SQL)
    assert res.plan_attempts == 2 and res.call_log[0]["errors"] and not res.call_log[1]["errors"]
    retry_messages = llm.calls[2][1]
    assert "closest valid values: F-150" in retry_messages[-1]["content"]
    assert res.grounded


def test_route_requires_both_tools(db):
    only_sql = {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "HONDA", "model": "CR-V", "model_year": 2020,
                                   "component": "AIR BAGS"}]}
    both = {**only_sql, "recall_searches": [{"query": "airbag", "make": "HONDA", "model": "CR-V", "model_year": 2020}]}
    llm = FakeLLM(routes=[{"route": "both"}], plans=[only_sql, both],
                  answers=[{"answer": "2 complaints [S1]; recall [25V003000] replaces the inflator.", "citations": ["S1", "25V003000"]}])
    res = agent(db, llm).ask(Q_BOTH)
    assert any("needs at least one recall_search" in e for e in res.call_log[0]["errors"])
    assert res.plan_attempts == 2 and res.grounded
    assert {e["id"] for e in res.evidence} == {"S1", "25V003000"}


def test_ungrounded_answer_is_rewritten(db):
    plan = {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-150", "model_year": 2020}]}
    llm = FakeLLM(routes=[{"route": "sql"}], plans=[plan],
                  answers=[{"answer": "About 40 complaints were filed.", "citations": []},
                           {"answer": "NHTSA received 2 complaints [S1].", "citations": ["S1"]}])
    res = agent(db, llm).ask(Q_SQL)
    assert res.answer_attempts == 2 and res.grounded
    assert res.answers[0]["problems"] and not res.answers[1]["problems"]
    assert "failed these checks" in llm.calls[-1][1][-1]["content"]


def test_retries_are_bounded_and_the_agent_declines(db):
    bad = {"stats_queries": [{"dataset": "complaints", "make": "NOPE"}]}
    llm = FakeLLM(routes=["not json"], plans=[bad, bad, bad], answers=[])
    res = agent(db, llm).ask(Q_SQL)
    assert res.route_source == "keyword fallback" and res.route == "sql"
    assert res.plan_attempts == 3 and res.evidence == [] and res.declined
    assert res.answer_attempts == 0 and "could not be turned into a valid query" in res.answer
    assert [k for k, _ in llm.calls] == ["route", "plan", "plan", "plan"]  # no answer call without evidence


def test_year_in_question_must_be_used(db):
    no_year = {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-150"}]}
    good = {"stats_queries": [{"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-150", "model_year": 2020}]}
    llm = FakeLLM(routes=[{"route": "sql"}], plans=[no_year, good], answers=[{"answer": "2 complaints [S1].", "citations": ["S1"]}])
    res = agent(db, llm).ask(Q_SQL)
    assert "mentions 2020 but no filter uses it" in res.call_log[0]["errors"][0]
    assert "rejected by the validator" in llm.calls[2][1][-1]["content"] and res.plan_attempts == 2
