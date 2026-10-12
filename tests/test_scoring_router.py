import math

from safetyagent.evalset import EvalItem, display_vehicle
from safetyagent.router import keyword_route
from safetyagent.scoring import (answer_correct, args_correct, evidence_has_gold, failure_stage, mcnemar_exact, wilson)

BOTH = EvalItem("b1", "complaints_and_recall", "both", "q", gold_value=1234, gold_campaigns=["25V002000"],
                gold_filters={"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-150", "model_year": 2020,
                              "component": "SERVICE BRAKES"})


def test_answer_correct_needs_value_and_campaign():
    assert answer_correct(BOTH, "1,234 complaints [S1]; recall 25V002000 covers it.", [])
    assert not answer_correct(BOTH, "1,234 complaints [S1]; no recall found.", [])
    assert not answer_correct(BOTH, "1,235 complaints; recall [25V002000].", ["25V002000"])
    top = EvalItem("t", "top_component", "sql", "q", gold_value="POWER TRAIN")
    assert answer_correct(top, "Most complaints concern the powertrain [S1].", [])


def test_args_correct_is_exact():
    stats = [{"dataset": "complaints", "measure": "count", "make": "FORD", "model": "F-150", "model_year": 2020,
              "component": "SERVICE BRAKES", "crash": None, "fire": False, "limit": 5}]
    search = [{"query": "brakes", "make": "FORD", "model": "F-150", "model_year": 2020}]
    assert args_correct(BOTH, stats, search)
    assert not args_correct(BOTH, [{**stats[0], "year": 2025}], search)
    assert not args_correct(BOTH, stats, [{**search[0], "model_year": 2021}])


def test_evidence_and_failure_stage():
    ev = [{"id": "S1", "kind": "stats", "text": "[S1] ... | filters: model_year=2020 | result: 1,234 complaints", "must_mention": ["1234"]},
          {"id": "25V002000", "kind": "recall", "text": "...", "must_mention": []}]
    assert evidence_has_gold(BOTH, ev)
    assert not evidence_has_gold(BOTH, ev[:1])
    assert failure_stage(True, True, False) == "answer synthesis"
    assert failure_stage(False, False, False) == "wrong route"
    assert failure_stage(True, False, False) == "tool call"


def test_statistics_helpers():
    lo, hi = wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(0, 6), 2 / 64)


def test_keyword_router_and_display_names():
    assert keyword_route("How many recall campaigns cover the 2018 Honda Accord?") == "sql"
    assert keyword_route("Why was the 2019 Jeep Cherokee recalled over its brakes?") == "semantic"
    assert keyword_route("How many complaints about the brakes ..., and is there a recall that covers this problem?") == "both"
    assert display_vehicle("CHRYSLER", "TOWN AND COUNTRY") == "Chrysler Town and Country"
    assert display_vehicle("HONDA", "CR-V") == "Honda CR-V"
