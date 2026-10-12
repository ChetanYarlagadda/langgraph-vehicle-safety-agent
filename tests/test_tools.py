import pytest
from pydantic import ValidationError

from safetyagent.evalset import GOLD_SQL
from safetyagent.tools import RecallSearch, StatsQuery, compile_stats, ungrounded_arguments


def run(db, **kw):
    q = StatsQuery.model_validate(kw, context={"vocab": db.vocab})
    return db.run_stats(q)


def test_component_count_does_not_double_count_complaints(db):
    # complaint 2 lists two brake components of the same family; it must count once
    res = run(db, dataset="complaints", make="FORD", model="F-150", model_year=2020, component="SERVICE BRAKES")
    assert res.value == 2
    gold = db.con.execute(GOLD_SQL["component_complaints"], ["FORD", "F-150", 2020, "SERVICE BRAKES"]).fetchone()[0]
    assert res.value == gold


def test_compiler_matches_hand_written_gold_sql(db):
    cases = [
        ({"dataset": "complaints", "make": "FORD", "model": "F-150", "model_year": 2020}, "complaint_count", ["FORD", "F-150", 2020]),
        ({"dataset": "complaints", "make": "HONDA", "year": 2026, "crash": True}, "make_crash_count", ["HONDA", 2026]),
        ({"dataset": "complaints", "make": "FORD", "year": 2026, "fire": True}, "make_fire_count", ["FORD", 2026]),
        ({"dataset": "recalls", "make": "FORD", "model": "F-150", "model_year": 2020}, "recall_count", ["FORD", "F-150", 2020]),
        ({"dataset": "recalls", "measure": "potential_units", "make": "FORD", "year": 2025}, "recall_units", [2025, "FORD"]),
    ]
    for kw, template, params in cases:
        assert run(db, **kw).value == db.con.execute(GOLD_SQL[template], params).fetchone()[0], template


def test_recall_units_are_summed_once_per_campaign(db):
    # 25V002000 covers three vehicles; its 5,000 units must be counted once
    assert run(db, dataset="recalls", measure="potential_units", make="FORD", year=2025).value == 5000
    by_model = run(db, dataset="recalls", make="FORD", group_by="model")
    assert dict(by_model.rows) == {"FORD F-150": 2, "FORD EXPLORER": 1, "FORD F-150 LIGHTNING": 1}


def test_group_by_component_ranks_families(db):
    res = run(db, dataset="complaints", make="FORD", model="F-150", model_year=2020, group_by="component")
    assert res.rows[0] == ("SERVICE BRAKES", 2)
    gold = db.con.execute(GOLD_SQL["top_component"], ["FORD", "F-150", 2020]).fetchall()
    assert res.rows[0] == tuple(gold[0])


def test_injuries_and_deaths(db):
    assert run(db, dataset="complaints", measure="injuries", make="HONDA").value == 2
    assert run(db, dataset="complaints", measure="deaths").value == 1


def test_vocabulary_canonicalises_and_suggests(db):
    q = StatsQuery.model_validate({"dataset": "complaints", "make": "ford", "model": "F150", "component": "service brakes, hydraulic"},
                                  context={"vocab": db.vocab})
    assert (q.make, q.model, q.component) == ("FORD", "F-150", "SERVICE BRAKES")
    with pytest.raises(ValidationError) as err:
        StatsQuery.model_validate({"dataset": "complaints", "make": "FORDD"}, context={"vocab": db.vocab})
    assert "closest valid values: FORD" in str(err.value)
    with pytest.raises(ValidationError) as err:
        StatsQuery.model_validate({"dataset": "complaints", "component": "BRAKES"}, context={"vocab": db.vocab})
    assert "SERVICE BRAKES" in str(err.value)


def test_cross_field_rules():
    with pytest.raises(ValidationError, match="potential_units"):
        StatsQuery(dataset="complaints", measure="potential_units")
    with pytest.raises(ValidationError, match="crash and fire"):
        StatsQuery(dataset="recalls", crash=True)
    with pytest.raises(ValidationError, match="model given without make"):
        RecallSearch(query="camera", model="F-150")
    assert StatsQuery(dataset="recalls", crash=False).filters() == {}  # false flags do not filter


def test_vehicle_and_coverage_rules(db):
    ctx = {"vocab": db.vocab}
    with pytest.raises(ValidationError, match='set model to "F-150 LIGHTNING"'):
        StatsQuery.model_validate({"dataset": "recalls", "make": "FORD", "model": "F-150", "model_year": 2022}, context=ctx)
    # a vehicle with no records and no similarly named model is a valid query with a zero answer
    assert StatsQuery.model_validate({"dataset": "complaints", "make": "HONDA", "model": "CR-V", "model_year": 2024}, context=ctx)
    with pytest.raises(ValidationError, match="use model_year"):
        StatsQuery.model_validate({"dataset": "complaints", "make": "FORD", "year": 2016}, context=ctx)


def test_arguments_must_be_supported_by_the_question():
    q = "How many complaints received in 2025 about Honda vehicles reported a fire?"
    ok = StatsQuery(dataset="complaints", make="HONDA", year=2025, fire=True)
    assert ungrounded_arguments(ok, q, "s") == []
    bad = StatsQuery(dataset="complaints", make="HONDA", year=2025, fire=True, crash=True, model_year=2019)
    msgs = ungrounded_arguments(bad, q, "s")
    assert any("s.crash" in m for m in msgs) and any("s.model_year: 2019" in m for m in msgs)
    units = StatsQuery(dataset="recalls", measure="potential_units", make="HONDA")
    assert ungrounded_arguments(units, "How many recall campaigns cover the 2018 Honda Accord?", "s")


def test_component_must_match_the_question():
    q = "How many complaints has NHTSA received about the 2025 Toyota Sienna Hybrid?"
    hybrid = StatsQuery(dataset="complaints", make="TOYOTA", model="SIENNA HYBRID", model_year=2025, component="FUEL/PROPULSION SYSTEM")
    assert any("does not mention FUEL/PROPULSION SYSTEM" in m for m in ungrounded_arguments(hybrid, q, "s"))
    q2 = "How many NHTSA complaints about the 2018 Ram 3500 involve its brakes?"
    missing = StatsQuery(dataset="complaints", make="RAM", model="3500", model_year=2018)
    assert any("SERVICE BRAKES" in m for m in ungrounded_arguments(missing, q2, "s"))
    assert ungrounded_arguments(missing.model_copy(update={"component": "SERVICE BRAKES"}), q2, "s") == []
    ranked = StatsQuery(dataset="complaints", make="RAM", model="3500", model_year=2018, component="ENGINE", group_by="component")
    assert any("when ranking by component" in m for m in ungrounded_arguments(ranked, "Which component of the 2018 Ram 3500 ...?", "s"))


def test_values_are_parameters_not_sql(db):
    q = StatsQuery(dataset="complaints", make="X' OR '1'='1")  # no vocabulary context: still only a bound parameter
    sql, params = compile_stats(q)
    assert "OR '1'" not in sql and params == ["X' OR '1'='1"]
    assert db.con.execute(sql, params).fetchone()[0] == 0
