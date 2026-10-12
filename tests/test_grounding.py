from safetyagent.grounding import Evidence, check_grounding, cited_ids, numbers_in

EVIDENCE = [
    Evidence("S1", "stats", "[S1] complaints, count | filters: make=FORD, model=F-150, model_year=2020 | result: 1,234 complaints", ["1234"]),
    Evidence("25V002000", "recall", "[25V002000] reported 2025-06-10; potentially affected vehicles: 5,000\n  Defect: camera"),
]
Q = "How many complaints about the 2020 Ford F-150, and is there a recall?"


def test_numbers_ignore_campaign_ids_and_commas():
    assert numbers_in("1,234 complaints and recall 25V002000 in 2020") == {"1234", "2020"}


def test_grounded_answer_passes():
    ans = "NHTSA received 1,234 complaints about the 2020 Ford F-150 [S1]; recall [25V002000] covers 5,000 vehicles."
    assert check_grounding(ans, ["S1", "25V002000"], EVIDENCE, Q) == []


def test_invented_number_and_unknown_citation_are_flagged():
    ans = "There were 1,300 complaints [S1] and recall [24V999000] applies."
    problems = check_grounding(ans, ["S1"], EVIDENCE, Q)
    assert any("24V999000" in p for p in problems)
    assert any("1300" in p for p in problems)
    assert any("does not state the result of [S1]" in p for p in problems)


def test_missing_citations_are_flagged():
    problems = check_grounding("There were 1,234 complaints.", [], EVIDENCE, Q)
    assert any("cites no evidence" in p for p in problems)


def test_inline_citation_lists_are_parsed():
    assert cited_ids("x [S1, 25V002000] y", []) == {"S1", "25V002000"}
