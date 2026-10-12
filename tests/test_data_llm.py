import duckdb

from safetyagent.data import CMPL_COLUMNS, RCL_COLUMNS, build_database, component_family
from safetyagent.llm import CachedChat, ChatReply


def test_component_family_mapping():
    assert component_family("SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS") == "SERVICE BRAKES"
    assert component_family("ENGINE AND ENGINE COOLING:ENGINE") == "ENGINE"
    assert component_family("VISIBILITY/WIPER") == "VISIBILITY"
    assert component_family("FUEL/PROPULSION SYSTEM") == "FUEL/PROPULSION SYSTEM"
    assert component_family("Chest Clip, Buckle, Harness") == "UNKNOWN OR OTHER"


def _row(columns: list[str], **values: str) -> str:
    return "\t".join(values.get(c, "") for c in columns)


def test_build_database_from_flat_files(tmp_path):
    rcl = tmp_path / "rcl.txt"
    rcl.write_text("\n".join([
        _row(RCL_COLUMNS, CAMPNO="25V001000", MAKETXT="FORD", MODELTXT="F-150", YEARTXT="2021", COMPNAME="AIR BAGS:FRONTAL",
             MFGNAME="Ford", RCLTYPECD="V", POTAFF="100", RCDATE="20250102", DESC_DEFECT='The "airbag" may not deploy.', DO_NOT_DRIVE="No"),
        _row(RCL_COLUMNS, CAMPNO="25V001000", MAKETXT="FORD", MODELTXT="F-150", YEARTXT="2022", COMPNAME="AIR BAGS:FRONTAL",
             MFGNAME="Ford", RCLTYPECD="V", POTAFF="100", RCDATE="20250102", DESC_DEFECT='The "airbag" may not deploy.', DO_NOT_DRIVE="No"),
        _row(RCL_COLUMNS, CAMPNO="25E002000", MAKETXT="ACME", MODELTXT="LAMP", YEARTXT="2025", COMPNAME="EQUIPMENT",
             MFGNAME="Acme", RCLTYPECD="E", POTAFF="5", RCDATE="20250103"),
    ]) + "\n", encoding="utf-8")
    cmpl = tmp_path / "cmpl.txt"
    cmpl.write_text("\n".join([
        _row(CMPL_COLUMNS, CMPLID="1", ODINO="11", MAKETXT="FORD", MODELTXT="F-150", YEARTXT="2021", CRASH="Y", FIRE="N", INJURED="1",
             DEATHS="0", COMPDESC="AIR BAGS", LDATE="20250105", PROD_TYPE="V", CITY="Austin", VEHICLE_OPERATOR="Jane Doe"),
        _row(CMPL_COLUMNS, CMPLID="2", ODINO="11", MAKETXT="FORD", MODELTXT="F-150", YEARTXT="2021", CRASH="Y", FIRE="N", INJURED="1",
             DEATHS="0", COMPDESC="STEERING", LDATE="20250105", PROD_TYPE="V"),
        _row(CMPL_COLUMNS, CMPLID="3", ODINO="12", MAKETXT="EVENFLO", MODELTXT="SEAT", YEARTXT="2024", COMPDESC="CHILD SEAT",
             LDATE="20250106", PROD_TYPE="C"),
    ]) + "\n", encoding="utf-8")
    counts = build_database(rcl, cmpl, tmp_path / "db.duckdb")
    assert counts == {"recalls": 1, "recall_vehicles": 2, "recall_components": 1, "complaints": 1, "complaint_components": 2}
    con = duckdb.connect(str(tmp_path / "db.duckdb"), read_only=True)
    assert con.execute("SELECT defect, potential_units FROM recalls").fetchone() == ('The "airbag" may not deploy.', 100)
    assert con.execute("SELECT crash, injured FROM complaints").fetchone() == (True, 1)
    cols = {r[0] for r in con.execute("DESCRIBE complaints").fetchall()}
    assert not cols & {"city", "vin", "vehicle_operator", "cdescr", "dealer_name"}  # personal fields are never loaded


class CountingModel:
    name = "counting"
    options = {"temperature": 0.0}

    def __init__(self) -> None:
        self.n = 0

    def chat(self, messages, schema, max_tokens):
        self.n += 1
        return ChatReply('{"route": "sql"}', 1.5, False, 3, 4)


def test_cached_chat_reuses_replies_and_keeps_latency(tmp_path):
    inner = CountingModel()
    llm = CachedChat(inner, tmp_path / "cache.sqlite")
    msgs = [{"role": "user", "content": "hi"}]
    first = llm.chat(msgs, {"type": "object"}, 10)
    second = llm.chat(msgs, {"type": "object"}, 10)
    llm.chat(msgs, {"type": "object", "x": 1}, 10)  # different schema -> new call
    assert inner.n == 2 and not first.cached and second.cached
    assert second.latency_s == 1.5 and second.text == first.text
    assert llm.hits == 1 and llm.misses == 2 and llm.total_model_seconds() == 3.0
