"""Evaluation questions with gold answers computed by gold SQL written separately from the agent's compiler.

The gold SQL below is written independently of :func:`safetyagent.tools.compile_stats`, so a bug in the agent's
SQL compiler shows up as a wrong answer rather than being copied into the gold labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
import numpy as np

# Plain-language names used in questions for each component family (the planner must map them back).
LAY_NAMES: dict[str, str] = {
    "AIR BAGS": "airbags",
    "BACK OVER PREVENTION": "rearview camera",
    "SERVICE BRAKES": "brakes",
    "STEERING": "steering",
    "POWER TRAIN": "transmission",
    "ELECTRICAL SYSTEM": "electrical system",
    "ENGINE": "engine",
    "FUEL SYSTEM": "fuel system",
    "SEAT BELTS": "seat belts",
    "EXTERIOR LIGHTING": "headlights or taillights",
    "SUSPENSION": "suspension",
    "STRUCTURE": "body structure",
    "VISIBILITY": "windshield wipers or windows",
    "LATCHES/LOCKS/LINKAGES": "door latches",
    "SEATS": "seats",
    "WHEELS": "wheels",
    "FORWARD COLLISION AVOIDANCE": "automatic emergency braking",
    "VEHICLE SPEED CONTROL": "cruise control or accelerator",
}

GOLD_SQL: dict[str, str] = {
    "complaint_count": "SELECT count(DISTINCT odino) FROM complaints WHERE make = ? AND model = ? AND model_year = ?",
    "component_complaints": (
        "SELECT count(DISTINCT c.odino) FROM complaints c JOIN complaint_components k ON k.odino = c.odino "
        "WHERE c.make = ? AND c.model = ? AND c.model_year = ? AND k.family = ?"
    ),
    "top_component": (
        "SELECT k.family, count(DISTINCT c.odino) AS n FROM complaints c JOIN complaint_components k ON k.odino = c.odino "
        "WHERE c.make = ? AND c.model = ? AND c.model_year = ? GROUP BY k.family ORDER BY n DESC, k.family"
    ),
    "make_fire_count": "SELECT count(DISTINCT odino) FROM complaints WHERE make = ? AND year(received) = ? AND fire",
    "make_crash_count": "SELECT count(DISTINCT odino) FROM complaints WHERE make = ? AND year(received) = ? AND crash",
    "recall_count": "SELECT count(DISTINCT campaign) FROM recall_vehicles WHERE make = ? AND model = ? AND model_year = ?",
    "recall_units": (
        "SELECT sum(potential_units) FROM recalls WHERE year(report_date) = ? "
        "AND campaign IN (SELECT campaign FROM recall_vehicles WHERE make = ?)"
    ),
    "vehicle_component_campaigns": (
        "SELECT DISTINCT v.campaign FROM recall_vehicles v JOIN recall_components k ON k.campaign = v.campaign "
        "WHERE v.make = ? AND v.model = ? AND v.model_year = ? AND k.family = ? ORDER BY 1"
    ),
}

TEMPLATE_ROUTE = {
    "complaint_count": "sql", "component_complaints": "sql", "top_component": "sql", "make_fire_count": "sql",
    "make_crash_count": "sql", "recall_count": "sql", "recall_units": "sql", "recall_explain": "semantic",
    "complaints_and_recall": "both",
}

# Separately written, freely phrased questions used only to test routing (no gold answers).
ROUTING_PROBES: list[tuple[str, str]] = [
    ("Give me the total number of complaints filed about Subaru Outbacks from model year 2019.", "sql"),
    ("Rank the components that 2022 Kia Sportage owners report problems with.", "sql"),
    ("What's the complaint volume for the 2018 Toyota Camry so far?", "sql"),
    ("Did complaints about Hyundai vehicles mention fires often in 2026? Give me the figure.", "sql"),
    ("In 2024, how many Volkswagen recalls told owners not to drive their cars?", "sql"),
    ("Tally the recall campaigns for the 2017 Nissan Rogue.", "sql"),
    ("My 2021 Ford Bronco's airbag light is on. What did Ford recall the airbags for?", "semantic"),
    ("Tell me about the fuel pump recall on the 2019 Toyota 4Runner.", "semantic"),
    ("What can go wrong with the rearview camera on a 2020 Jeep Gladiator according to the recalls?", "semantic"),
    ("What fix is Honda offering for the 2023 Pilot's seat issue?", "semantic"),
    ("Summarize the danger described in the 2022 Rivian R1T steering recall.", "semantic"),
    ("Is the 2016 Mazda CX-5 under any recall for its headlights, and what is the defect?", "semantic"),
    ("Owners of the 2020 Chevy Bolt keep complaining about the battery - how many complaints are there, and has GM recalled it for that?", "both"),
    ("Count the transmission complaints for the 2019 Honda Pilot and tell me whether a recall explains them.", "both"),
    ("How often do 2021 Ram 1500 owners complain about their brakes, and what recall addresses it?", "both"),
    ("For the 2018 Jeep Grand Cherokee, give the number of electrical complaints and describe any related recall.", "both"),
    ("What does the data say about 2022 Tesla Model 3 seat belt complaints (how many) and the matching recall's remedy?", "both"),
    ("2017 Ford Escape engine fires: number of complaints and the recall that covers them?", "both"),
]


@dataclass
class EvalItem:
    qid: str
    template: str
    route: str
    question: str
    gold_value: Any = None  # int count/sum, or a component family for top_component
    gold_campaigns: list[str] = field(default_factory=list)
    gold_filters: dict = field(default_factory=dict)
    gold_sql: list[str] = field(default_factory=list)
    gold_params: list[list] = field(default_factory=list)
    notes: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["gold_campaigns"] = ";".join(self.gold_campaigns)
        d["gold_filters"] = str(self.gold_filters)
        d["gold_sql"] = " || ".join(self.gold_sql)
        d["gold_params"] = str(self.gold_params)
        return d


MAKE_STYLE = {"BMW": "BMW", "GMC": "GMC", "MERCEDES-BENZ": "Mercedes-Benz", "RAM": "Ram", "KIA": "Kia", "MINI": "MINI"}
SMALL_WORDS = {"AND": "and", "FE": "Fe"}


def display_vehicle(make: str, model: str) -> str:
    """'FORD', 'F-150' -> 'Ford F-150'; 'CHEVROLET', 'SILVERADO 1500' -> 'Chevrolet Silverado 1500'."""
    def part(t: str) -> str:
        return SMALL_WORDS.get(t, t.capitalize() if t.isalpha() and len(t) > 3 else t)
    mk = MAKE_STYLE.get(make, " ".join(w.capitalize() for w in make.split()))
    return f"{mk} {' '.join('-'.join(part(p) for p in t.split('-')) for t in model.split())}"


def _one(con: duckdb.DuckDBPyConnection, sql: str, params: list) -> Any:
    return con.execute(sql, params).fetchone()[0]


def _campaigns(con: duckdb.DuckDBPyConnection, make: str, model: str, year: int, family: str) -> list[str]:
    return [r[0] for r in con.execute(GOLD_SQL["vehicle_component_campaigns"], [make, model, year, family]).fetchall()]


def _pick(rows: list, n: int, rng: np.random.Generator, key=lambda r: r[0], used: set | None = None) -> list:
    """n rows in random order, preferring a key (e.g. make) not used before."""
    order = rng.permutation(len(rows))
    picked, keys = [], set(used or ())
    for i in order:
        if key(rows[i]) not in keys:
            picked.append(rows[i])
            keys.add(key(rows[i]))
        if len(picked) == n:
            return picked
    for i in order:  # not enough distinct keys: fill up
        if rows[i] not in picked:
            picked.append(rows[i])
        if len(picked) == n:
            break
    return picked


def build_eval_set(con: duckdb.DuckDBPyConnection, seed: int = 0, n: dict[str, int] | None = None,
                   exclude_vehicles: set[tuple] | None = None) -> list[EvalItem]:
    """~50 questions over the SQL, semantic and both routes, with gold answers from GOLD_SQL.

    ``exclude_vehicles`` ((make, model, model_year) tuples) keeps a test set disjoint from a development set.
    """
    n = n or {"complaint_count": 5, "component_complaints": 4, "top_component": 4, "make_fire_count": 2,
              "make_crash_count": 2, "recall_count": 4, "recall_units": 3, "recall_explain": 14, "complaints_and_recall": 12}
    rng = np.random.default_rng(seed)
    items: list[EvalItem] = []
    lay = list(LAY_NAMES)
    used_vehicles: set[tuple] = set(exclude_vehicles or ())

    popular = con.execute(
        "SELECT make, model, model_year, count(*) AS n FROM complaints WHERE make <> 'UNKNOWN' AND model_year BETWEEN 2012 AND 2026 "
        "GROUP BY ALL HAVING n >= 150 ORDER BY n DESC, make, model, model_year"
    ).fetchall()

    def add(template: str, question: str, **kw) -> None:
        items.append(EvalItem(f"{template}_{sum(i.template == template for i in items) + 1:02d}", template,
                              TEMPLATE_ROUTE[template], question, **kw))

    # --- SQL route
    for mk, md, yr, _ in _pick(popular, n["complaint_count"], rng):
        p = [mk, md, yr]
        add("complaint_count", f"How many complaints has NHTSA received about the {yr} {display_vehicle(mk, md)}?",
            gold_value=int(_one(con, GOLD_SQL["complaint_count"], p)), gold_sql=[GOLD_SQL["complaint_count"]], gold_params=[p],
            gold_filters={"dataset": "complaints", "measure": "count", "make": mk, "model": md, "model_year": yr})
        used_vehicles.add((mk, md, yr))

    comp_rows = con.execute(
        "SELECT c.make, c.model, c.model_year, k.family, count(DISTINCT c.odino) AS n FROM complaints c "
        "JOIN complaint_components k ON k.odino = c.odino WHERE c.make <> 'UNKNOWN' AND c.model_year BETWEEN 2012 AND 2026 "
        f"AND k.family IN ({', '.join('?' * len(lay))}) GROUP BY ALL HAVING n >= 20 ORDER BY ALL", lay
    ).fetchall()
    comp_rows = [r for r in comp_rows if r[:3] not in used_vehicles]
    for mk, md, yr, fam, _ in _pick(comp_rows, n["component_complaints"], rng):
        p = [mk, md, yr, fam]
        add("component_complaints", f"How many NHTSA complaints about the {yr} {display_vehicle(mk, md)} involve its {LAY_NAMES[fam]}?",
            gold_value=int(_one(con, GOLD_SQL["component_complaints"], p)), gold_sql=[GOLD_SQL["component_complaints"]],
            gold_params=[p], gold_filters={"dataset": "complaints", "measure": "count", "make": mk, "model": md, "model_year": yr, "component": fam})
        used_vehicles.add((mk, md, yr))

    top_rows = []
    for mk, md, yr, _ in popular:
        ranks = con.execute(GOLD_SQL["top_component"], [mk, md, yr]).fetchall()
        if len(ranks) >= 2 and ranks[0][0] != "UNKNOWN OR OTHER" and ranks[0][1] >= 1.25 * ranks[1][1] and (mk, md, yr) not in used_vehicles:
            top_rows.append((mk, md, yr, ranks[0][0], ranks[0][1], ranks[1][1]))
    for mk, md, yr, fam, n1, n2 in _pick(top_rows, n["top_component"], rng):
        p = [mk, md, yr]
        add("top_component", f"Which component do owners of the {yr} {display_vehicle(mk, md)} complain about most often?",
            gold_value=fam, gold_sql=[GOLD_SQL["top_component"]], gold_params=[p],
            gold_filters={"dataset": "complaints", "measure": "count", "make": mk, "model": md, "model_year": yr, "group_by": "component"},
            notes=f"top {n1} vs runner-up {n2}")
        used_vehicles.add((mk, md, yr))

    makes = [r[0] for r in con.execute(
        "SELECT make FROM complaints WHERE year(received) = 2025 AND make <> 'UNKNOWN' GROUP BY make HAVING count(*) >= 1500 ORDER BY make"
    ).fetchall()]
    used_makes: set[str] = set()
    for template, flag in (("make_fire_count", "fire"), ("make_crash_count", "crash")):
        for (mk,) in _pick([(m,) for m in makes], n[template], rng, used=used_makes):
            used_makes.add(mk)
            p = [mk, 2025]
            add(template, f"How many complaints received in 2025 about {display_vehicle(mk, '').strip()} vehicles reported a {flag}?",
                gold_value=int(_one(con, GOLD_SQL[template], p)), gold_sql=[GOLD_SQL[template]], gold_params=[p],
                gold_filters={"dataset": "complaints", "measure": "count", "make": mk, "year": 2025, flag: True})

    rc_rows = [r for r in popular if (r[0], r[1], r[2]) not in used_vehicles]
    for mk, md, yr, _ in _pick(rc_rows, n["recall_count"], rng):
        p = [mk, md, yr]
        val = int(_one(con, GOLD_SQL["recall_count"], p))
        add("recall_count", f"How many recall campaigns cover the {yr} {display_vehicle(mk, md)}?",
            gold_value=val, gold_sql=[GOLD_SQL["recall_count"]], gold_params=[p],
            gold_filters={"dataset": "recalls", "measure": "count", "make": mk, "model": md, "model_year": yr})
        used_vehicles.add((mk, md, yr))

    unit_makes = [r[0] for r in con.execute(
        "SELECT v.make FROM recall_vehicles v JOIN recalls r ON r.campaign = v.campaign WHERE year(r.report_date) = 2025 "
        "AND v.make IN (SELECT make FROM complaints GROUP BY make HAVING count(*) >= 500) "
        "GROUP BY v.make HAVING count(DISTINCT v.campaign) >= 10 ORDER BY v.make"
    ).fetchall()]
    for (mk,) in _pick([(m,) for m in unit_makes], n["recall_units"], rng, used=used_makes):
        used_makes.add(mk)
        p = [2025, mk]
        add("recall_units", f"How many vehicles were potentially affected by {display_vehicle(mk, '').strip()} recalls reported in 2025?",
            gold_value=int(_one(con, GOLD_SQL["recall_units"], p)), gold_sql=[GOLD_SQL["recall_units"]], gold_params=[p],
            gold_filters={"dataset": "recalls", "measure": "potential_units", "make": mk, "year": 2025})

    # --- semantic route: vehicle + component family with 1-2 matching campaigns among >= 4 for the vehicle
    pairs = con.execute(
        f"""
        WITH veh AS (SELECT make, model, model_year, count(DISTINCT campaign) AS n_veh FROM recall_vehicles
                     WHERE model_year BETWEEN 2012 AND 2026 GROUP BY ALL HAVING n_veh >= 4),
             fam AS (SELECT v.make, v.model, v.model_year, k.family, count(DISTINCT v.campaign) AS n_fam
                     FROM recall_vehicles v JOIN recall_components k ON k.campaign = v.campaign
                     WHERE k.family IN ({', '.join('?' * len(lay))}) GROUP BY ALL)
        SELECT f.make, f.model, f.model_year, f.family, f.n_fam, veh.n_veh,
               (SELECT count(*) FROM complaints c WHERE c.make = f.make AND c.model = f.model AND c.model_year = f.model_year) AS n_cmpl
        FROM fam f JOIN veh USING (make, model, model_year)
        WHERE f.n_fam BETWEEN 1 AND 2 ORDER BY ALL
        """, lay
    ).fetchall()
    sem_pool = [r for r in pairs if r[6] >= 25 and r[:3] not in used_vehicles]
    for mk, md, yr, fam, _, n_veh, _ in _pick(sem_pool, n["recall_explain"], rng):
        gold = _campaigns(con, mk, md, yr, fam)
        add("recall_explain", f"Why was the {yr} {display_vehicle(mk, md)} recalled over its {LAY_NAMES[fam]}? "
                              "Describe the defect, the risk and the remedy.",
            gold_campaigns=gold, gold_sql=[GOLD_SQL["vehicle_component_campaigns"]], gold_params=[[mk, md, yr, fam]],
            gold_filters={"make": mk, "model": md, "model_year": yr, "component": fam}, notes=f"{n_veh} campaigns cover this vehicle")
        used_vehicles.add((mk, md, yr))

    # --- both: complaint count for vehicle + family (>= 5) and 1-3 matching recall campaigns
    both_pool = []
    for mk, md, yr, fam, n_fam, n_veh, n_cmpl in con.execute(
        f"""
        WITH fam AS (SELECT v.make, v.model, v.model_year, k.family, count(DISTINCT v.campaign) AS n_fam
                     FROM recall_vehicles v JOIN recall_components k ON k.campaign = v.campaign
                     WHERE v.model_year BETWEEN 2012 AND 2026 AND k.family IN ({', '.join('?' * len(lay))}) GROUP BY ALL),
             cm AS (SELECT c.make, c.model, c.model_year, k.family, count(DISTINCT c.odino) AS n_cmpl FROM complaints c
                    JOIN complaint_components k ON k.odino = c.odino GROUP BY ALL)
        SELECT f.make, f.model, f.model_year, f.family, f.n_fam,
               (SELECT count(DISTINCT campaign) FROM recall_vehicles v WHERE v.make = f.make AND v.model = f.model AND v.model_year = f.model_year),
               cm.n_cmpl
        FROM fam f JOIN cm USING (make, model, model_year, family) WHERE f.n_fam BETWEEN 1 AND 3 AND cm.n_cmpl >= 5 ORDER BY ALL
        """, lay
    ).fetchall():
        if (mk, md, yr) not in used_vehicles and n_veh >= 3:
            both_pool.append((mk, md, yr, fam, n_fam, n_veh, n_cmpl))
    for mk, md, yr, fam, _, n_veh, _ in _pick(both_pool, n["complaints_and_recall"], rng):
        p = [mk, md, yr, fam]
        add("complaints_and_recall", f"How many complaints about the {LAY_NAMES[fam]} of the {yr} {display_vehicle(mk, md)} has NHTSA "
                                     "received, and is there a recall that covers this problem?",
            gold_value=int(_one(con, GOLD_SQL["component_complaints"], p)), gold_campaigns=_campaigns(con, mk, md, yr, fam),
            gold_sql=[GOLD_SQL["component_complaints"], GOLD_SQL["vehicle_component_campaigns"]], gold_params=[p, p],
            gold_filters={"dataset": "complaints", "measure": "count", "make": mk, "model": md, "model_year": yr, "component": fam},
            notes=f"{n_veh} campaigns cover this vehicle")
        used_vehicles.add((mk, md, yr))
    return items


def vehicles_of(items: list[EvalItem]) -> set[tuple]:
    """(make, model, model_year) of every vehicle-level question."""
    return {(i.gold_filters['make'], i.gold_filters['model'], i.gold_filters['model_year']) for i in items if i.gold_filters.get('model')}
