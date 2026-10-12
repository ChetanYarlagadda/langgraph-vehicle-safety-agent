"""Agent tools: Pydantic-validated arguments, a parameterised SQL compiler and their execution on DuckDB.

The LLM never writes SQL. It fills a :class:`StatsQuery` (counts and sums over complaints or recalls) or a
:class:`RecallSearch` (semantic search over recall summaries). Both are validated against the database
vocabulary (makes, models, component families); a failed validation returns messages with the closest valid
values, which the agent feeds back to the LLM for a retry.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from safetyagent.data import COMPONENT_FAMILIES, component_family

GROUP_KEYS = ("component", "make", "model", "model_year", "year")


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


@dataclass
class Vocabulary:
    """Valid makes, models per make, model years per vehicle, data years and component families, read from the database."""

    makes: set[str]
    models: dict[str, set[str]]
    years: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    complaint_years: tuple[int, int] = (1995, 2100)
    recall_years: tuple[int, int] = (1966, 2100)
    families: tuple[str, ...] = tuple(COMPONENT_FAMILIES)

    @classmethod
    def from_db(cls, con: duckdb.DuckDBPyConnection) -> "Vocabulary":
        rows = con.execute(
            "SELECT DISTINCT make, model, model_year FROM (SELECT make, model, model_year FROM complaints "
            "UNION ALL SELECT make, model, model_year FROM recall_vehicles) WHERE make IS NOT NULL AND model IS NOT NULL"
        ).fetchall()
        models: dict[str, set[str]] = {}
        years: dict[tuple[str, str], set[int]] = {}
        for make, model, year in rows:
            models.setdefault(make, set()).add(model)
            if year is not None:
                years.setdefault((make, model), set()).add(int(year))
        cy = con.execute("SELECT min(year(received)), max(year(received)) FROM complaints").fetchone()
        ry = con.execute("SELECT min(year(report_date)), max(year(report_date)) FROM recalls").fetchone()
        return cls(makes=set(models), models=models, years=years, complaint_years=(int(cy[0]), int(cy[1])),
                   recall_years=(int(ry[0]), int(ry[1])))

    def similar_vehicles(self, make: str, model: str, model_year: int) -> list[str]:
        """Models of ``make`` that exist for ``model_year`` and whose name contains (or is contained in) ``model``."""
        if model_year in self.years.get((make, model), set()):
            return []
        return sorted(m for m in self.models.get(make, set())
                      if m != model and (model in m or m in model) and model_year in self.years.get((make, m), set()))

    @staticmethod
    def _resolve(value: str, options: set[str], what: str) -> str:
        """Exact match, else a unique punctuation/space-insensitive match, else ValueError with suggestions."""
        v = value.strip().upper()
        if v in options:
            return v
        loose = [o for o in options if _alnum(o) == _alnum(v)]
        if len(loose) == 1:
            return loose[0]
        close = difflib.get_close_matches(v, sorted(options), n=4, cutoff=0.5)
        hint = f"; closest valid values: {', '.join(close)}" if close else ""
        raise ValueError(f"{what} '{value}' is not in the NHTSA data{hint}")

    def make(self, value: str) -> str:
        return self._resolve(value, self.makes, "make")

    def model(self, make: str, value: str) -> str:
        return self._resolve(value, self.models.get(make, set()), f"model for make {make}")

    def family(self, value: str) -> str:
        v = value.strip().upper()
        if v in self.families:
            return v
        mapped = component_family(v)
        if mapped != "UNKNOWN OR OTHER" or v.startswith("UNKNOWN"):
            return mapped
        words = [w.rstrip("S") for w in re.findall(r"[A-Z]+", v) if len(w) >= 4]
        close = [f for f in self.families if any(w in f"{f} {COMPONENT_FAMILIES.get(f, '').upper()}" for w in words)][:4]
        close = close or difflib.get_close_matches(v, list(self.families), n=3, cutoff=0.4)
        hint = f"; closest valid values: {', '.join(close)}" if close else ""
        raise ValueError(f"component '{value}' is not a component family{hint}")


def _vocab(info: ValidationInfo) -> Vocabulary | None:
    return (info.context or {}).get("vocab") if info.context else None


class VehicleFilter(BaseModel):
    """Make / model / model-year filter shared by both tools."""

    model_config = ConfigDict(extra="forbid")

    make: str | None = Field(None, description="NHTSA make in capitals, e.g. FORD")
    model: str | None = Field(None, description="NHTSA model in capitals, e.g. F-150")
    model_year: int | None = Field(None, ge=1950, le=2028, description="vehicle model year")

    @field_validator("make", "model", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            return v.upper() or None
        return v

    @model_validator(mode="after")
    def _check_vehicle(self, info: ValidationInfo):
        if self.model and not self.make:
            raise ValueError("model given without make")
        vocab = _vocab(info)
        if vocab is not None and self.make:
            self.make = vocab.make(self.make)
            if self.model:
                self.model = vocab.model(self.make, self.model)
                if self.model_year is not None:
                    alt = vocab.similar_vehicles(self.make, self.model, self.model_year)
                    if len(alt) == 1:
                        raise ValueError(f"NHTSA has no {self.model_year} {self.make} {self.model}; set model to \"{alt[0]}\"")
                    if alt:
                        raise ValueError(f"NHTSA has no {self.model_year} {self.make} {self.model}; models with that model "
                                         f"year: {', '.join(alt[:4])}")
        return self


class StatsQuery(VehicleFilter):
    """Count complaints or recall campaigns (or sum units, injuries, deaths) matching filters; optionally ranked by a group."""

    dataset: Literal["complaints", "recalls"]
    measure: Literal["count", "potential_units", "injuries", "deaths"] = "count"
    component: str | None = Field(None, description="component family")
    year: int | None = Field(None, ge=2010, le=2026, description="year the complaint was received / recall was reported")
    crash: bool | None = Field(None, description="true: only complaints reporting a crash")
    fire: bool | None = Field(None, description="true: only complaints reporting a fire")
    do_not_drive: bool | None = Field(None, description="true: only recalls with a do-not-drive advisory")
    group_by: Literal["component", "make", "model", "model_year", "year"] | None = None
    limit: int = Field(5, ge=1, le=20)

    @field_validator("component", mode="before")
    @classmethod
    def _component_blank(cls, v: Any) -> Any:
        return (v.strip().upper() or None) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_query(self, info: ValidationInfo):
        if self.measure == "potential_units" and self.dataset != "recalls":
            raise ValueError("measure 'potential_units' is only available for dataset 'recalls'")
        if self.measure in ("injuries", "deaths") and self.dataset != "complaints":
            raise ValueError(f"measure '{self.measure}' is only available for dataset 'complaints'")
        if self.dataset == "recalls" and (self.crash or self.fire):
            raise ValueError("crash and fire filters only apply to dataset 'complaints'")
        if self.dataset == "complaints" and self.do_not_drive:
            raise ValueError("do_not_drive only applies to dataset 'recalls'")
        vocab = _vocab(info)
        if vocab is not None and self.year is not None:
            lo, hi = vocab.complaint_years if self.dataset == "complaints" else vocab.recall_years
            if not lo <= self.year <= hi:
                raise ValueError(f"'year' filters the year a {self.dataset[:-1]} was received/reported, and the data only covers "
                                 f"{lo}-{hi}; use model_year for the vehicle's model year")
        if self.component:
            self.component = vocab.family(self.component) if vocab else self.component
        return self

    def filters(self) -> dict[str, Any]:
        """Active filters (true flags only)."""
        keys = ["make", "model", "model_year", "component", "year"]
        out = {k: getattr(self, k) for k in keys if getattr(self, k) is not None}
        out.update({k: True for k in ("crash", "fire", "do_not_drive") if getattr(self, k)})
        return out


class RecallSearch(VehicleFilter):
    """Semantic search over recall defect and consequence summaries, optionally restricted to one vehicle."""

    query: str = Field(min_length=3, max_length=300, description="the problem to look for, in plain words")
    k: int = Field(5, ge=1, le=10)


@dataclass
class StatsResult:
    id: str
    query: StatsQuery
    sql: str
    params: list[Any]
    columns: list[str]
    rows: list[tuple]
    seconds: float = 0.0

    @property
    def value(self) -> float | int | None:
        """The single result of an ungrouped query."""
        return self.rows[0][0] if self.rows and self.query.group_by is None else None

    def render(self) -> str:
        q = self.query
        unit = {"count": "complaints" if q.dataset == "complaints" else "recall campaigns",
                "potential_units": "potentially affected vehicles", "injuries": "injuries", "deaths": "deaths"}[q.measure]
        filt = ", ".join(f"{k}={v}" for k, v in q.filters().items()) or "none"
        head = f"[{self.id}] {q.dataset}, {q.measure}" + (f" by {q.group_by}" if q.group_by else "") + f" | filters: {filt}"
        if q.group_by is None:
            v = self.value or 0
            return f"{head} | result: {v:,} {unit}"
        body = "; ".join(f"{g}: {v:,}" for g, v in self.rows) or "no matching records"
        return f"{head} | result ({unit}): {body}"


@dataclass
class RecallHit:
    campaign: str
    score: float
    report_date: str
    manufacturer: str
    components: str
    potential_units: int | None
    defect: str
    consequence: str
    remedy: str
    do_not_drive: bool = False

    def render(self, max_chars: int = 420) -> str:
        cut = lambda s, n: (s or "").strip()[:n] + ("..." if len(s or "") > n else "")  # noqa: E731
        units = f"{self.potential_units:,}" if self.potential_units is not None else "unknown"
        return (
            f"[{self.campaign}] reported {self.report_date} by {self.manufacturer}; components: {self.components}; "
            f"potentially affected vehicles: {units}" + ("; DO NOT DRIVE advisory" if self.do_not_drive else "") + "\n"
            f"  Defect: {cut(self.defect, max_chars)}\n  Risk: {cut(self.consequence, 260)}\n  Remedy: {cut(self.remedy, 260)}"
        )


@dataclass
class SearchResult:
    id: str
    query: RecallSearch
    hits: list[RecallHit] = field(default_factory=list)
    seconds: float = 0.0

    def render(self) -> str:
        if not self.hits:
            return f"[{self.id}] recall search '{self.query.query}': no recalls found for this vehicle"
        return "\n".join(h.render() for h in self.hits)


def compile_stats(q: StatsQuery) -> tuple[str, list[Any]]:
    """Compile a validated StatsQuery into one parameterised DuckDB query."""
    params: list[Any] = []
    where: list[str] = []
    joins: list[str] = []
    if q.dataset == "complaints":
        uid, val = "c.odino", {"injuries": "c.injured", "deaths": "c.deaths"}.get(q.measure, "1")
        src = "complaints c"
        for col in ("make", "model", "model_year"):
            if getattr(q, col) is not None:
                where.append(f"c.{col} = ?")
                params.append(getattr(q, col))
        if q.year is not None:
            where.append("year(c.received) = ?")
            params.append(q.year)
        if q.crash:
            where.append("c.crash")
        if q.fire:
            where.append("c.fire")
        if q.component:
            where.append("c.odino IN (SELECT odino FROM complaint_components WHERE family = ?)")
            params.append(q.component)
        key = {"component": "k.family", "make": "c.make", "model": "c.make || ' ' || c.model",
               "model_year": "c.model_year", "year": "year(c.received)"}.get(q.group_by or "")
        if q.group_by == "component":
            joins.append("JOIN (SELECT DISTINCT odino, family FROM complaint_components) k ON k.odino = c.odino")
    else:
        uid, val = "r.campaign", "r.potential_units" if q.measure == "potential_units" else "1"
        src = "recalls r"
        veh = [(col, getattr(q, col)) for col in ("make", "model", "model_year") if getattr(q, col) is not None]
        if veh:
            where.append("r.campaign IN (SELECT campaign FROM recall_vehicles WHERE " + " AND ".join(f"{c} = ?" for c, _ in veh) + ")")
            params.extend(v for _, v in veh)
        if q.component:
            where.append("r.campaign IN (SELECT campaign FROM recall_components WHERE family = ?)")
            params.append(q.component)
        if q.year is not None:
            where.append("year(r.report_date) = ?")
            params.append(q.year)
        if q.do_not_drive:
            where.append("r.do_not_drive")
        key = {"component": "k.family", "make": "v.make", "model": "v.make || ' ' || v.model",
               "model_year": "v.model_year", "year": "year(r.report_date)"}.get(q.group_by or "")
        if q.group_by == "component":
            joins.append("JOIN (SELECT DISTINCT campaign, family FROM recall_components) k ON k.campaign = r.campaign")
        elif q.group_by in ("make", "model", "model_year"):
            cond = "".join(f" AND v.{c} = ?" for c, _ in veh)
            joins.append(f"JOIN recall_vehicles v ON v.campaign = r.campaign{cond}")
            # join parameters come before the WHERE parameters
            params = [v for _, v in veh] + params

    agg = "count(*)" if q.measure == "count" else "coalesce(sum(val), 0)"
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    join_sql = (" " + " ".join(joins)) if joins else ""
    if q.group_by is None:
        inner = f"SELECT DISTINCT {uid} AS uid, {val} AS val FROM {src}{join_sql}{where_sql}"
        return f"SELECT {agg} AS value FROM ({inner})", params
    inner = f"SELECT DISTINCT {key} AS grp, {uid} AS uid, {val} AS val FROM {src}{join_sql}{where_sql}"
    sql = f"SELECT grp, {agg} AS value FROM ({inner}) GROUP BY grp ORDER BY value DESC, grp LIMIT {int(q.limit)}"
    return sql, params


class SafetyDB:
    """Read-only DuckDB access for the agent's tools."""

    def __init__(self, db_path: Path | str, con: duckdb.DuckDBPyConnection | None = None) -> None:
        self.con = con if con is not None else duckdb.connect(str(db_path), read_only=True)
        self.con.execute("SET threads TO 2")
        self.vocab = Vocabulary.from_db(self.con)

    def run_stats(self, q: StatsQuery, rid: str = "S1") -> StatsResult:
        sql, params = compile_stats(q)
        cur = self.con.execute(sql, params)
        rows = [tuple(r) for r in cur.fetchall()]
        return StatsResult(rid, q, sql, params, [d[0] for d in cur.description], rows)

    def vehicle_campaigns(self, make: str | None, model: str | None, model_year: int | None) -> list[str]:
        """Recall campaigns covering a (partial) vehicle specification."""
        conds, params = [], []
        for col, v in (("make", make), ("model", model), ("model_year", model_year)):
            if v is not None:
                conds.append(f"{col} = ?")
                params.append(v)
        if not conds:
            return [r[0] for r in self.con.execute("SELECT campaign FROM recalls ORDER BY campaign").fetchall()]
        sql = "SELECT DISTINCT campaign FROM recall_vehicles WHERE " + " AND ".join(conds) + " ORDER BY campaign"
        return [r[0] for r in self.con.execute(sql, params).fetchall()]

    def recall_details(self, campaigns: list[str]) -> dict[str, dict]:
        """Recall fields plus a component list for each campaign."""
        if not campaigns:
            return {}
        rows = self.con.execute(
            """
            SELECT r.campaign, strftime(r.report_date, '%Y-%m-%d'), r.manufacturer, r.potential_units, r.defect,
                   r.consequence, r.remedy, r.do_not_drive,
                   (SELECT string_agg(DISTINCT component, '; ' ORDER BY component) FROM recall_components k WHERE k.campaign = r.campaign)
            FROM recalls r WHERE r.campaign IN (SELECT unnest(?))
            """,
            [campaigns],
        ).fetchall()
        keys = ["campaign", "report_date", "manufacturer", "potential_units", "defect", "consequence", "remedy", "do_not_drive", "components"]
        return {r[0]: dict(zip(keys, r)) for r in rows}


FLAG_CUES = {
    "crash": r"crash|collision|accident|wreck",
    "fire": r"fire|burn|flame|smok",
    "do_not_drive": r"not (to )?drive|do-not-drive|don't drive|stop driving",
}
MEASURE_CUES = {"potential_units": r"vehicle|unit|affected|car|truck", "injuries": r"injur|hurt", "deaths": r"death|fatal|died|kill"}
_CUE_STOP = {"other", "system", "control", "unknown", "taxonomy", "complaint", "built", "high", "voltage", "only", "vehicle",
             "back", "over", "prevention", "call"}
_STEMS = {f: sorted({t.rstrip("s") for t in re.findall(r"[a-z]+", f"{f} {g}".lower()) if len(t) >= 4 and t not in _CUE_STOP})
          for f, g in COMPONENT_FAMILIES.items()}
# A family is named in a question when a word starts with a stem from the family name or its gloss.
FAMILY_CUES = {f: re.compile(r"\b(" + "|".join(stems) + ")") for f, stems in _STEMS.items() if stems}


def families_mentioned(text: str) -> set[str]:
    """Component families whose cue words occur in ``text`` (lower case)."""
    return {f for f, cue in FAMILY_CUES.items() if cue.search(text)}


def ungrounded_arguments(call: StatsQuery | RecallSearch, question: str, prefix: str) -> list[str]:
    """Arguments the question gives no support for: years that do not appear in it, flags and measures it never mentions."""
    q = question.lower()
    numbers = set(re.findall(r"\d{4}", question))
    out = []
    for key in ("model_year", "year"):
        v = getattr(call, key, None)
        if v is not None and str(v) not in numbers:
            out.append(f"{prefix}.{key}: {v} does not appear in the question; set it to null")
    years = {n for n in numbers if 1980 <= int(n) <= 2030}
    used = {str(getattr(call, k)) for k in ("model_year", "year") if getattr(call, k, None) is not None}
    if len(years) == 1 and not years <= used:
        out.append(f"{prefix}: the question mentions {years.pop()} but no filter uses it; set model_year for a vehicle's "
                   "model year" + (", or year for when complaints were received or recalls reported" if isinstance(call, StatsQuery) else ""))
    if isinstance(call, StatsQuery):
        for flag, cue in FLAG_CUES.items():
            if getattr(call, flag) and not re.search(cue, q):
                out.append(f"{prefix}.{flag}: the question does not ask about this; set it to null")
        cue = MEASURE_CUES.get(call.measure)
        if cue and not re.search(cue, q):
            out.append(f"{prefix}.measure: the question does not ask for {call.measure}; use 'count' to count complaints or recall campaigns")
        text = q
        for name in (call.make, call.model):  # model names such as SIENNA HYBRID must not count as component words
            text = text.replace(name.lower(), " ") if name else text
        if call.component and call.group_by == "component":
            out.append(f"{prefix}.component: do not filter on a component when ranking by component; set it to null")
        elif call.component and call.component in FAMILY_CUES and not FAMILY_CUES[call.component].search(text):
            out.append(f"{prefix}.component: the question does not mention {call.component}; set it to null")
        elif call.component is None and call.group_by != "component" and families_mentioned(text):
            out.append(f"{prefix}.component: the question names a component ({', '.join(sorted(families_mentioned(text)))}) "
                       "but component is null")
    return out


def validation_messages(err: ValidationError, prefix: str) -> list[str]:
    """Short, LLM-readable messages from a Pydantic ValidationError."""
    out = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"])
        msg = e["msg"].removeprefix("Value error, ")
        out.append(f"{prefix}{('.' + loc) if loc else ''}: {msg}")
    return out
