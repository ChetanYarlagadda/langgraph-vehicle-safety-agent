"""Scoring of agent answers against the SQL gold labels, plus small statistics helpers."""

from __future__ import annotations

import math
import re
from typing import Any

from safetyagent.evalset import EvalItem
from safetyagent.grounding import CAMPAIGN_RE, numbers_in

FILTER_KEYS = ("dataset", "measure", "make", "model", "model_year", "component", "year", "crash", "fire", "do_not_drive", "group_by")


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def campaigns_cited(answer: str, citations: list[str]) -> set[str]:
    return set(CAMPAIGN_RE.findall(answer)) | {c for c in (x.strip().strip("[]").upper() for x in citations) if CAMPAIGN_RE.fullmatch(c)}


def value_correct(item: EvalItem, answer: str) -> bool:
    """The gold number (or the gold component family) is stated in the answer."""
    if isinstance(item.gold_value, str):
        return _alnum(item.gold_value) in _alnum(answer)
    return str(item.gold_value) in numbers_in(re.sub(r"\[[^\]]*\]", " ", answer))


def answer_correct(item: EvalItem, answer: str, citations: list[str]) -> bool:
    """SQL route: gold value stated. Semantic: a gold campaign cited. Both: both conditions."""
    ok_value = value_correct(item, answer) if item.gold_value is not None else True
    ok_campaign = bool(campaigns_cited(answer, citations) & set(item.gold_campaigns)) if item.gold_campaigns else True
    return ok_value and ok_campaign


def _active(call: dict) -> dict:
    out = {}
    for k in FILTER_KEYS:
        v = call.get(k)
        if v is None or v is False:
            continue
        out[k] = v
    return out


def args_correct(item: EvalItem, stats_calls: list[dict], search_calls: list[dict]) -> bool:
    """A tool call with exactly the gold filters exists (stats), and/or a search restricted to the gold vehicle."""
    ok = True
    if item.route in ("sql", "both"):
        gold = {k: v for k, v in item.gold_filters.items() if k in FILTER_KEYS}
        ok &= any(_active(c) == gold for c in stats_calls)
    if item.route in ("semantic", "both"):
        veh = {k: item.gold_filters[k] for k in ("make", "model", "model_year")}
        ok &= any(all(c.get(k) == v for k, v in veh.items()) for c in search_calls)
    return ok


def evidence_has_gold(item: EvalItem, evidence: list[dict]) -> bool:
    """Whether the tool output handed to the answer node contained the gold answer."""
    ok = True
    if item.gold_value is not None:
        if isinstance(item.gold_value, str):
            ok &= any(e["kind"] == "stats" and e["must_mention"][:1] == [item.gold_value] for e in evidence)
        else:
            ok &= any(e["kind"] == "stats" and str(item.gold_value) in numbers_in(e["text"].split("result", 1)[-1]) for e in evidence)
    if item.gold_campaigns:
        ok &= any(e["kind"] == "recall" and e["id"] in item.gold_campaigns for e in evidence)
    return ok


def failure_stage(route_ok: bool, evidence_ok: bool, answer_ok: bool) -> str:
    """Where a question was lost: route, tool call (evidence lacks the gold answer) or answer synthesis."""
    if answer_ok:
        return "correct"
    if not route_ok and not evidence_ok:
        return "wrong route"
    if not evidence_ok:
        return "tool call"
    return "answer synthesis"


ERROR_CATEGORIES = [
    ("NHTSA has no", "model name does not exist for that model year"),
    ("is not in the NHTSA data", "unknown make or model"),
    ("is not a component family", "unknown component"),
    ("use model_year for the vehicle's model year", "model year used as calendar year"),
    ("does not appear in the question", "year not in the question"),
    ("but no filter uses it", "year in the question not used"),
    ("the question does not ask about this", "crash/fire flag not asked for"),
    ("the question does not ask for", "measure not asked for"),
    ("the question does not mention", "component not in the question"),
    ("names a component", "component in the question not used"),
    ("when ranking by component", "component filter while ranking components"),
    ("needs at least one", "tool missing for the route"),
    ("not a JSON object", "invalid JSON"),
]


def error_category(message: str) -> str:
    """Coarse category of a validator message (for the validation tables and figure)."""
    return next((cat for key, cat in ERROR_CATEGORIES if key in message), "other schema error")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(rows: list[dict[str, Any]], flag: str) -> dict[str, Any]:
    """Share of rows with a true flag, with a Wilson interval."""
    k = sum(bool(r[flag]) for r in rows)
    lo, hi = wilson(k, len(rows))
    return {"k": k, "n": len(rows), "rate": k / len(rows) if rows else math.nan, "ci_low": lo, "ci_high": hi}
