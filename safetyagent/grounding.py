"""Deterministic grounding checks for a drafted answer: citations, numbers and use of the SQL results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE_RE = re.compile(r"\[([^\[\]]{1,80})\]")
CAMPAIGN_RE = re.compile(r"\b\d{2}[VETC]\d{6}\b")
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


@dataclass
class Evidence:
    """One piece of tool output the answer may cite."""

    id: str
    kind: str  # "stats" | "recall" | "empty"
    text: str
    must_mention: list[str] = field(default_factory=list)  # for stats: values the answer has to state (any of them)


def normalize_number(token: str) -> str:
    """'1,234' -> '1234'; '12.50' -> '12.5'; '3.0' -> '3'."""
    t = token.replace(",", "").rstrip(".")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t.lstrip("0") or "0"


def numbers_in(text: str) -> set[str]:
    """Normalised numbers in free text, ignoring recall campaign ids."""
    return {normalize_number(m) for m in NUM_RE.findall(CAMPAIGN_RE.sub(" ", text))}


def normalize_id(token: str) -> str:
    return token.strip().strip("[]").strip().upper()


def cited_ids(answer: str, citations: list[str]) -> set[str]:
    """Ids cited in the citation list and inline in the answer text (``[S1]``, ``[S1, 24V123000]``)."""
    inline = {normalize_id(part) for group in CITE_RE.findall(answer) for part in re.split(r"[,;]", group)}
    return {normalize_id(c) for c in citations if c and c.strip()} | {i for i in inline if i}


def check_grounding(answer: str, citations: list[str], evidence: list[Evidence], question: str) -> list[str]:
    """Problems with a drafted answer (empty list = grounded).

    1. The answer is non-empty and cites at least one evidence id when evidence exists.
    2. Every cited id exists in the evidence.
    3. Every number in the answer occurs in the evidence or in the question.
    4. Each statistics result is actually stated (its value, or one of its group names for a ranking).
    """
    problems: list[str] = []
    if not answer.strip():
        return ["the answer is empty"]
    ids = {e.id.upper() for e in evidence}
    cited = cited_ids(answer, citations)
    if evidence and not cited:
        problems.append("the answer cites no evidence id; cite ids such as " + ", ".join(sorted(ids)[:3]))
    unknown = sorted(cited - ids)
    if unknown:
        problems.append(f"cited ids not in the evidence: {', '.join(unknown)}")
    allowed = numbers_in(question)
    for e in evidence:
        allowed |= numbers_in(e.text) | numbers_in(e.id)
    stray = sorted(numbers_in(CITE_RE.sub(" ", answer)) - allowed, key=lambda s: (len(s), s))
    if stray:
        problems.append(f"numbers not found in the evidence: {', '.join(stray)}")
    low = answer.lower()
    answer_numbers = numbers_in(CITE_RE.sub(" ", answer))
    for e in evidence:
        if e.kind != "stats" or not e.must_mention:
            continue
        ok = any((normalize_number(v) in answer_numbers) if re.fullmatch(r"[\d,.]+", v) else (v.lower() in low) for v in e.must_mention)
        if e.must_mention == ["0"] and re.search(r"\b(no|zero|none)\b", low):
            ok = True
        if not ok:
            problems.append(f"the answer does not state the result of [{e.id}] ({e.must_mention[0]})")
    return problems
