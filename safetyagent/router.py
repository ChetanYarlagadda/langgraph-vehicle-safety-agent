"""Keyword router: the no-LLM baseline for the classify node (also its fallback when the LLM output is invalid)."""

from __future__ import annotations

import re

SQL_CUES = (r"\bhow many\b", r"\bnumber of\b", r"\bcount\b", r"\btotal\b", r"\bmost\b", r"\bhow much\b", r"\brank",
            r"\btop \d", r"\bwhich (component|model|make|year)")
SEMANTIC_CUES = (r"\bwhy\b", r"\bwhat (is|was) the (defect|problem|risk|remedy|fix)", r"\bdescribe\b", r"\bremedy\b",
                 r"\bis there a recall\b", r"\bhas there been a recall\b", r"\brecalled (over|for|because)\b",
                 r"\bwhat (problem|defect)\b", r"\bexplain\b")


def keyword_route(question: str) -> str:
    """'sql', 'semantic' or 'both' from surface cues; defaults to 'semantic' when nothing matches."""
    q = question.lower()
    sql = any(re.search(p, q) for p in SQL_CUES)
    sem = any(re.search(p, q) for p in SEMANTIC_CUES)
    if sql and sem:
        return "both"
    if sql:
        return "sql"
    return "semantic"
