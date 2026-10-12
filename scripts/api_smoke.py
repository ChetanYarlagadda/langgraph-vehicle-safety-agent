"""In-process check of the FastAPI service with the real engine (DuckDB, FAISS index, cached LLM replies).

    python scripts/api_smoke.py

Asks three questions from the test set (one per route), so the LLM replies come from the pipeline's cache, and
calls the two tool endpoints directly. Writes reports/results/api_smoke.json.
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

if __name__ == "__main__":
    ev = pd.read_csv(ROOT / "reports" / "results" / "eval_set.csv")
    questions = [ev[ev.route == r].question.iloc[0] for r in ("sql", "semantic", "both")]
    out: dict = {"ask": [], "tools": {}}
    with TestClient(app) as c:
        for q in questions:
            t0 = time.perf_counter()
            r = c.post("/ask", json={"question": q})
            body = r.json()
            out["ask"].append({"question": q, "status": r.status_code, "route": body.get("route"), "grounded": body.get("grounded"),
                               "citations": body.get("citations"), "answer": body.get("answer"),
                               "wall_ms_with_cached_llm": round((time.perf_counter() - t0) * 1000, 1)})
        stats_body = {"dataset": "complaints", "make": "toyota", "model": "camry", "model_year": 2018, "group_by": "component", "limit": 3}
        times, searches = [], []
        for _ in range(20):
            t0 = time.perf_counter()
            r = c.post("/tools/stats", json=stats_body)
            times.append((time.perf_counter() - t0) * 1000)
        out["tools"]["stats"] = {"request": stats_body, "status": r.status_code, "rows": r.json()["rows"],
                                 "median_ms": round(statistics.median(times), 1)}
        search_body = {"query": "rearview camera image does not display", "make": "FORD", "model": "F-150", "model_year": 2021, "k": 3}
        for _ in range(20):
            t0 = time.perf_counter()
            r = c.post("/tools/search", json=search_body)
            searches.append((time.perf_counter() - t0) * 1000)
        out["tools"]["search"] = {"request": search_body, "status": r.status_code, "campaigns": [h["campaign"] for h in r.json()],
                                  "median_ms": round(statistics.median(searches), 1)}
        bad = c.post("/tools/stats", json={"dataset": "complaints", "make": "FORDD"})
        out["tools"]["invalid_request"] = {"status": bad.status_code, "detail": bad.json()["detail"]}
    (ROOT / "reports" / "results" / "api_smoke.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:3000])
