"""Cross-check of the flat-file database against the live NHTSA API (api.nhtsa.gov, no key)."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime
from pathlib import Path

import duckdb
import requests

API = "https://api.nhtsa.gov"
HEADERS = {"User-Agent": "langgraph-vehicle-safety-agent data cross-check (research)"}


def fetch_json(url: str, params: dict, cache_dir: Path, pause_s: float = 0.5) -> tuple[int, dict]:
    """GET with an on-disk cache keyed by URL and parameters; returns (HTTP status, JSON body)."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return (saved["status"], saved["body"]) if set(saved) == {"status", "body"} else (200, saved)
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    try:
        body = r.json()
    except ValueError:
        r.raise_for_status()
        raise
    if r.status_code >= 500:
        r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": r.status_code, "body": body}), encoding="utf-8")
    time.sleep(pause_s)
    return r.status_code, body


def crosscheck_vehicle(con: duckdb.DuckDBPyConnection, make: str, model: str, model_year: int, complaints_from: date,
                       complaints_to: date, cache_dir: Path) -> dict:
    """Recall campaigns and complaint counts for one vehicle: flat-file database vs live API.

    The API sometimes answers HTTP 400 with an empty result for vehicles that do have data; such responses are
    recorded with their status and left out of the comparison.
    """
    p = {"make": make, "model": model, "modelYear": model_year}
    rs, rec = fetch_json(f"{API}/recalls/recallsByVehicle", p, cache_dir)
    db_camp = {r[0] for r in con.execute(
        "SELECT DISTINCT campaign FROM recall_vehicles WHERE make = ? AND model = ? AND model_year = ?", [make, model, model_year]).fetchall()}
    cs, cmp = fetch_json(f"{API}/complaints/complaintsByVehicle", p, cache_dir)
    db_n = con.execute("SELECT count(DISTINCT odino) FROM complaints WHERE make = ? AND model = ? AND model_year = ?",
                       [make, model, model_year]).fetchone()[0]
    row = {"make": make, "model": model, "model_year": model_year, "api_status_recalls": rs, "api_status_complaints": cs,
           "recalls_db": len(db_camp), "complaints_db": int(db_n)}
    if rs == 200:
        api_camp = {r["NHTSACampaignNumber"] for r in rec.get("results", [])}
        row.update(recalls_api=len(api_camp), recalls_both=len(db_camp & api_camp),
                   recalls_only_db=";".join(sorted(db_camp - api_camp)), recalls_only_api=";".join(sorted(api_camp - db_camp)))
    if cs == 200:
        filed = [datetime.strptime(r["dateComplaintFiled"], "%m/%d/%Y").date() for r in cmp.get("results", []) if r.get("dateComplaintFiled")]
        row.update(complaints_api_same_window=sum(complaints_from <= d <= complaints_to for d in filed), complaints_api_all_years=len(filed))
    return row
