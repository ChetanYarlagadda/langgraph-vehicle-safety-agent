"""Build the database and index, run the agent on the evaluation set, and write every result table and figure.

    python scripts/run_pipeline.py            # full run (50 questions + baselines)
    python scripts/run_pipeline.py --quick    # smoke test on 9 questions; outputs go to data/cache/smoke/

Every LLM reply is cached in data/cache/llm_cache.sqlite under sha256(model, options, messages, schema), and the
recall embeddings in data/cache/emb/, so a re-run only calls the model for prompts it has not seen.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydantic  # noqa: E402
import sentence_transformers  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version  # noqa: E402

from safetyagent.agent import SafetyAgent, classify_only, closed_book_answer  # noqa: E402
from safetyagent.crosscheck import crosscheck_vehicle  # noqa: E402
from safetyagent.data import COMPLAINTS_FILE, RECALLS_FILE, build_database, coverage, download, manifest  # noqa: E402
from safetyagent.evalset import LAY_NAMES, ROUTING_PROBES, EvalItem, build_eval_set, vehicles_of  # noqa: E402
from safetyagent.figures import make_all  # noqa: E402
from safetyagent.llm import CachedChat, OllamaChat  # noqa: E402
from safetyagent.router import keyword_route  # noqa: E402
from safetyagent.scoring import (answer_correct, args_correct, error_category, evidence_has_gold, failure_stage,  # noqa: E402
                                 mcnemar_exact, percentile, wilson)
from safetyagent.search import EMBED_MODEL, Encoder, RecallIndex, encode_cached, recall_text  # noqa: E402
from safetyagent.tools import SafetyDB  # noqa: E402

torch.set_num_threads(2)

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "reports" / "results"
FIGS = ROOT / "reports" / "figures"
LLM_MODEL = "qwen3:4b"
DEV_SEED = 0  # development questions: prompts and validation rules were tuned on these (smoke runs)
TEST_SEED = 1  # test questions: disjoint vehicles, run once at the end
N_CROSSCHECK = 20
ROUTES = ["sql", "semantic", "both"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock per stage."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def run(self, name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        return out


# --------------------------------------------------------------------------- data and index


def load_database() -> tuple[Path, dict]:
    rcl = download(RECALLS_FILE, RAW)
    cmpl = download(COMPLAINTS_FILE, RAW)
    man = manifest(RAW)
    key = "".join(v["sha256"][:6] for v in man.values())
    db_path = CACHE / f"nhtsa__{key}.duckdb"
    if not db_path.exists():
        counts = build_database(rcl, cmpl, db_path)
        log(f"database built: {counts}")
    return db_path, man


def build_index(db: SafetyDB, timer: Timer) -> tuple[RecallIndex, Encoder, Path]:
    rows = db.con.execute(
        "SELECT r.campaign, coalesce(string_agg(DISTINCT k.component, '; ' ORDER BY k.component), ''), r.defect, r.consequence "
        "FROM recalls r LEFT JOIN recall_components k ON k.campaign = r.campaign GROUP BY r.campaign, r.defect, r.consequence ORDER BY r.campaign"
    ).fetchall()
    ids = [r[0] for r in rows]
    texts = [recall_text(r[1], r[2], r[3]) for r in rows]
    enc = Encoder(EMBED_MODEL)
    emb = timer.run("embed_recalls_s", encode_cached, enc, ids, texts, CACHE / "emb")
    path = sorted((CACHE / "emb").glob("emb__*.npy"), key=lambda p: p.stat().st_mtime)[-1]
    log(f"index: {len(ids):,} recall campaigns, dim {emb.shape[1]}")
    return RecallIndex(ids, emb, enc.encode_query), enc, path


# --------------------------------------------------------------------------- experiments


def run_crosscheck(db: SafetyDB, items: list[EvalItem], cov: dict, quick: bool) -> pd.DataFrame:
    vehicles = []
    for it in items:
        f = it.gold_filters
        if f.get("model") and (f["make"], f["model"], f["model_year"]) not in vehicles:
            vehicles.append((f["make"], f["model"], f["model_year"]))
    vehicles = vehicles[: (3 if quick else N_CROSSCHECK)]
    lo, hi = pd.Timestamp(cov["complaints_from"]).date(), pd.Timestamp(cov["complaints_to"]).date()
    rows = [crosscheck_vehicle(db.con, mk, md, yr, lo, hi, CACHE / "api") for mk, md, yr in vehicles]
    return pd.DataFrame(rows)


def run_retrieval(db: SafetyDB, items: list[EvalItem], enc: Encoder, index: RecallIndex) -> pd.DataFrame:
    """Recall-search strategies on the questions whose gold answer is a campaign (no LLM involved)."""
    rows = []
    for it in items:
        if not it.gold_campaigns:
            continue
        f = it.gold_filters
        veh = db.vehicle_campaigns(f["make"], f["model"], f["model_year"])
        gold = set(it.gold_campaigns)
        dates = dict(db.con.execute("SELECT campaign, report_date FROM recalls WHERE campaign IN (SELECT unnest(?))", [veh]).fetchall())
        newest = sorted(veh, key=lambda c: (dates[c], c), reverse=True)
        q_question = enc.encode_query(it.question).reshape(1, -1)
        q_component = enc.encode_query(LAY_NAMES[f["component"]]).reshape(1, -1)
        rankings = {
            "vehicle filter + semantic (question)": [c for c, _ in index.search_vector(q_question, len(veh), veh)],
            "vehicle filter + semantic (component words)": [c for c, _ in index.search_vector(q_component, len(veh), veh)],
            "vehicle filter + newest first": newest,
            "no filter + semantic (question)": [c for c, _ in index.search_vector(q_question, 50, None)],
        }
        for name, ranked in rankings.items():
            first = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
            rows.append({"qid": it.qid, "strategy": name, "n_vehicle_campaigns": len(veh), "n_gold": len(gold),
                         "first_gold_rank": first, **{f"hit@{k}": bool(first and first <= k) for k in (1, 3, 5)}})
        n, g = len(veh), len(gold & set(veh))
        rows.append({"qid": it.qid, "strategy": "vehicle filter + random order (expected)", "n_vehicle_campaigns": n, "n_gold": len(gold),
                     "first_gold_rank": None, **{f"hit@{k}": 1 - comb(n - g, min(k, n)) / comb(n, min(k, n)) for k in (1, 3, 5)}})
    return pd.DataFrame(rows)


def run_agent(agent: SafetyAgent, items: list[EvalItem]) -> tuple[pd.DataFrame, list[dict]]:
    rows, traces = [], []
    for i, it in enumerate(items, start=1):
        res = agent.ask(it.question)
        first = res.answers[0] if res.answers else {"answer": "", "citations": [], "problems": ["no answer"]}
        route_ok = res.route == it.route
        evidence_ok = evidence_has_gold(it, res.evidence)
        ok_final = answer_correct(it, res.answer, res.citations)
        ok_first = answer_correct(it, first["answer"], first["citations"])
        node_s: dict[str, float] = {}
        for e in res.events:
            node_s[e["node"]] = node_s.get(e["node"], 0.0) + e.get("llm_s", 0.0) + e.get("tool_s", 0.0)
        rows.append({
            "qid": it.qid, "template": it.template, "route_gold": it.route, "route_pred": res.route, "route_source": res.route_source,
            "route_ok": route_ok, "plan_attempts": res.plan_attempts, "plan_valid_first": not res.call_log[0]["errors"],
            "plan_valid_final": not res.call_log[-1]["errors"], "n_calls_first": res.call_log[0]["n_calls"],
            "args_ok": args_correct(it, res.stats_calls, res.search_calls), "evidence_ok": evidence_ok,
            "answer_attempts": res.answer_attempts, "declined": res.declined, "flagged_first": bool(first["problems"]),
            "grounded_final": res.grounded,
            "correct_first": ok_first, "correct_final": ok_final, "stage": failure_stage(route_ok, evidence_ok, ok_final),
            "latency_s": res.latency_s, "llm_s": res.llm_seconds, "tool_s": res.tool_seconds, "llm_calls": res.llm_calls,
            **{f"{k}_s": v for k, v in node_s.items()},
            "answer": res.answer, "citations": ";".join(res.citations), "problems_final": " | ".join(res.problems),
        })
        traces.append({"qid": it.qid, **res.to_dict()})
        log(f"agent {i}/{len(items)} {it.qid}: route {res.route} ({'ok' if route_ok else 'WRONG'}), "
            f"plans {res.plan_attempts}, answers {res.answer_attempts}, correct {ok_final}, {res.latency_s:.1f}s")
    return pd.DataFrame(rows), traces


def run_closed_book(llm, items: list[EvalItem], cov: dict) -> pd.DataFrame:
    rows = []
    for it in items:
        ans, reply = closed_book_answer(llm, it.question, cov)
        rows.append({"qid": it.qid, "template": it.template, "route_gold": it.route, "answer": ans,
                     "correct": answer_correct(it, ans, []), "latency_s": reply.latency_s})
    return pd.DataFrame(rows)


def run_routing(llm, items: list[EvalItem], agent_rows: pd.DataFrame, quick: bool) -> pd.DataFrame:
    pred = dict(zip(agent_rows.qid, agent_rows.route_pred))
    rows = [{"set": "evaluation questions (templated)", "question": it.question, "route_gold": it.route,
             "route_llm": pred[it.qid], "route_keyword": keyword_route(it.question)} for it in items]
    for q, gold in ROUTING_PROBES[:: (6 if quick else 1)]:
        route, _ = classify_only(llm, q)
        rows.append({"set": "separately written paraphrases", "question": q, "route_gold": gold, "route_llm": route or "invalid",
                     "route_keyword": keyword_route(q)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summaries


def summarize(agent_rows: pd.DataFrame, closed: pd.DataFrame, routing: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cb = dict(zip(closed.qid, closed.correct))
    agent_rows = agent_rows.assign(closed_book_correct=agent_rows.qid.map(cb))
    out = []
    for route in ROUTES + ["all"]:
        g = agent_rows if route == "all" else agent_rows[agent_rows.route_gold == route]
        n = len(g)
        if n == 0:
            continue
        k_final, k_cb = int(g.correct_final.sum()), int(g.closed_book_correct.sum())
        lo, hi = wilson(k_final, n)
        cb_lo, cb_hi = wilson(k_cb, n)
        out.append({
            "route": route, "n": n, "route_accuracy": g.route_ok.mean(),
            "plan_valid_first_try": g.plan_valid_first.mean(), "plan_valid_after_retries": g.plan_valid_final.mean(),
            "tool_args_exact": g.args_ok.mean(), "evidence_contains_gold": g.evidence_ok.mean(),
            "answer_accuracy_first_draft": g.correct_first.mean(), "answer_accuracy": k_final / n, "answer_ci_low": lo, "answer_ci_high": hi,
            "closed_book_accuracy": k_cb / n, "closed_book_ci_low": cb_lo, "closed_book_ci_high": cb_hi,
            "agent_declined": g.declined.mean(),
            "agent_answered_wrong": (~g.correct_final & ~g.declined).mean(),
            "grounded_final": g.grounded_final.mean(), "latency_p50_s": percentile(g.latency_s.tolist(), 50),
            "latency_p95_s": percentile(g.latency_s.tolist(), 95), "llm_calls_mean": g.llm_calls.mean(),
        })
    summary = pd.DataFrame(out)

    def mcn(a: pd.Series, b: pd.Series, name: str) -> dict:
        b_only, a_only = int((~a & b).sum()), int((a & ~b).sum())
        return {"comparison": name, "n": len(a), "accuracy_a": a.mean(), "accuracy_b": b.mean(), "only_a_correct": a_only,
                "only_b_correct": b_only, "p_value_exact_mcnemar": mcnemar_exact(a_only, b_only)}

    comps = pd.DataFrame([
        mcn(agent_rows.closed_book_correct.astype(bool), agent_rows.correct_final.astype(bool), "closed-book LLM (a) vs agent (b)"),
        mcn(agent_rows.correct_first.astype(bool), agent_rows.correct_final.astype(bool), "agent first draft (a) vs after grounding retries (b)"),
    ])

    first = np.where(agent_rows.declined, "declined (no draft)", np.where(agent_rows.correct_first, "first draft correct", "first draft wrong"))
    grounding = (agent_rows.assign(first=first, check=np.where(agent_rows.declined, "not checked",
                                                              np.where(agent_rows.flagged_first, "flagged by grounding check", "passed grounding check")))
                 .groupby(["first", "check"]).agg(n=("qid", "size"), correct_after_retries=("correct_final", "sum")).reset_index())

    stages = agent_rows.groupby(["route_gold", "stage"]).size().rename("n").reset_index()

    route_rows = []
    for name, g in routing.groupby("set"):
        for router in ("llm", "keyword"):
            route_rows.append({"set": name, "router": router, "n": len(g), "accuracy": (g[f"route_{router}"] == g.route_gold).mean()})
    route_acc = pd.DataFrame(route_rows)

    node_cols = [c for c in ("classify_s", "plan_s", "execute_s", "answer_s") if c in agent_rows]
    latency = agent_rows.groupby("route_gold")[node_cols + ["latency_s", "llm_calls"]].mean().reset_index()
    latency["latency_p50_s"] = agent_rows.groupby("route_gold").latency_s.median().values
    latency["latency_p95_s"] = [percentile(g.tolist(), 95) for _, g in agent_rows.groupby("route_gold").latency_s]
    return {"summary": summary, "comparisons": comps, "grounding": grounding, "stages": stages, "routing_accuracy": route_acc,
            "latency": latency, "agent_rows": agent_rows}


def plan_errors(traces: list[dict]) -> pd.DataFrame:
    """Every validator message, its category, and whether the question's final plan was valid."""
    rows = []
    for t in traces:
        resolved = not t["call_log"][-1]["errors"]
        for c in t["call_log"]:
            for e in c["errors"]:
                rows.append({"qid": t["qid"], "attempt": c["attempt"], "category": error_category(e), "resolved_by_retry": resolved, "error": e})
    return pd.DataFrame(rows, columns=["qid", "attempt", "category", "resolved_by_retry", "error"])


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smoke test on 9 development questions; outputs to data/cache/smoke/")
    parser.add_argument("--skip-api-check", action="store_true", help="don't call the live NHTSA API")
    parser.add_argument("--api-check-only", action="store_true", help="only (re)run the live NHTSA API cross-check")
    args = parser.parse_args()

    global RESULTS, FIGS
    if args.quick:
        RESULTS = FIGS = CACHE / "smoke"
    for d in (CACHE, RESULTS, FIGS):
        d.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    t_all = time.perf_counter()

    db_path, man = timer.run("database_s", load_database)
    db = SafetyDB(db_path)
    cov = coverage(db.con)
    log(f"data: {cov}")

    dev = build_eval_set(db.con, seed=DEV_SEED)
    if args.quick:
        firsts: dict[str, EvalItem] = {}
        for it in dev:
            firsts.setdefault(it.template, it)
        items = list(firsts.values())
    else:
        items = build_eval_set(db.con, seed=TEST_SEED, exclude_vehicles=vehicles_of(dev))
    log(f"evaluation set: {len(items)} questions " + str(pd.Series([i.route for i in items]).value_counts().to_dict()))

    cross = None
    if not args.skip_api_check:
        try:
            cross = timer.run("api_crosscheck_s", run_crosscheck, db, items, cov, args.quick)
            r_ok, c_ok = cross[cross.api_status_recalls == 200], cross[cross.api_status_complaints == 200]
            same_recalls = int(((r_ok.recalls_db == r_ok.recalls_both) & (r_ok.recalls_api == r_ok.recalls_both)).sum())
            same_counts = int((c_ok.complaints_db == c_ok.complaints_api_same_window).sum())
            log(f"API cross-check: recall sets identical for {same_recalls}/{len(r_ok)} vehicles, complaint counts equal for "
                f"{same_counts}/{len(c_ok)} (of {len(cross)} vehicles queried)")
        except Exception as exc:  # the API being down should not sink the run
            log(f"API cross-check skipped: {exc}")
    if args.api_check_only:
        if cross is not None:
            cross.to_csv(RESULTS / "api_crosscheck.csv", index=False)
        return

    index, enc, emb_path = build_index(db, timer)
    retrieval = timer.run("retrieval_eval_s", run_retrieval, db, items, enc, index)

    llm = CachedChat(OllamaChat(LLM_MODEL), CACHE / "llm_cache.sqlite")
    agent = SafetyAgent(llm, db, index, cov)
    agent_rows, traces = timer.run("agent_eval_s", run_agent, agent, items)
    closed = timer.run("closed_book_s", run_closed_book, llm, items, cov)
    routing = timer.run("routing_probes_s", run_routing, llm, items, agent_rows, args.quick)
    s = summarize(agent_rows, closed, routing)
    log("summary:\n" + s["summary"][["route", "n", "route_accuracy", "tool_args_exact", "evidence_contains_gold", "answer_accuracy",
                                       "closed_book_accuracy", "latency_p50_s"]].round(3).to_string(index=False))

    # ---- write tables
    w = lambda frame, name: frame.to_csv(RESULTS / name, index=False, float_format="%.4f")  # noqa: E731
    w(pd.DataFrame([i.to_row() for i in items]), "eval_set.csv")
    w(s["agent_rows"], "agent_results.csv")
    w(s["summary"], "summary_by_route.csv")
    w(s["comparisons"], "comparisons.csv")
    w(s["grounding"], "grounding_check.csv")
    w(s["stages"], "failure_stages.csv")
    w(s["routing_accuracy"], "routing_accuracy.csv")
    w(routing, "routing.csv")
    w(s["latency"], "latency_by_route.csv")
    w(plan_errors(traces), "plan_validation_errors.csv")
    w(closed, "closed_book.csv")
    w(retrieval, "retrieval_per_question.csv")
    rsum = retrieval.groupby("strategy")[["hit@1", "hit@3", "hit@5"]].mean().reset_index()
    rsum["n"] = retrieval.groupby("strategy").size().values
    w(rsum, "retrieval_summary.csv")
    if cross is not None:
        w(cross, "api_crosscheck.csv")
    with open(RESULTS / "agent_traces.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, default=str) + "\n")

    # ---- artifacts for the API
    api_dir = CACHE / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "config.json").write_text(json.dumps({"db": db_path.relative_to(ROOT).as_posix(), "embeddings": emb_path.relative_to(ROOT).as_posix(),
                                                     "embed_model": EMBED_MODEL, "llm_cache": "data/cache/llm_cache.sqlite"}, indent=2))

    timer.stages["total_s"] = time.perf_counter() - t_all
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": args.quick,
        "data": {"files": {k: {kk: v[kk] for kk in ("url", "bytes", "sha256", "inner_timestamp")} for k, v in man.items()},
                 "coverage": cov, "tables": {t: int(db.con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
                                             for t in ("recalls", "recall_vehicles", "recall_components", "complaints", "complaint_components")}},
        "eval": {"n_questions": len(items), "by_route": pd.Series([i.route for i in items]).value_counts().to_dict(),
                 "split": "development (smoke)" if args.quick else "test", "dev_seed": DEV_SEED, "test_seed": TEST_SEED,
                 "dev_questions": len(dev), "test_vehicles_disjoint_from_dev": True,
                 "n_routing_probes": int((routing.set == "separately written paraphrases").sum())},
        "agent": {"llm": llm.name, "options": llm.inner.options, "max_plan_attempts": agent.max_plan_attempts,
                  "max_answer_attempts": agent.max_answer_attempts, "embed_model": EMBED_MODEL},
        "llm_usage": {"calls_this_run": llm.hits + llm.misses, "cache_hits_this_run": llm.hits, "new_calls_this_run": llm.misses,
                      "new_model_seconds_this_run": round(llm.new_seconds, 1), "model_seconds_all_cached_replies": round(llm.total_model_seconds(), 1)},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cpu": platform.processor(),
                     "torch_threads": torch.get_num_threads()},
        "versions": {"python": platform.python_version(), "langgraph": version("langgraph"), "duckdb": duckdb.__version__,
                     "pydantic": pydantic.__version__, "faiss": faiss.__version__, "sentence_transformers": sentence_transformers.__version__,
                     "torch": torch.__version__},
        "timings_s": {k: round(v, 1) for k, v in timer.stages.items()},
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    make_all(RESULTS, FIGS)
    log(f"done in {timer.stages['total_s'] / 60:.1f} min; LLM: {llm.misses} new calls ({llm.new_seconds / 60:.1f} model-min), {llm.hits} cache hits")


if __name__ == "__main__":
    main()
