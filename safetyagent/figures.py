"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from safetyagent import plotting
from safetyagent.plotting import AXIS, INK, INK_2, MUTED, SERIES, STAGE_COLOR

# Takeaway titles, written after reading the results.
TITLES = {
    "scorecard": "With SQL and recall-search tools qwen3:4b answers 43 of 50 questions correctly; without tools, none",
    "stages": "6 of the 7 misses start in the tool call (vehicle name or filter); 1 picked the wrong recall from good evidence",
    "retries": "Validator retries lift valid tool calls from 78% to 94%; grounding rewrites changed no answer's correctness",
    "latency": "A question takes 4.9 s at the median (7.4 s at p95) on a laptop GPU; planning takes the largest share",
    "retrieval": "Restricting recall search to the vehicle's campaigns lifts hit@1 from 0.46 to 0.77; newest-first is no better than random",
    "routing": "Both routers are perfect on templated questions; on paraphrases the LLM gets 17/18, keywords 11/18",
    "validation": "Retries fix filter mistakes; dropped model suffixes ('SILVERADO' for 'Silverado 1500') never got fixed",
}

ROUTE_ORDER = ["sql", "semantic", "both", "all"]


def _route_label(route: str, n: int) -> str:
    return {"sql": "SQL", "semantic": "Semantic", "both": "Both", "all": "All"}[route] + f"\n(n = {n})"


def _suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def _bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}", size: float = 7.5) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, fmt.format(h), ha="center", va="bottom", fontsize=size, color=INK_2)


def fig_scorecard(summary: pd.DataFrame, path: Path) -> Path:
    """Per route: route accuracy, exact tool arguments, evidence containing the answer, answer accuracy vs closed-book."""
    s = summary.set_index("route").loc[[r for r in ROUTE_ORDER if r in set(summary.route)]]
    metrics = [
        ("route_accuracy", "route chosen correctly", SERIES[6]),
        ("tool_args_exact", "tool arguments exactly right", SERIES[0]),
        ("evidence_contains_gold", "tool output contains the answer", SERIES[2]),
        ("answer_accuracy", "final answer correct (agent)", SERIES[1]),
        ("closed_book_accuracy", "answer correct, same LLM without tools", AXIS),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(s))
    width = 0.16
    for j, (col, label, color) in enumerate(metrics):
        bars = ax.bar(x + (j - 2) * width, s[col], width, color=color, label=label)
        _bar_labels(ax, bars, size=6.8)
        if col in ("answer_accuracy", "closed_book_accuracy"):
            lo = s["answer_ci_low" if col == "answer_accuracy" else "closed_book_ci_low"]
            hi = s["answer_ci_high" if col == "answer_accuracy" else "closed_book_ci_high"]
            ax.errorbar(x + (j - 2) * width, s[col], yerr=[s[col] - lo, hi - s[col]], fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=2)
    ax.set_xticks(x, [_route_label(r, int(n)) for r, n in zip(s.index, s.n)])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of questions")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.2)
    _suptitle(fig, TITLES["scorecard"])
    ax.set_title("Gold route of each question; whiskers are 95% Wilson intervals", loc="left", fontsize=9, color=MUTED, fontweight="normal")
    plotting.save(fig, path)
    return path


def fig_stages(stages: pd.DataFrame, path: Path) -> Path:
    """Stacked outcome of each question per route: correct, or the first stage that lost it."""
    order = ["correct", "answer synthesis", "tool call", "wrong route"]
    piv = stages.pivot(index="route_gold", columns="stage", values="n").reindex(columns=order).fillna(0)
    piv = piv.loc[[r for r in ("sql", "semantic", "both") if r in piv.index]]
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    left = np.zeros(len(piv))
    y = np.arange(len(piv))[::-1]
    for st in order:
        vals = piv[st].to_numpy()
        ax.barh(y, vals, left=left, color=STAGE_COLOR[st], label=st, height=0.6)
        for yi, l, v in zip(y, left, vals):
            if v >= 1:
                ax.text(l + v / 2, yi, f"{int(v)}", ha="center", va="center", fontsize=8.5, color="white" if st != "answer synthesis" else INK)
        left += vals
    ax.set_yticks(y, [{"sql": "SQL", "semantic": "Semantic", "both": "Both"}[r] for r in piv.index])
    ax.set_xlabel("Questions")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=8.2,
              labels=["answer correct", "lost in answer synthesis (evidence had the answer)", "lost in the tool call", "wrong route"])
    _suptitle(fig, TITLES["stages"])
    plotting.save(fig, path)
    return path


def fig_retries(summary: pd.DataFrame, path: Path) -> Path:
    """Tool-call validity before/after planner retries, answer accuracy before/after grounding retries."""
    s = summary.set_index("route").loc[[r for r in ROUTE_ORDER if r in set(summary.route)]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    x = np.arange(len(s))
    panels = [
        (axes[0], [("plan_valid_first_try", "first plan", SERIES[0]), ("plan_valid_after_retries", "after validation retries", SERIES[2])],
         "Tool calls that pass Pydantic validation"),
        (axes[1], [("answer_accuracy_first_draft", "first draft", SERIES[0]), ("answer_accuracy", "after grounding retries", SERIES[2])],
         "Correct answers"),
    ]
    for ax, cols, title in panels:
        for j, (col, label, color) in enumerate(cols):
            bars = ax.bar(x + (j - 0.5) * 0.36, s[col], 0.36, color=color, label=label)
            _bar_labels(ax, bars)
        ax.set_xticks(x, [_route_label(r, int(n)) for r, n in zip(s.index, s.n)])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.2)
    axes[0].set_ylabel("Share of questions")
    _suptitle(fig, TITLES["retries"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plotting.save(fig, path)
    return path


def fig_latency(rows: pd.DataFrame, path: Path) -> Path:
    """Per-question latency by route (left) and mean seconds per graph node (right)."""
    routes = [r for r in ("sql", "semantic", "both") if r in set(rows.route_gold)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    rng = np.random.default_rng(0)
    for i, r in enumerate(routes):
        v = rows.loc[rows.route_gold == r, "latency_s"].to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=16, color=SERIES[0], alpha=0.75, edgecolor="none")
        med = float(np.median(v))
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.6)
        ax.text(i + 0.31, med, f"median {med:.1f} s", va="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(routes)), [{"sql": "SQL", "semantic": "Semantic", "both": "Both"}[r] for r in routes])
    ax.set_xlim(-0.5, len(routes) - 0.2)
    ax.set_ylim(0, None)
    ax.set_ylabel("Seconds per question (LLM + tools)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Each dot is one question", loc="left", fontsize=10)
    nodes = [("classify_s", "classify", SERIES[6]), ("plan_s", "plan (incl. retries)", SERIES[0]), ("execute_s", "execute tools", SERIES[3]),
             ("answer_s", "answer (incl. retries)", SERIES[1])]
    left = np.zeros(len(routes))
    y = np.arange(len(routes))[::-1]
    means = rows.groupby("route_gold")[[c for c, _, _ in nodes if c in rows]].mean().reindex(routes).fillna(0)
    for col, label, color in nodes:
        if col not in means:
            continue
        ax2.barh(y, means[col], left=left, color=color, label=label, height=0.55)
        left += means[col].to_numpy()
    ax2.set_yticks(y, [{"sql": "SQL", "semantic": "Semantic", "both": "Both"}[r] for r in routes])
    ax2.set_xlabel("Mean seconds per question")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Where the time goes", loc="left", fontsize=10)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.2)
    _suptitle(fig, TITLES["latency"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plotting.save(fig, path)
    return path


def fig_retrieval(rsum: pd.DataFrame, path: Path) -> Path:
    """Hit@k of recall-search strategies on the questions whose gold answer is a recall campaign."""
    order = ["vehicle filter + semantic (component words)", "vehicle filter + semantic (question)", "vehicle filter + newest first",
             "vehicle filter + random order (expected)", "no filter + semantic (question)"]
    r = rsum.set_index("strategy").reindex([o for o in order if o in set(rsum.strategy)])
    fig, ax = plt.subplots(figsize=(9.5, 4))
    y = np.arange(len(r))[::-1]
    for j, (k, color) in enumerate([("hit@1", SERIES[0]), ("hit@3", SERIES[2]), ("hit@5", SERIES[3])]):
        bars = ax.barh(y + (1 - j) * 0.25, r[k], 0.25, color=color, label=k.replace("hit@", "gold recall in top "))
        for b in bars:
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, [s.replace(" + ", "\n+ ") for s in r.index], fontsize=8.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(f"Share of the {int(r.n.iloc[0])} questions whose answer is a recall campaign")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.2)
    _suptitle(fig, TITLES["retrieval"])
    plotting.save(fig, path)
    return path


def fig_routing(acc: pd.DataFrame, path: Path) -> Path:
    """Route accuracy of the LLM classifier vs the keyword router on templated and separately written questions."""
    sets = ["evaluation questions (templated)", "separately written paraphrases"]
    sets = [s for s in sets if s in set(acc.set)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(sets))
    for j, (router, label, color) in enumerate([("llm", "LLM classify node (qwen3:4b)", SERIES[0]), ("keyword", "keyword router", SERIES[3])]):
        vals = [float(acc[(acc.set == s) & (acc.router == router)].accuracy.iloc[0]) for s in sets]
        bars = ax.bar(x + (j - 0.5) * 0.34, vals, 0.34, color=color, label=label)
        _bar_labels(ax, bars)
    ns = [int(acc[acc.set == s].n.iloc[0]) for s in sets]
    ax.set_xticks(x, [f"{s}\n(n = {n})" for s, n in zip(sets, ns)])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Route accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.2)
    _suptitle(fig, TITLES["routing"])
    plotting.save(fig, path)
    return path


def fig_validation(errors: pd.DataFrame, n_questions: int, path: Path) -> Path:
    """Validator messages on the first plan, by category, split by whether a retry produced a valid plan."""
    first = errors[errors.attempt == 1].drop_duplicates(["qid", "category"])
    counts = first.groupby(["category", "resolved_by_retry"]).size().unstack(fill_value=0)
    for col in (True, False):
        if col not in counts:
            counts[col] = 0
    counts = counts.assign(total=counts[True] + counts[False]).sort_values("total")
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * max(len(counts), 3) + 1.4))
    y = np.arange(len(counts))
    ax.barh(y, counts[True], color=SERIES[2], height=0.6, label="a retry produced a valid plan")
    ax.barh(y, counts[False], left=counts[True], color=SERIES[7], height=0.6, label="still invalid after 3 attempts")
    for yi, t in zip(y, counts.total):
        ax.text(t + 0.1, yi, str(int(t)), va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, counts.index, fontsize=8.5)
    ax.set_xlabel(f"Questions (of {n_questions}) whose first plan triggered this validator message")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, counts.total.max() + 0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.2)
    _suptitle(fig, TITLES["validation"])
    plotting.save(fig, path)
    return path


def make_all(results: Path, figs: Path) -> list[Path]:
    """Redraw every figure from the CSVs in ``results``."""
    plotting.apply_style()
    summary = pd.read_csv(results / "summary_by_route.csv")
    out = [
        fig_scorecard(summary, figs / "01_agent_scorecard.png"),
        fig_stages(pd.read_csv(results / "failure_stages.csv"), figs / "02_failure_stages.png"),
        fig_retries(summary, figs / "03_validation_retries.png"),
        fig_latency(pd.read_csv(results / "agent_results.csv"), figs / "04_latency.png"),
        fig_retrieval(pd.read_csv(results / "retrieval_summary.csv"), figs / "05_recall_search.png"),
        fig_routing(pd.read_csv(results / "routing_accuracy.csv"), figs / "06_routing.png"),
        fig_validation(pd.read_csv(results / "plan_validation_errors.csv"), int(summary.loc[summary.route == "all", "n"].iloc[0]),
                       figs / "07_validation_errors.png"),
    ]
    return out
