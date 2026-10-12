"""Redraw every figure from reports/results/ without re-running the experiments.

    python scripts/make_figures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetyagent.figures import make_all  # noqa: E402

if __name__ == "__main__":
    for p in make_all(ROOT / "reports" / "results", ROOT / "reports" / "figures"):
        print("wrote", p.relative_to(ROOT))
