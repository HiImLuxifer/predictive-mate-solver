"""Equal WALL-CLOCK comparison (complements the equal-expansion benchmark).

Each (puzzle, system, budget) runs alone with a cleared prediction cache, so
its time is honest. A gets much larger expansion budgets to span the same time.
"""
import json
import os
import time

from experiment import run_benchmark, load_puzzles, ROOT
from run_all import config

BUDGETS = {
    2: {"A": [4, 16, 64, 256, 1024], "B": [4, 8, 16, 32, 64], "C": [4, 8, 16, 32, 64]},
    3: {"A": [50, 200, 800, 3200, 12800], "B": [50, 100, 200, 400, 800], "C": [50, 100, 200, 400, 800]},
}

if __name__ == "__main__":
    params, extra = config()
    test = [p for p in load_puzzles("test") if p["n"] >= 2]
    t0 = time.time()
    workers = int(os.environ.get("WORKERS", 4))
    run_benchmark(test, ["A", "B", "C"], BUDGETS, os.path.join(ROOT, "models", "gbm.pkl"), params,
                  workers=workers, extra_models=extra, tag="walltime", fair_time=True)
    print(f"done in {time.time() - t0:.0f}s")
