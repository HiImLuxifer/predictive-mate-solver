"""Tune regulator hyper-parameters on VALIDATION puzzles only.

Objective: mean solve rate of C over the budgets, ties broken by proven rate.
Never touches the test split.
"""
import itertools
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

from model import Predictor
from minichess import Board
from search import Searcher, ModelPolicy
from regulator import solve_regulated, RegParams

ROOT = os.path.join(os.path.dirname(__file__), "..")
_G = {}


def _init(model_path):
    _G["pred"] = Predictor(pickle.load(open(model_path, "rb")))


def _eval(args):
    pz, configs, budgets = args
    pred = _G["pred"]
    pred.clear()
    out = []
    for ci, cfg in enumerate(configs):
        for bud in budgets:
            b = Board(pz["fen"])
            s = Searcher(ModelPolicy(pred), bud)
            if cfg is None:      # system B reference
                m, pr = s.solve(b, pz["n"])
            else:
                m, pr = solve_regulated(s, b, pz["n"], pred, RegParams(**cfg))
            out.append((ci, bud, b.uci(m) in pz["keys"], pr, s.used))
    return out


def tune(puzzles, budgets, grid, base, workers=1, model_path=None):
    model_path = model_path or os.path.join(ROOT, "models", "gbm.pkl")
    keys = list(grid)
    configs = [None] + [{**base, **dict(zip(keys, v))} for v in itertools.product(*grid.values())]
    with Pool(workers, initializer=_init, initargs=(model_path,)) as pool:
        res = sum(pool.map(_eval, [(p, configs, budgets) for p in puzzles], chunksize=4), [])
    res = np.array(res, dtype=float)
    rows = []
    for ci, cfg in enumerate(configs):
        r = res[res[:, 0] == ci]
        rows.append({"config": "B (reference)" if cfg is None else {k: cfg[k] for k in keys},
                     "solve": r[:, 2].mean(), "proven": r[:, 3].mean(), "used": r[:, 4].mean(),
                     **{f"solve@{b}": r[r[:, 1] == b][:, 2].mean() for b in budgets}})
    return rows


if __name__ == "__main__":
    from experiment import load_puzzles
    which = sys.argv[1] if len(sys.argv) > 1 else "n2"
    T = json.load(open(os.path.join(ROOT, "results", "predictor_metrics.json")))["gbm"]["T"]
    base = vars(RegParams(T=T))
    grid = {"eps0": [0.05, 0.3], "s0_frac": [1 / 32, 1 / 8], "c_min_mult": [1.0, 2.0], "k_R": [0, 2], "lam": [0.0, 1.0]}
    if which == "n2":
        ps = [p for p in load_puzzles("val") if p["n"] == 2]
        budgets = [8, 16, 32]
    else:
        ps = [json.loads(l) for l in open(os.path.join(ROOT, "data", "puzzles_n3_extra.jsonl"))]
        ps = [p for p in ps if p["id"].startswith("x3_val")][:120]
        budgets = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [100, 300, 1000]
    t0 = time.time()
    rows = tune(ps, budgets, grid, base, workers=int(os.environ.get("WORKERS", 1)))
    rows.sort(key=lambda r: (-r["solve"], -r["proven"]))
    for r in rows[:8] + [r for r in rows if r["config"] == "B (reference)"]:
        print(json.dumps(r))
    json.dump(rows, open(os.path.join(ROOT, "results", f"tuning_{which}.json"), "w"), indent=1)
    print(f"{time.time() - t0:.0f}s")
