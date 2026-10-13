"""Isolated wall-clock per search (fresh prediction cache per run, 1 process).

In the benchmark the prediction cache is shared by all systems run on the same
puzzle, so its time column is not comparable; this script measures it fairly.
"""
import json
import os
import pickle
import time

import numpy as np

from experiment import load_puzzles, ROOT
from model import Predictor
from minichess import Board
from search import Searcher, HeuristicPolicy, ModelPolicy
from regulator import solve_regulated
from run_all import config


def main(n_puzzles=60):
    params, _ = config()
    pred = Predictor(pickle.load(open(os.path.join(ROOT, "models", "gbm.pkl"), "rb")))
    out = {}
    for n, bud in ((2, 16), (3, 400)):
        ps = [p for p in load_puzzles("test") if p["n"] == n][:n_puzzles]
        for sysname in "ABC":
            ts, used = [], []
            for p in ps:
                pred.clear()
                b = Board(p["fen"])
                t0 = time.perf_counter()
                if sysname == "A":
                    s = Searcher(HeuristicPolicy(), bud)
                    s.solve(b, n)
                elif sysname == "B":
                    s = Searcher(ModelPolicy(pred), bud)
                    s.solve(b, n)
                else:
                    s = Searcher(ModelPolicy(pred), bud)
                    solve_regulated(s, b, n, pred, params)
                ts.append(time.perf_counter() - t0)
                used.append(s.used)
            out[f"N{n}@{bud}_{sysname}"] = {"ms_per_search": 1000 * float(np.mean(ts)),
                                            "ms_per_expansion": 1000 * float(np.sum(ts) / np.sum(used))}
    json.dump(out, open(os.path.join(ROOT, "results", "timing.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
