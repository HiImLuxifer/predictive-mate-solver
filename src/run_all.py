"""Step 8-9: the full benchmark on the TEST split (A, B, C + ablations).

Usage: python run_all.py            (~10-20 min on 2 cores)
"""
import json
import os
import time

from experiment import run_benchmark, load_puzzles, ROOT
from regulator import RegParams

BUDGETS = {1: [1, 2, 3], 2: [4, 8, 16, 32, 64], 3: [50, 100, 200, 400, 800]}
MAIN = ["A", "B", "C"]
ABLATIONS = ["C_flat", "C_noR", "C_noStop", "C_admit", "B_noise", "C_noise", "B_logreg", "C_logreg"]
SHIFT = ["B_withn3", "C_withn3"]
NOISE_SIGMA = 1.5


def config():
    metrics = json.load(open(os.path.join(ROOT, "results", "predictor_metrics.json")))
    shift = json.load(open(os.path.join(ROOT, "results", "shift_model.json")))
    # regulator chosen on validation (results/tuning_n2.json, tuning_n3.json): max-min over N=2 and N=3
    params = RegParams(T=metrics["gbm"]["T"], eps0=0.05, s0_frac=1 / 8, c_min_mult=2.0, k_R=0, lam=1.0)
    extra = {"logreg": (os.path.join(ROOT, "models", "logreg.pkl"), metrics["logreg"]["T"]),
             "withn3": (os.path.join(ROOT, "models", "gbm_with_n3.pkl"), shift["T"])}
    return params, extra


if __name__ == "__main__":
    params, extra = config()
    systems = {1: MAIN + ABLATIONS, 2: MAIN + ABLATIONS, 3: MAIN + ABLATIONS + SHIFT}
    test = load_puzzles("test")
    t0 = time.time()
    workers = int(os.environ.get("WORKERS", 4))
    run_benchmark(test, systems, BUDGETS, os.path.join(ROOT, "models", "gbm.pkl"), params,
                  sigma=NOISE_SIGMA, workers=workers, extra_models=extra, tag="main")
    json.dump({"budgets": BUDGETS, "systems": systems, "params": vars(params), "noise_sigma": NOISE_SIGMA,
               "extra_models": {k: {"path": os.path.relpath(v[0], ROOT), "T": v[1]} for k, v in extra.items()},
               "runtime_s": time.time() - t0},
              open(os.path.join(ROOT, "results", "config.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")
