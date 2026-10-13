"""Step 8-10: benchmark A / B / C (+ ablations) under equal online budgets.

Every system receives the same budget of EXPANSIONS (legal-move generations)
per puzzle. Feature/prediction time is not an expansion (like gives_check in the
baseline heuristic) but wall-clock time is recorded for transparency.
"""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from minichess import Board
from search import (Searcher, HeuristicPolicy, ModelPolicy, softmax_logits, entropy_norm, frontier_stats,
                    UNK, PROVEN, DISPROVEN)
from regulator import solve_regulated, RegParams
from model import Predictor, root_candidates

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

_G = {}

# regulator ablations: parameter overrides of system C
C_VARIANTS = {
    "C_flat": {"flatten": True},          # ablation 2: pi uniform (T -> infinity)
    "C_noR": {"use_R": False},            # ablation 3: no 'needs more thinking' signal
    "C_noStop": {"early_stop": False},    # stopping rule disabled
    "C_admit": {"k_R": 2},                # R_hat admission of deferred moves switched on
}


def _init(model_path, params, sigma, extra_models):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    _G["pred"] = Predictor(model)
    _G["pred_noise"] = Predictor(model, noise_sigma=sigma)
    _G["params"] = params
    _G["extra"] = {}
    _G["extra_T"] = {}
    for name, (path, T) in (extra_models or {}).items():
        with open(path, "rb") as f:
            _G["extra"][name] = Predictor(pickle.load(f))
        _G["extra_T"][name] = T


def posthoc_rho(searcher, board, moves_pi, chosen):
    """rho at stop, measured identically for every system with the main pi."""
    st = searcher.root_status() if searcher.root is not None else {}
    rho_unres, rho_untouched = 0.0, 0.0
    for m, p in moves_pi.items():
        s, touched = st.get(m, (UNK, False))
        if m == chosen or s != UNK:
            continue
        rho_unres += p
        if not touched:
            rho_untouched += p
    return rho_unres, rho_untouched


def run_one(system, pz, budget):
    b = Board(pz["fen"])
    n = pz["n"]
    P = _G["params"]
    pred = _G["pred"]
    t0 = time.time()
    trace = []
    if system == "A":
        s = Searcher(HeuristicPolicy(), budget)
        m, proven = s.solve(b, n)
    elif system.startswith("B"):
        pr = pred if system == "B" else _G["pred_noise"] if system == "B_noise" else _G["extra"][system[2:]]
        s = Searcher(ModelPolicy(pr), budget)
        m, proven = s.solve(b, n)
    else:
        pr = pred
        q = RegParams(**vars(P))
        if system in C_VARIANTS:
            for k, v in C_VARIANTS[system].items():
                setattr(q, k, v)
        elif system == "C_noise":
            pr = _G["pred_noise"]
        elif system.startswith("C_") and system[2:] in _G["extra"]:
            pr = _G["extra"][system[2:]]
            q.T = _G["extra_T"][system[2:]]
        s = Searcher(ModelPolicy(pr), budget)
        m, proven = solve_regulated(s, b, n, pr, q, trace=trace)
    dt = time.time() - t0
    uci = b.uci(m) if m is not None else None
    # post-hoc rho with the main (clean) predictor
    moves = root_candidates(b, n)
    pi = softmax_logits(pred.predict(b, moves, b.side, 2 * n - 1), P.T)
    rho, rho_unt = posthoc_rho(s, b, dict(zip(moves, pi)), m)
    return {"id": pz["id"], "n": n, "system": system, "budget": budget, "move": uci,
            "correct": uci in pz["keys"], "proven": bool(proven), "used": s.used, "time": dt,
            "rho": rho, "rho_untouched": rho_unt, **{f"fr_{k}": v for k, v in frontier_stats(s).items()},
            "trace": trace if system == "C" else None}


def puzzle_info(pz):
    """Prediction-side information about the key move (for Q13 / Q16)."""
    pred = _G["pred"]
    b = Board(pz["fen"])
    n = pz["n"]
    moves = root_candidates(b, n)
    p = pred.predict(b, moves, b.side, 2 * n - 1)
    pi = softmax_logits(p, _G["params"].T)
    names = [b.uci(m) for m in moves]
    order = list(np.argsort(-p, kind="stable"))
    ki = [names.index(k) for k in pz["keys"]]
    best = min(ki, key=lambda i: order.index(i))
    X = pred.features(b, moves, b.side, 2 * n - 1)
    w = 1.0 if (X[best, pred.col("is_quiet")] or X[best, pred.col("is_sacrifice")]) else 0.5
    H = entropy_norm(pi)
    return {"id": pz["id"], "n": n, "key_rank": order.index(best) + 1, "p_key": float(p[best]),
            "pi_key": float(pi[best]), "pi_top": float(pi.max()), "H": H, "R_key": H * w,
            "n_cand": len(moves), "sacrifice": pz["sacrifice"], "quiet": pz["quiet"], "check": pz["check"]}


def _work(args):
    pz, jobs = args
    _G["pred"].clear()
    _G["pred_noise"].clear()
    for p in _G["extra"].values():
        p.clear()
    out = [run_one(sysname, pz, bud) for sysname, bud in jobs]
    info = puzzle_info(pz)
    return out, info


def run_benchmark(puzzles, systems, budgets_by_n, model_path, params, sigma=1.5, workers=2,
                  extra_models=None, tag="main", fair_time=False):
    """fair_time=True: one (system, budget) per task with a cleared prediction
    cache, so the recorded wall-clock time of every run is comparable."""
    os.makedirs(RES, exist_ok=True)
    tasks = []
    for pz in puzzles:
        sy = systems[pz["n"]] if isinstance(systems, dict) else systems
        bu = budgets_by_n[pz["n"]]
        if fair_time:
            for s in sy:
                for b in (bu[s] if isinstance(bu, dict) else bu):
                    tasks.append((pz, [(s, b)]))
        else:
            tasks.append((pz, [(s, b) for b in bu for s in sy]))
    rows, infos, seen = [], [], set()
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(model_path, params, sigma, extra_models)) as pool:
        for i, (out, info) in enumerate(pool.imap_unordered(_work, tasks, chunksize=2)):
            rows.extend(out)
            if not any(x["id"] == info["id"] for x in infos[-1:]) and info["id"] not in seen:
                infos.append(info)
                seen.add(info["id"])
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} puzzles  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(RES, f"runs_{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(RES, f"info_{tag}.jsonl"), "w") as f:
        for r in infos:
            f.write(json.dumps(r) + "\n")
    return rows, infos


def load_puzzles(split=None):
    ps = [json.loads(l) for l in open(os.path.join(DATA, "puzzles.jsonl"))]
    if split is None:
        return ps
    ids = set(json.load(open(os.path.join(DATA, "split.json")))[split])
    return [p for p in ps if p["id"] in ids]
