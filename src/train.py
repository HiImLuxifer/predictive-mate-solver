"""Step 3-4 + 6: build teacher targets, train P_theta, evaluate, calibrate T.

Usage: python train.py
Outputs (in models/ and results/):
  models/gbm.pkl, models/logreg.pkl
  results/predictor_metrics.json
  results/dataset_stats.json
"""
import json
import os
import pickle
import time

import numpy as np

from model import (build_dataset, train_gbm, train_logreg, pair_metrics, Predictor,
                   calibrate_temperature, root_metrics)
from experiment import load_puzzles

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.path.join(ROOT, "models")
RES = os.path.join(ROOT, "results")


def main(workers=2):
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    tr = load_puzzles("train")
    va = load_puzzles("val")
    te = load_puzzles("test")
    t0 = time.time()
    Xtr, ytr, ttr, ptr = build_dataset(tr, workers)
    Xva, yva, tva, pva = build_dataset(va, workers)
    te12 = [p for p in te if p["n"] <= 2]
    te3 = [p for p in te if p["n"] == 3]
    Xte, yte, tte, pte = build_dataset(te12, workers)
    Xte3, yte3, tte3, pte3 = build_dataset(te3, workers)
    t_label = time.time() - t0
    stats = {"label_time_s": t_label,
             "train": {"puzzles": len(tr), "pairs": int(len(ytr)), "pos_rate": float(ytr.mean()),
                       "by_type": {t: int((ttr == t).sum()) for t in set(ttr)}},
             "val_pairs": int(len(yva)), "test12_pairs": int(len(yte)), "test3_pairs": int(len(yte3))}
    print(json.dumps(stats, indent=1))
    np.savez_compressed(os.path.join(ROOT, "data", "pairs.npz"), Xtr=Xtr, ytr=ytr, ttr=ttr, Xva=Xva, yva=yva,
                        tva=tva, Xte=Xte, yte=yte, tte=tte, Xte3=Xte3, yte3=yte3, tte3=tte3,
                        ptr=ptr, pva=pva, pte=pte, pte3=pte3)

    t0 = time.time()
    gbm = train_gbm(Xtr, ytr)
    lr = train_logreg(Xtr, ytr)
    print(f"trained in {time.time() - t0:.1f}s")
    for name, m in (("gbm", gbm), ("logreg", lr)):
        with open(os.path.join(MODELS, f"{name}.pkl"), "wb") as f:
            pickle.dump(m, f)

    metrics = {}
    for name, m in (("gbm", gbm), ("logreg", lr)):
        mm = {}
        for split, X, y, t in (("val", Xva, yva, tva), ("test_n12", Xte, yte, tte), ("test_n3", Xte3, yte3, tte3)):
            mm[split] = {"all": pair_metrics(m, X, y)}
            for typ in ("root", "and", "or_int"):
                s = t == typ
                if s.sum() > 0 and len(set(y[s])) > 1:
                    mm[split][typ] = pair_metrics(m, X[s], y[s])
        pred = Predictor(m)
        T, nll = calibrate_temperature(pred, va)
        mm["T"] = T
        mm["T_nll"] = nll
        for split, ps in (("val", va), ("test_n1", [p for p in te if p["n"] == 1]),
                          ("test_n2", [p for p in te if p["n"] == 2]), ("test_n3", te3)):
            mm.setdefault("root_decision", {})[split] = root_metrics(pred, ps)
        metrics[name] = mm
        print(name, json.dumps({k: mm["root_decision"][k] for k in mm["root_decision"]}, indent=1), "T=", T)
    # feature importance proxy for the report (logreg coefficients on standardised features)
    from board_utils import FEATURE_NAMES
    coefs = lr[-1].coef_[0]
    metrics["logreg_coefficients"] = dict(sorted(zip(FEATURE_NAMES, map(float, coefs)), key=lambda x: -abs(x[1])))
    json.dump(stats, open(os.path.join(RES, "dataset_stats.json"), "w"), indent=1)
    json.dump(metrics, open(os.path.join(RES, "predictor_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
