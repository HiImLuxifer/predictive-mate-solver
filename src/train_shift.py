"""Ablation 4b: the same predictor trained WITH mate-in-3 examples.

Main model  : trained on mate-in-1/2 only -> mate-in-3 test is out of distribution.
This model  : main training set + 400 extra mate-in-3 puzzles (x3_train_*),
              T calibrated on val + x3_val_*. The mate-in-3 TEST set is unchanged.
"""
import json
import os
import pickle

import numpy as np

from model import build_dataset, train_gbm, Predictor, calibrate_temperature, root_metrics
from experiment import load_puzzles

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    extra = [json.loads(l) for l in open(os.path.join(ROOT, "data", "puzzles_n3_extra.jsonl"))]
    x3_train = [p for p in extra if p["id"].startswith("x3_train")]
    x3_val = [p for p in extra if p["id"].startswith("x3_val")]
    d = np.load(os.path.join(ROOT, "data", "pairs.npz"))
    X3, y3, _, _ = build_dataset(x3_train, 2)
    X = np.vstack([d["Xtr"], X3])
    y = np.concatenate([d["ytr"], y3])
    m = train_gbm(X, y)
    pickle.dump(m, open(os.path.join(ROOT, "models", "gbm_with_n3.pkl"), "wb"))
    pred = Predictor(m)
    T, _ = calibrate_temperature(pred, load_puzzles("val") + x3_val)
    te3 = [p for p in load_puzzles("test") if p["n"] == 3]
    out = {"extra_n3_train_puzzles": len(x3_train), "extra_pairs": int(len(y3)), "T": T,
           "test_n3_root": root_metrics(pred, te3)}
    main_pred = Predictor(pickle.load(open(os.path.join(ROOT, "models", "gbm.pkl"), "rb")))
    out["main_model_test_n3_root"] = root_metrics(main_pred, te3)
    json.dump(out, open(os.path.join(ROOT, "results", "shift_model.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
