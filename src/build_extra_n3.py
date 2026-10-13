"""Extra mate-in-3 puzzles (disjoint seeds from build_data.py), used ONLY for
ablation 4b (train WITH mate-in-3, i.e. no distribution shift) and as a small
mate-in-3 tuning/validation set. The main mate-in-3 test set is untouched.

Usage: python build_extra_n3.py --n 600 --workers 1
Output: data/puzzles_n3_extra.jsonl  (ids x3_train_*, x3_val_*)
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

from build_data import work, DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=990000)
    a = ap.parse_args()
    main_fens = {json.loads(l)["fen"] for l in open(os.path.join(DATA, "puzzles.jsonl"))}
    got, seen, seed, t0 = [], set(), a.seed, time.time()
    with Pool(a.workers) as pool:
        while len(got) < a.n:
            batch = [(seed + i, 200, 3) for i in range(a.workers * 4)]
            seed += len(batch)
            for res in pool.imap_unordered(work, batch):
                for p in res:
                    if p["n"] == 3 and p["fen"] not in seen and p["fen"] not in main_fens and len(got) < a.n:
                        seen.add(p["fen"])
                        got.append(p)
            print(f"[{time.time() - t0:6.0f}s] N=3 extra: {len(got)}/{a.n}", flush=True)
    random.Random(a.seed).shuffle(got)
    n_val = len(got) // 3
    with open(os.path.join(DATA, "puzzles_n3_extra.jsonl"), "w") as f:
        for i, p in enumerate(got):
            p["id"] = f"x3_val_{i:04d}" if i < n_val else f"x3_train_{i:04d}"
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
