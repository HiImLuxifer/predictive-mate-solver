"""Step 1: generate and certify synthetic puzzles in parallel, then split.

Usage: python build_data.py --n1 2500 --n2 2500 --n3 500 --workers 2
Output: data/puzzles.jsonl (all), data/split.json (ids per split)
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

from puzzles import random_position, certify

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def work(args):
    seed, n_pos, need_max = args
    rng = random.Random(seed)
    found = []
    for _ in range(n_pos):
        fen = random_position(rng)
        if fen is None:
            continue
        p = certify(fen, max_n=need_max)
        if p:
            found.append(p)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=2500)
    ap.add_argument("--n2", type=int, default=2500)
    ap.add_argument("--n3", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    want = {1: a.n1, 2: a.n2, 3: a.n3}
    got = {1: [], 2: [], 3: []}
    seen = set()
    t0 = time.time()
    seed = a.seed
    with Pool(a.workers) as pool:
        while any(len(got[k]) < want[k] for k in want):
            need_max = max(k for k in want if len(got[k]) < want[k])
            batch = [(seed + i, 200, need_max) for i in range(a.workers * 4)]
            seed += len(batch)
            for res in pool.imap_unordered(work, batch):
                for p in res:
                    if p["fen"] in seen or len(got[p["n"]]) >= want[p["n"]]:
                        continue
                    seen.add(p["fen"])
                    got[p["n"]].append(p)
            print(f"[{time.time() - t0:6.0f}s] " + "  ".join(f"N={k}: {len(got[k])}/{want[k]}" for k in want), flush=True)
    # ---- split (puzzles are independent random positions -> split by puzzle id)
    rng = random.Random(a.seed)
    allp, split = [], {"train": [], "val": [], "test": []}
    for n in (1, 2, 3):
        ps = got[n]
        rng.shuffle(ps)
        for i, p in enumerate(ps):
            p["id"] = f"m{n}_{i:05d}"
            allp.append(p)
            if n == 3:
                split["test"].append(p["id"])              # mate-in-3: test only (shift)
            elif i < 400:
                split["test"].append(p["id"])
            elif i < 600:
                split["val"].append(p["id"])
            else:
                split["train"].append(p["id"])
    with open(os.path.join(DATA, "puzzles.jsonl"), "w") as f:
        for p in allp:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(DATA, "split.json"), "w") as f:
        json.dump(split, f)
    print({k: len(v) for k, v in split.items()}, f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
