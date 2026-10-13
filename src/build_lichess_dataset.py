"""Build dataset directly from Lichess open database.

Downloads/streams chunks from https://database.lichess.org/lichess_db_puzzle.csv.zst,
filters mateIn1, mateIn2, mateIn3, certifies them with Teacher in parallel,
and produces:
  - data/puzzles.jsonl
  - data/split.json
  - data/puzzles_n3_extra.jsonl
"""
import argparse
import csv
import io
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool

from minichess import Board
from puzzles import certify

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
CSV_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

def _certify_worker(item):
    fen, n_theme, p_id, game_url, rating = item
    try:
        cert = certify(fen, max_n=3)
        if cert and cert["n"] == n_theme:
            cert["puzzle_id"] = p_id
            cert["game_url"] = game_url
            cert["rating"] = rating
            return cert
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", type=int, default=2500)
    parser.add_argument("--n2", type=int, default=2500)
    parser.add_argument("--n3", type=int, default=500)
    parser.add_argument("--n3_extra", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    os.makedirs(DATA, exist_ok=True)
    want = {1: args.n1, 2: args.n2, 3: args.n3 + args.n3_extra}
    got = {1: [], 2: [], 3: []}
    seen = set()

    print(f"Obiettivo puzzle Lichess: N=1: {want[1]}, N=2: {want[2]}, N=3: {want[3]}")
    t0 = time.time()

    curl_cmd = ["curl", "-s", "-N", CSV_URL]
    zstd_cmd = ["zstd", "-d", "-c"]

    p_curl = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE)
    p_zstd = subprocess.Popen(zstd_cmd, stdin=p_curl.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    p_curl.stdout.close()

    reader = csv.DictReader(p_zstd.stdout)

    batch_candidates = []
    pool = Pool(args.workers)

    try:
        for row in reader:
            if all(len(got[k]) >= want[k] for k in want):
                break

            themes = row.get("Themes", "").split()
            mate_theme = None
            for t in ("mateIn1", "mateIn2", "mateIn3"):
                if t in themes:
                    mate_theme = t
                    break
            if not mate_theme:
                continue

            n_theme = int(mate_theme[-1])
            if len(got[n_theme]) >= want[n_theme]:
                continue

            moves = row.get("Moves", "").split()
            if not moves:
                continue

            try:
                b = Board(row["FEN"])
                b.make(b.parse_uci(moves[0]))
                fen = b.fen()
            except Exception:
                continue

            if fen in seen:
                continue
            seen.add(fen)

            batch_candidates.append((fen, n_theme, row.get("PuzzleId"), row.get("GameUrl"), int(row.get("Rating", 0))))

            if len(batch_candidates) >= args.workers * 25:
                results = pool.map(_certify_worker, batch_candidates)
                batch_candidates.clear()
                for c in results:
                    if c and len(got[c["n"]]) < want[c["n"]]:
                        got[c["n"]].append(c)

                dt = time.time() - t0
                print(f"[{dt:6.0f}s] Lichess Certificati -> N=1: {len(got[1])}/{want[1]}  N=2: {len(got[2])}/{want[2]}  N=3: {len(got[3])}/{want[3]}", flush=True)

        if batch_candidates:
            results = pool.map(_certify_worker, batch_candidates)
            batch_candidates.clear()
            for c in results:
                if c and len(got[c["n"]]) < want[c["n"]]:
                    got[c["n"]].append(c)

    finally:
        pool.close()
        pool.join()
        p_zstd.terminate()
        p_curl.terminate()

    print(f"\nDownload & Certificazione completati in {time.time() - t0:.1f}s")
    for k in (1, 2, 3):
        print(f"  N={k}: {len(got[k])}")

    rng = random.Random(args.seed)

    # Separiamo N=3 tra dataset principale (500) e extra (600)
    got_n3_all = got[3]
    rng.shuffle(got_n3_all)
    got_n3_main = got_n3_all[:args.n3]
    got_n3_extra = got_n3_all[args.n3:]

    # Salvataggio puzzles.jsonl e split.json
    allp, split = [], {"train": [], "val": [], "test": []}
    
    for n in (1, 2, 3):
        ps = got[n] if n < 3 else got_n3_main
        rng.shuffle(ps)
        for i, p in enumerate(ps):
            p["id"] = f"m{n}_{i:05d}"
            allp.append(p)
            if n == 3:
                split["test"].append(p["id"])
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

    print(f"Creato puzzles.jsonl ({len(allp)} puzzle) e split.json: { {k: len(v) for k, v in split.items()} }")

    # Salvataggio puzzles_n3_extra.jsonl
    n_val = len(got_n3_extra) // 3
    with open(os.path.join(DATA, "puzzles_n3_extra.jsonl"), "w") as f:
        for i, p in enumerate(got_n3_extra):
            p["id"] = f"x3_val_{i:04d}" if i < n_val else f"x3_train_{i:04d}"
            f.write(json.dumps(p) + "\n")

    print(f"Creato puzzles_n3_extra.jsonl ({len(got_n3_extra)} puzzle)")

if __name__ == "__main__":
    main()
