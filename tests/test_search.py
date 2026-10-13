"""Correctness of the budgeted search: with an unlimited budget system A must
prove a winning first move on 100% of certified puzzles, and every proof must
agree with the exhaustive teacher."""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from minichess import Board
from search import Searcher, HeuristicPolicy
from puzzles import generate

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "data", "puzzles.jsonl")
    if os.path.exists(path) and len(sys.argv) < 2:
        ps = [json.loads(l) for l in open(path)]
        ps = [p for p in ps if p["n"] <= 2][:200] + [p for p in ps if p["n"] == 3][:30]
    else:
        g = generate(seed=7, n_target=10)
        ps = g[1] + g[2] + g[3]
    ok = 0
    for p in ps:
        b = Board(p["fen"])
        s = Searcher(HeuristicPolicy(), budget=10 ** 7)
        m, proven = s.solve(b, p["n"])
        assert proven, p
        assert b.uci(m) in p["keys"], (p, b.uci(m))
        assert b.fen() == p["fen"], "board not restored"
        ok += 1
    print(f"A with unlimited budget proved the correct key on {ok}/{len(ps)} puzzles")
