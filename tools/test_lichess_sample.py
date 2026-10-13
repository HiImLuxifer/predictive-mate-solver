import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from minichess import Board
from search import Searcher, HeuristicPolicy, ModelPolicy
from regulator import solve_regulated, RegParams
from model import Predictor
import pickle

pred = Predictor(pickle.load(open(os.path.join(os.path.dirname(__file__), "..", "models", "gbm.pkl"), "rb")))
reg_params = RegParams()

with open(os.path.join(os.path.dirname(__file__), "..", "data", "puzzles_lichess_sample.jsonl")) as f:
    puzzles = [json.loads(line) for line in f]

for target_n in (1, 2, 3):
    p = next(x for x in puzzles if x["n"] == target_n)
    print("=" * 70)
    print(f"LICHESS PUZZLE: id={p['puzzle_id']} (Rating {p['rating']}) | Matto in {p['n']}")
    print(f"Lichess Game URL: {p['game_url']}")
    print(f"FEN: {p['fen']}")
    print(f"Mossa vincente: {p['keys']} ({p['key_san']})")
    
    fen = p["fen"]
    n = p["n"]
    budget = 40 if n <= 2 else 200
    
    # Sistema A
    sa = Searcher(HeuristicPolicy(), budget=budget)
    ma, proven_a = sa.solve(Board(fen), n)
    move_a = Board(fen).uci(ma) if ma is not None else "None"
    print(f"[A - Euristica]   Mossa: {move_a:5} | Corretta: {move_a in p['keys']:<5} | Dimostrata: {proven_a:<5} | Espansioni: {sa.used}")
    
    # Sistema B
    sb = Searcher(ModelPolicy(pred), budget=budget)
    mb, proven_b = sb.solve(Board(fen), n)
    move_b = Board(fen).uci(mb) if mb is not None else "None"
    print(f"[B - Predittore]  Mossa: {move_b:5} | Corretta: {move_b in p['keys']:<5} | Dimostrata: {proven_b:<5} | Espansioni: {sb.used}")
    
    # Sistema C
    sc = Searcher(ModelPolicy(pred), budget=budget)
    mc, proven_c = solve_regulated(sc, Board(fen), n, pred, reg_params)
    move_c = Board(fen).uci(mc) if mc is not None else "None"
    print(f"[C - Regolatore]  Mossa: {move_c:5} | Corretta: {move_c in p['keys']:<5} | Dimostrata: {proven_c:<5} | Espansioni: {sc.used}")
