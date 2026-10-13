"""Validate the move generator against the standard perft reference counts
(https://www.chessprogramming.org/Perft_Results)."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from minichess import Board, perft

CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),  # Kiwipete
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890]),
]

if __name__ == "__main__":
    total, t0 = 0, time.time()
    for fen, expected in CASES:
        b = Board(fen)
        for d, exp in enumerate(expected, 1):
            got = perft(b, d)
            total += got
            status = "ok" if got == exp else "FAIL"
            print(f"{status:4s} depth {d}: {got:>7} (expected {exp:>7})  {fen[:40]}")
            assert got == exp
            assert b.fen().split()[:4] == fen.split()[:4], "board not restored"
    dt = time.time() - t0
    print(f"all perft tests passed; {total} leaf nodes in {dt:.1f}s ({total / dt:.0f} nodes/s)")
