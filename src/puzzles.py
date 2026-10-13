"""Puzzle generation (synthetic) and an optional Lichess CSV loader.

Synthetic generator
-------------------
Random but chess-plausible positions: the defender king is often castled
behind a pawn shield (back-rank / smothered / corridor motifs), the attacker
has a material edge. Every candidate is certified by the exhaustive teacher:

  * attacker mates in exactly N moves (mates(2N-1) and not mates(2N-3));
  * at most MAX_KEYS first moves work (the puzzle has a real decision);
  * N in {1, 2, 3}.

The set of *all* winning first moves is stored, so 'correct decision' is
defined independently of any search system.
"""
import csv
import io
import random

from minichess import Board, KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN, KING_T
from teacher import Teacher
from board_utils import classify_move

MAX_KEYS = 2


def _cheb(a, b):
    return max(abs(a % 8 - b % 8), abs(a // 8 - b // 8))


def random_position(rng):
    """Returns a FEN with the attacker to move, or None if illegal."""
    att = rng.choice([1, -1])
    dfd = -att
    b = [0] * 64
    # defender back rank from the defender's point of view
    back = 0 if dfd > 0 else 7
    fwd = 1 if dfd > 0 else -1           # direction the defender's pawns advance
    occupied = set()

    def put(sq, p):
        b[sq] = p
        occupied.add(sq)

    style = rng.random()
    if style < 0.55:                                     # castled king behind pawns
        kf = rng.choice([6, 6, 7, 6, 2, 1])
        kr = back if rng.random() < 0.8 else back + fwd
        ksq = kr * 8 + kf
        put(ksq, dfd * KING)
        for df in (-1, 0, 1):
            f = kf + df
            if 0 <= f < 8 and rng.random() < 0.85:
                r = kr + fwd * (1 if rng.random() < 0.75 else 2)
                if 0 < r < 7:
                    s = r * 8 + f
                    if s not in occupied:
                        put(s, dfd * PAWN)
    else:                                                # king somewhere in the open
        ksq = rng.randrange(64)
        put(ksq, dfd * KING)
    # defender extra material
    for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 4])):
        p = rng.choice([PAWN, PAWN, KNIGHT, BISHOP, ROOK, ROOK, QUEEN])
        s = rng.randrange(64)
        if s in occupied or (p == PAWN and s // 8 in (0, 7)):
            continue
        put(s, dfd * p)
    # attacker king: away from the defender king
    for _ in range(20):
        s = rng.randrange(64)
        if s not in occupied and _cheb(s, ksq) >= 3:
            put(s, att * KING)
            aksq = s
            break
    else:
        return None
    # attacker material: a material edge, mostly pieces
    pool = [QUEEN, ROOK, ROOK, BISHOP, BISHOP, KNIGHT, KNIGHT, PAWN, PAWN]
    for _ in range(rng.choice([2, 2, 3, 3, 4, 5])):
        p = rng.choice(pool)
        s = rng.randrange(64)
        if s in occupied or (p == PAWN and s // 8 in (0, 7)):
            continue
        # bias pieces towards the defender king half of the board
        if rng.random() < 0.6 and _cheb(s, ksq) > 4:
            continue
        put(s, att * p)
    board = Board("8/8/8/8/8/8/8/8 w - - 0 1")
    board.b = b
    board.kings = [None, -1, -1]
    for s, p in enumerate(b):
        if abs(p) == KING:
            board.kings[1 if p > 0 else -1] = s
    if board.kings[1] < 0 or board.kings[-1] < 0:
        return None
    board.side = att
    board.castling = 0
    board.ep = -1
    if board.in_check(dfd):         # side not to move must not be in check
        return None
    if board.in_check(att) and not board.legal_moves():
        return None
    return board.fen()


def certify(fen, max_n=3, teacher=None):
    """Return puzzle dict if fen is an exact mate-in-N with <= MAX_KEYS keys."""
    t = teacher or Teacher()
    board = Board(fen)
    n = t.mate_length(board, max_n)
    if n is None:
        return None
    labels = t.label_moves(board, 2 * n - 1, True)
    keys = [m for m, y in labels if y]
    if not 1 <= len(keys) <= MAX_KEYS:
        return None
    tags = classify_move(board, keys[0], board.side)
    return {
        "fen": fen, "n": n,
        "keys": [board.uci(m) for m in keys],
        "key_san": board.san(keys[0]),
        "n_legal": len(board.legal_moves()),
        **tags,
    }


def generate(seed, n_target, max_tries=10 ** 7, max_n=3, want=None):
    """Generate puzzles until counts per N reach `want` (dict N->count)."""
    rng = random.Random(seed)
    want = want or {1: n_target, 2: n_target, 3: n_target}
    got = {k: [] for k in want}
    seen = set()
    for _ in range(max_tries):
        if all(len(got[k]) >= want[k] for k in want):
            break
        fen = random_position(rng)
        if fen is None or fen in seen:
            continue
        seen.add(fen)
        mn = max(k for k in want if len(got[k]) < want[k])
        p = certify(fen, max_n=mn)
        if p and p["n"] in want and len(got[p["n"]]) < want[p["n"]]:
            got[p["n"]].append(p)
    return got


# ------------------------------------------------------- Lichess loader ---
def load_lichess_csv(path, themes=("mateIn1", "mateIn2", "mateIn3"), limit=None):
    """Optional: load real puzzles from lichess_db_puzzle.csv(.zst).

    Lichess FEN is the position BEFORE the opponent's last move: the first UCI
    move of `Moves` must be played to obtain the puzzle position.
    Returns dicts compatible with the synthetic ones (keys certified later).
    """
    if path.endswith(".zst"):
        import zstandard
        fh = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8")
    else:
        fh = open(path, encoding="utf-8")
    out = []
    for row in csv.DictReader(fh):
        th = row["Themes"].split()
        hit = [t for t in themes if t in th]
        if not hit:
            continue
        b = Board(row["FEN"])
        b.make(b.parse_uci(row["Moves"].split()[0]))
        out.append({"fen": b.fen(), "n": int(hit[0][-1]), "game_id": row["GameUrl"].split("/")[-1].split("#")[0],
                    "lichess_themes": th, "lichess_solution": row["Moves"].split()[1]})
        if limit and len(out) >= limit:
            break
    return out
