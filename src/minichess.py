"""minichess: a small, dependency-free legal chess move generator.

Written for this homework because the execution environment could not install
python-chess. It supports the full rules needed for tactical puzzles
(castling, en passant, promotion, pins, checks) and is validated with the
standard perft node counts in tests/test_perft.py.

Board: list of 64 ints, square = rank*8 + file (a1 = 0, h8 = 63).
Pieces: +1..+6 = white P N B R Q K, negative = black. side: +1 white, -1 black.
Move: int  from | to<<6 | promo<<12 | flag<<15.
"""

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
VALUE = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}
FLAG_EP, FLAG_CASTLE, FLAG_DOUBLE = 1, 2, 3
PIECE_CHARS = {PAWN: "p", KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q", KING: "k"}
CHAR_PIECES = {v: k for k, v in PIECE_CHARS.items()}

# ---------------------------------------------------------------- tables ---
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 0-3 rook, 4-7 bishop


def _on(f, r):
    return 0 <= f < 8 and 0 <= r < 8


KNIGHT_T, KING_T, RAYS = [], [], []
PAWN_FROM = ([], [])  # PAWN_FROM[ci][sq]: squares from which a pawn of colour ci attacks sq
for sq in range(64):
    f, r = sq % 8, sq // 8
    KNIGHT_T.append([(r + dr) * 8 + f + df for df, dr in
                     [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
                     if _on(f + df, r + dr)])
    KING_T.append([(r + dr) * 8 + f + df for df, dr in DIRS if _on(f + df, r + dr)])
    rays = []
    for df, dr in DIRS:
        ray, ff, rr = [], f + df, r + dr
        while _on(ff, rr):
            ray.append(rr * 8 + ff)
            ff += df
            rr += dr
        rays.append(ray)
    RAYS.append(rays)
    PAWN_FROM[0].append([(r - 1) * 8 + f + df for df in (-1, 1) if _on(f + df, r - 1)])  # white pawns
    PAWN_FROM[1].append([(r + 1) * 8 + f + df for df in (-1, 1) if _on(f + df, r + 1)])  # black pawns

# LINE_DIR[a][b] = direction index from a to b if aligned, else -1
LINE_DIR = [[-1] * 64 for _ in range(64)]
for a in range(64):
    for d in range(8):
        for b in RAYS[a][d]:
            LINE_DIR[a][b] = d

SQUARE_NAMES = [f"{'abcdefgh'[s % 8]}{s // 8 + 1}" for s in range(64)]
NAME_TO_SQ = {n: i for i, n in enumerate(SQUARE_NAMES)}


def ci(side):
    return 0 if side > 0 else 1


def make_move(fr, to, promo=0, flag=0):
    return fr | (to << 6) | (promo << 12) | (flag << 15)


def m_from(m):
    return m & 63


def m_to(m):
    return (m >> 6) & 63


def m_promo(m):
    return (m >> 12) & 7


def m_flag(m):
    return m >> 15


# ----------------------------------------------------------------- board ---
class Board:
    __slots__ = ("b", "side", "castling", "ep", "kings", "half", "full")

    def __init__(self, fen=None):
        self.set_fen(fen or "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # ---------------------------------------------------------- fen I/O ---
    def set_fen(self, fen):
        parts = fen.split()
        self.b = [0] * 64
        self.kings = [None, -1, -1]
        r, f = 7, 0
        for ch in parts[0]:
            if ch == "/":
                r -= 1
                f = 0
            elif ch.isdigit():
                f += int(ch)
            else:
                p = CHAR_PIECES[ch.lower()] * (1 if ch.isupper() else -1)
                self.b[r * 8 + f] = p
                if abs(p) == KING:
                    self.kings[1 if p > 0 else -1] = r * 8 + f
                f += 1
        self.side = 1 if parts[1] == "w" else -1
        c = parts[2] if len(parts) > 2 else "-"
        self.castling = (1 if "K" in c else 0) | (2 if "Q" in c else 0) | (4 if "k" in c else 0) | (8 if "q" in c else 0)
        self.ep = NAME_TO_SQ[parts[3]] if len(parts) > 3 and parts[3] != "-" else -1
        self.half = int(parts[4]) if len(parts) > 4 else 0
        self.full = int(parts[5]) if len(parts) > 5 else 1

    def fen(self):
        rows = []
        for r in range(7, -1, -1):
            row, empty = "", 0
            for f in range(8):
                p = self.b[r * 8 + f]
                if p == 0:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    ch = PIECE_CHARS[abs(p)]
                    row += ch.upper() if p > 0 else ch
            if empty:
                row += str(empty)
            rows.append(row)
        c = "".join(ch for bit, ch in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if self.castling & bit) or "-"
        ep = SQUARE_NAMES[self.ep] if self.ep >= 0 else "-"
        return f"{'/'.join(rows)} {'w' if self.side > 0 else 'b'} {c} {ep} {self.half} {self.full}"

    def key(self):
        """Hashable position key (for transposition tables)."""
        return (bytes(p + 6 for p in self.b), self.side, self.castling, self.ep)

    def copy(self):
        n = Board.__new__(Board)
        n.b = self.b[:]
        n.side, n.castling, n.ep, n.half, n.full = self.side, self.castling, self.ep, self.half, self.full
        n.kings = self.kings[:]
        return n

    # ---------------------------------------------------------- attacks ---
    def is_attacked(self, sq, by):
        b = self.b
        for s in PAWN_FROM[ci(by)][sq]:
            if b[s] == by * PAWN:
                return True
        kn = by * KNIGHT
        for s in KNIGHT_T[sq]:
            if b[s] == kn:
                return True
        kg = by * KING
        for s in KING_T[sq]:
            if b[s] == kg:
                return True
        rq, qq, bq = by * ROOK, by * QUEEN, by * BISHOP
        rays = RAYS[sq]
        for d in range(4):
            for s in rays[d]:
                p = b[s]
                if p:
                    if p == rq or p == qq:
                        return True
                    break
        for d in range(4, 8):
            for s in rays[d]:
                p = b[s]
                if p:
                    if p == bq or p == qq:
                        return True
                    break
        return False

    def attackers_count(self, sq, by):
        """Number of pieces of side `by` attacking `sq` (x-rays ignored)."""
        b, n = self.b, 0
        for s in PAWN_FROM[ci(by)][sq]:
            if b[s] == by * PAWN:
                n += 1
        for s in KNIGHT_T[sq]:
            if b[s] == by * KNIGHT:
                n += 1
        for s in KING_T[sq]:
            if b[s] == by * KING:
                n += 1
        for d in range(8):
            ok = (by * ROOK, by * QUEEN) if d < 4 else (by * BISHOP, by * QUEEN)
            for s in RAYS[sq][d]:
                p = b[s]
                if p:
                    if p in ok:
                        n += 1
                    break
        return n

    def in_check(self, side=None):
        side = self.side if side is None else side
        return self.is_attacked(self.kings[side], -side)

    # ------------------------------------------------------- generation ---
    def legal_moves(self):
        b, us = self.b, self.side
        them = -us
        ksq = self.kings[us]
        # ---- checkers and pins
        checkers, block = [], None
        pinned = {}
        rq, qq, bq = them * ROOK, them * QUEEN, them * BISHOP
        for d in range(8):
            first = -1
            sliders = (rq, qq) if d < 4 else (bq, qq)
            for s in RAYS[ksq][d]:
                p = b[s]
                if p == 0:
                    continue
                if first < 0:
                    if p * us > 0:
                        first = s
                        continue
                    if p in sliders:
                        checkers.append(s)
                        block = set()
                        for t in RAYS[ksq][d]:
                            block.add(t)
                            if t == s:
                                break
                    break
                else:
                    if p in sliders:
                        pinned[first] = d
                    break
        kn = them * KNIGHT
        for s in KNIGHT_T[ksq]:
            if b[s] == kn:
                checkers.append(s)
                block = {s}
        pw = them * PAWN
        for s in PAWN_FROM[ci(them)][ksq]:
            if b[s] == pw:
                checkers.append(s)
                block = {s}
        nchk = len(checkers)

        moves = []
        # ---- king moves (king removed from board while testing)
        b[ksq] = 0
        for t in KING_T[ksq]:
            if b[t] * us <= 0 and not self.is_attacked(t, them):
                moves.append(ksq | (t << 6))
        b[ksq] = us * KING
        if nchk >= 2:
            return moves
        # ---- castling
        if nchk == 0 and self.castling:
            if us > 0:
                if self.castling & 1 and b[5] == 0 and b[6] == 0 and b[7] == ROOK and \
                        not self.is_attacked(5, them) and not self.is_attacked(6, them):
                    moves.append(make_move(4, 6, 0, FLAG_CASTLE))
                if self.castling & 2 and b[3] == 0 and b[2] == 0 and b[1] == 0 and b[0] == ROOK and \
                        not self.is_attacked(3, them) and not self.is_attacked(2, them):
                    moves.append(make_move(4, 2, 0, FLAG_CASTLE))
            else:
                if self.castling & 4 and b[61] == 0 and b[62] == 0 and b[63] == -ROOK and \
                        not self.is_attacked(61, them) and not self.is_attacked(62, them):
                    moves.append(make_move(60, 62, 0, FLAG_CASTLE))
                if self.castling & 8 and b[59] == 0 and b[58] == 0 and b[57] == 0 and b[56] == -ROOK and \
                        not self.is_attacked(59, them) and not self.is_attacked(58, them):
                    moves.append(make_move(60, 58, 0, FLAG_CASTLE))

        def ok(s, t):
            if block is not None and t not in block:
                return False
            if s in pinned and LINE_DIR[ksq][t] != pinned[s]:
                return False
            return True

        fwd = 8 * us
        start_rank = 1 if us > 0 else 6
        promo_rank = 7 if us > 0 else 0
        for s in range(64):
            p = b[s] * us
            if p <= 0 or p == KING:
                continue
            if p == PAWN:
                r = s // 8
                f = s % 8
                t = s + fwd
                if 0 <= t < 64 and b[t] == 0:
                    if ok(s, t):
                        if t // 8 == promo_rank:
                            for pr in (QUEEN, ROOK, BISHOP, KNIGHT):
                                moves.append(make_move(s, t, pr))
                        else:
                            moves.append(s | (t << 6))
                    t2 = t + fwd
                    if r == start_rank and b[t2] == 0 and ok(s, t2):
                        moves.append(make_move(s, t2, 0, FLAG_DOUBLE))
                for df in (-1, 1):
                    if not 0 <= f + df < 8:
                        continue
                    t = s + fwd + df
                    if not 0 <= t < 64:
                        continue
                    if b[t] * us < 0:
                        if ok(s, t):
                            if t // 8 == promo_rank:
                                for pr in (QUEEN, ROOK, BISHOP, KNIGHT):
                                    moves.append(make_move(s, t, pr))
                            else:
                                moves.append(s | (t << 6))
                    elif t == self.ep:
                        m = make_move(s, t, 0, FLAG_EP)
                        u = self.make(m)
                        legal = not self.is_attacked(ksq, them)
                        self.unmake(m, u)
                        if legal:
                            moves.append(m)
            elif p == KNIGHT:
                if s in pinned:
                    continue
                for t in KNIGHT_T[s]:
                    if b[t] * us <= 0 and ok(s, t):
                        moves.append(s | (t << 6))
            else:
                dirs = range(4) if p == ROOK else range(4, 8) if p == BISHOP else range(8)
                for d in dirs:
                    for t in RAYS[s][d]:
                        q = b[t]
                        if q * us > 0:
                            break
                        if ok(s, t):
                            moves.append(s | (t << 6))
                        if q:
                            break
        return moves

    def has_legal_move(self):
        # Simple version: reuse full generation (cost identical for budget accounting).
        return bool(self.legal_moves())

    # ------------------------------------------------------- make/unmake ---
    def make(self, m):
        b = self.b
        fr, to = m & 63, (m >> 6) & 63
        promo, flag = (m >> 12) & 7, m >> 15
        us = self.side
        p = b[fr]
        cap = b[to]
        undo = (cap, self.castling, self.ep, self.half)
        if flag == FLAG_EP:
            cs = to - 8 * us
            undo = (b[cs], self.castling, self.ep, self.half)
            b[cs] = 0
        b[to] = promo * us if promo else p
        b[fr] = 0
        if flag == FLAG_CASTLE:
            if to > fr:
                b[fr + 1], b[fr + 3] = b[fr + 3], 0
            else:
                b[fr - 1], b[fr - 4] = b[fr - 4], 0
        if p == us * KING:
            self.kings[us] = to
            self.castling &= ~(3 if us > 0 else 12)
        if self.castling:
            for sq in (fr, to):
                if sq == 0:
                    self.castling &= ~2
                elif sq == 7:
                    self.castling &= ~1
                elif sq == 56:
                    self.castling &= ~8
                elif sq == 63:
                    self.castling &= ~4
        self.ep = (fr + to) // 2 if flag == FLAG_DOUBLE else -1
        self.half = 0 if (cap or abs(p) == PAWN) else self.half + 1
        if us < 0:
            self.full += 1
        self.side = -us
        return undo

    def unmake(self, m, undo):
        b = self.b
        fr, to = m & 63, (m >> 6) & 63
        promo, flag = (m >> 12) & 7, m >> 15
        self.side = us = -self.side
        cap, self.castling, self.ep, self.half = undo
        if us < 0:
            self.full -= 1
        p = us * PAWN if promo else b[to]
        b[fr] = p
        if flag == FLAG_EP:
            b[to] = 0
            b[to - 8 * us] = cap
        else:
            b[to] = cap
        if flag == FLAG_CASTLE:
            if to > fr:
                b[fr + 3], b[fr + 1] = b[fr + 1], 0
            else:
                b[fr - 4], b[fr - 1] = b[fr - 1], 0
        if p == us * KING:
            self.kings[us] = fr

    # ------------------------------------------------------------ helpers ---
    def gives_check(self, m):
        u = self.make(m)
        c = self.is_attacked(self.kings[self.side], -self.side)
        self.unmake(m, u)
        return c

    def is_capture(self, m):
        return self.b[m_to(m)] != 0 or m_flag(m) == FLAG_EP

    def captured_type(self, m):
        if m_flag(m) == FLAG_EP:
            return PAWN
        return abs(self.b[m_to(m)])

    def piece_type(self, m):
        return abs(self.b[m_from(m)])

    def material(self, side):
        return sum(VALUE[abs(p)] for p in self.b if p * side > 0)

    def uci(self, m):
        s = SQUARE_NAMES[m_from(m)] + SQUARE_NAMES[m_to(m)]
        if m_promo(m):
            s += PIECE_CHARS[m_promo(m)]
        return s

    def parse_uci(self, u):
        for m in self.legal_moves():
            if self.uci(m) == u:
                return m
        raise ValueError(f"illegal move {u} in {self.fen()}")

    def san(self, m):
        """Short algebraic notation (good enough for reports)."""
        pt = self.piece_type(m)
        if m_flag(m) == FLAG_CASTLE:
            s = "O-O" if m_to(m) > m_from(m) else "O-O-O"
        else:
            cap = self.is_capture(m)
            to = SQUARE_NAMES[m_to(m)]
            if pt == PAWN:
                s = (SQUARE_NAMES[m_from(m)][0] + "x" if cap else "") + to
                if m_promo(m):
                    s += "=" + PIECE_CHARS[m_promo(m)].upper()
            else:
                s = PIECE_CHARS[pt].upper()
                others = [o for o in self.legal_moves() if o != m and m_to(o) == m_to(m)
                          and self.piece_type(o) == pt]
                if others:
                    fr = SQUARE_NAMES[m_from(m)]
                    if all(SQUARE_NAMES[m_from(o)][0] != fr[0] for o in others):
                        s += fr[0]
                    elif all(SQUARE_NAMES[m_from(o)][1] != fr[1] for o in others):
                        s += fr[1]
                    else:
                        s += fr
                s += ("x" if cap else "") + to
        u = self.make(m)
        if self.in_check():
            s += "#" if not self.legal_moves() else "+"
        self.unmake(m, u)
        return s

    def __str__(self):
        rows = []
        for r in range(7, -1, -1):
            row = []
            for f in range(8):
                p = self.b[r * 8 + f]
                ch = PIECE_CHARS[abs(p)] if p else "."
                row.append(ch.upper() if p > 0 else ch)
            rows.append(f"{r + 1} " + " ".join(row))
        rows.append("  a b c d e f g h")
        return "\n".join(rows)


def perft(board, depth):
    if depth == 0:
        return 1
    moves = board.legal_moves()
    if depth == 1:
        return len(moves)
    n = 0
    for m in moves:
        u = board.make(m)
        n += perft(board, depth - 1)
        board.unmake(m, u)
    return n
