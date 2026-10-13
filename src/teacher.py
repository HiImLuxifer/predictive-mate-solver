"""Teacher: exhaustive AND/OR mate solver (no budget).

Used OFFLINE only: (1) to certify puzzles, (2) to produce training targets
P*(s, a), (3) as ground truth for the test set. Its computation is never
counted in the online budget of systems A, B, C.

Conventions
-----------
attacker = side trying to mate.  d = plies the attacker still has.
mates(board, d)          attacker to move: can attacker force mate within d plies?
defender_fails(board, d) defender to move: does every reply lose to mate within d?
"""
from minichess import Board, VALUE


def _order(board, moves):
    """Cheap ordering for the teacher (checks, captures first) - only speed."""
    def k(m):
        return (not board.gives_check(m), -VALUE.get(board.captured_type(m), 0) if board.is_capture(m) else 1)
    return sorted(moves, key=k)


class Teacher:
    def __init__(self):
        self.tt = {}
        self.nodes = 0

    def mates(self, board, d):
        if d <= 0:
            return False
        key = (board.key(), d, 1)
        r = self.tt.get(key)
        if r is not None:
            return r
        self.nodes += 1
        moves = board.legal_moves()
        if d == 1:
            moves = [m for m in moves if board.gives_check(m)]
        else:
            moves = _order(board, moves)
        res = False
        for m in moves:
            u = board.make(m)
            ok = self.defender_fails(board, d - 1)
            board.unmake(m, u)
            if ok:
                res = True
                break
        self.tt[key] = res
        return res

    def defender_fails(self, board, d):
        key = (board.key(), d, 0)
        r = self.tt.get(key)
        if r is not None:
            return r
        self.nodes += 1
        moves = board.legal_moves()
        if not moves:
            res = board.in_check()          # checkmate -> attacker succeeded; stalemate -> failed
        elif d <= 0:
            res = False
        else:
            res = True
            # king moves and captures first: they are the most frequent escapes
            moves.sort(key=lambda m: (board.piece_type(m) != 6, not board.is_capture(m)))
            for m in moves:
                u = board.make(m)
                ok = self.mates(board, d - 1)
                board.unmake(m, u)
                if not ok:
                    res = False
                    break
        self.tt[key] = res
        return res

    # ------------------------------------------------------------ labels ---
    def label_moves(self, board, d, attacker_to_move):
        """Exhaustive label of EVERY candidate move at a node.

        attacker node: y = 1  iff move a starts a forced mate within d plies
        defender node: y = 1  iff reply b still loses to mate (i.e. b does NOT escape)
        Returns list of (move, y). At attacker nodes with d == 1 only checks are
        candidates (a non-check can never mate in one ply), exactly as in search.
        """
        moves = board.legal_moves()
        if attacker_to_move and d == 1:
            moves = [m for m in moves if board.gives_check(m)]
        out = []
        for m in moves:
            u = board.make(m)
            if attacker_to_move:
                y = self.defender_fails(board, d - 1)
            else:
                y = self.mates(board, d - 1)
            board.unmake(m, u)
            out.append((m, int(y)))
        return out

    def mate_length(self, board, max_n):
        """Smallest n <= max_n such that attacker mates in n moves, else None."""
        for n in range(1, max_n + 1):
            if self.mates(board, 2 * n - 1):
                return n
        return None
