"""Budgeted AND/OR mate search with an explicit, resumable tree.

What is searched
----------------
The decision is the attacker's FIRST move in a "mate in N" problem.
The tree alternates OR nodes (attacker to move: one good move suffices) and
AND nodes (defender to move: every reply must be refuted).

State      : (position, d) where d = plies the attacker still has.
Expansion  : one call to legal-move generation for a node (the budget unit).
Horizon    : the leaves of the explicit tree = generated but not yet expanded
             nodes, plus depth-limit leaves (d exhausted).
Active     : horizon nodes still eligible for computation, i.e. not inside a
frontier     resolved (PROVEN / DISPROVEN) subtree and not below a root move
             that the regulator has deferred.

Alternatives leaving the active frontier are either
  * resolved   : PROVEN or DISPROVEN, the result is certain (sound solver);
  * cut        : siblings below a resolved node (an OR node proven by one child,
                 an AND node refuted by one reply) - irrelevant to the decision;
  * deferred   : (system C only) unresolved root moves the regulator keeps out
                 of the candidate width. These are NOT resolved - their
                 probability mass is the residual rho_t.

Systems
-------
A  heuristic ordering (checks, captures MVV-LVA), plain depth-first allocation
B  ordering from the predictor P_theta, plain depth-first allocation
C  same ordering as B + regulator at the root (width, share, early stop)
"""
import math

import numpy as np

from board_utils import heuristic_order

UNK, PROVEN, DISPROVEN = 0, 1, 2


class Node:
    __slots__ = ("d", "is_or", "status", "moves", "children", "best")

    def __init__(self, d, is_or):
        self.d, self.is_or = d, is_or
        self.status = UNK
        self.moves = None          # None = on the horizon (not expanded yet)
        self.children = None
        self.best = None


# ------------------------------------------------------------- policies ---
class HeuristicPolicy:
    name = "heuristic"

    def order(self, board, moves, node, attacker):
        return heuristic_order(board, moves)


class ModelPolicy:
    """OR nodes: highest P_theta first. AND nodes: most likely escape first."""
    name = "model"

    def __init__(self, predictor):
        self.pred = predictor

    def order(self, board, moves, node, attacker):
        p = self.pred.predict(board, moves, attacker, node.d)
        idx = np.argsort(-p if node.is_or else p, kind="stable")
        return [moves[i] for i in idx]


# --------------------------------------------------------------- engine ---
class Searcher:
    def __init__(self, policy, budget):
        self.policy = policy
        self.budget = budget
        self.used = 0

    def expand(self, node, board):
        self.used += 1
        moves = board.legal_moves()
        if not moves:
            # OR: attacker mated/stalemated -> failure. AND: mate -> success, stalemate -> failure
            node.status = DISPROVEN if node.is_or else (PROVEN if board.in_check() else DISPROVEN)
            return
        if not node.is_or and node.d == 0:
            node.status = DISPROVEN            # defender survives, attacker out of plies
            return
        if node.is_or and node.d == 1:
            moves = [m for m in moves if board.gives_check(m)]   # only checks can mate in 1 ply
            if not moves:
                node.status = DISPROVEN
                return
        node.moves = self.policy.order(board, moves, node, self.attacker)
        node.children = [None] * len(node.moves)

    def run(self, node, board, limit):
        """Depth-first proof search below `node`, resumable, until `limit` expansions."""
        if node.status != UNK:
            return node.status
        if node.moves is None:
            if self.used >= limit:
                return UNK
            self.expand(node, board)
            if node.status != UNK:
                return node.status
        all_resolved = True
        for i, m in enumerate(node.moves):
            c = node.children[i]
            if c is None:
                c = node.children[i] = Node(node.d - 1, not node.is_or)
            if c.status == UNK:
                u = board.make(m)
                s = self.run(c, board, limit)
                board.unmake(m, u)
            else:
                s = c.status
            if node.is_or and s == PROVEN:
                node.status, node.best = PROVEN, m
                return PROVEN
            if not node.is_or and s == DISPROVEN:
                node.status = DISPROVEN
                return DISPROVEN
            if s == UNK:
                all_resolved = False
                if self.used >= limit:
                    return UNK
        if all_resolved:
            node.status = DISPROVEN if node.is_or else PROVEN
            return node.status
        return UNK

    # ------------------------------------------------------- root API ---
    def solve(self, board, n):
        """Plain depth-first allocation (systems A and B)."""
        self.attacker = board.side
        self.root = Node(2 * n - 1, True)
        self.run(self.root, board, self.budget)
        return self.decision()

    def root_status(self):
        """{move: status, touched} for every root candidate."""
        r = self.root
        out = {}
        if r.moves is None:
            return out
        for m, c in zip(r.moves, r.children):
            out[m] = (c.status if c else UNK, bool(c is not None and (c.moves is not None or c.status != UNK)))
        return out

    def decision(self):
        r = self.root
        if r.status == PROVEN:
            return r.best, True
        if r.moves is None:
            return None, False
        for m, c in zip(r.moves, r.children):          # first non-refuted in policy order
            if c is None or c.status != DISPROVEN:
                return m, False
        return r.moves[0], False


def frontier_stats(searcher):
    """Count horizon / active frontier of the explicit tree at stop time.

    horizon  = generated but not expanded nodes (incl. alternatives generated by
               an expansion but never instantiated)
    active   = horizon nodes with no resolved ancestor and not below a root move
               the regulator deferred (searcher.deferred, system C only)
    closed   = horizon nodes below a resolved node (cut: no longer relevant)
    deferred = horizon nodes below a deferred root move (unresolved, rho)
    """
    deferred_moves = getattr(searcher, "deferred", set())
    st = {"expanded": 0, "horizon": 0, "active": 0, "closed": 0, "deferred": 0}

    def leaf(closed, dfr):
        st["horizon"] += 1
        st["closed" if closed else "deferred" if dfr else "active"] += 1

    def visit(node, closed, dfr, is_root=False):
        st["expanded"] += 1
        if node.moves is None:
            return
        closed = closed or node.status != UNK
        for m, c in zip(node.moves, node.children):
            d2 = dfr or (is_root and m in deferred_moves)
            if c is None or (c.moves is None and c.status == UNK):
                leaf(closed, d2)
            elif c.moves is None:          # expanded and immediately resolved (terminal)
                st["expanded"] += 1
            else:
                visit(c, closed, d2)

    if searcher.root is not None and (searcher.root.moves is not None or searcher.root.status != UNK):
        visit(searcher.root, False, False, True)
    return st


def subtree_expansions(node):
    """Number of expansions spent inside the subtree rooted at node."""
    if node is None or (node.moves is None and node.status == UNK):
        return 0
    if node.moves is None:
        return 1
    return 1 + sum(subtree_expansions(c) for c in node.children)


def root_allocation(searcher):
    """{move: expansions spent below that root move} - where computation went."""
    r = searcher.root
    if r is None or r.moves is None:
        return {}
    return {m: subtree_expansions(c) for m, c in zip(r.moves, r.children)}


def softmax_logits(p, T):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p)) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_norm(pi):
    pi = pi[pi > 0]
    if len(pi) <= 1:
        return 0.0
    return float(-(pi * np.log(pi)).sum() / math.log(len(pi)))
