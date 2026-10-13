"""System C: predictive search + regulator at the root decision point.

    O_t  ->  R  ->  U_t

O_t (observations, recomputed after every allocation round)
    rho_t    : residual probability mass = sum of pi(a|s) over root moves that are
               still UNRESOLVED and are not the current choice a*_t
    margin_t : pi~(top1) - pi~(top2), pi~ = pi renormalised over non-refuted moves
    R_hat(a) : "needs more thinking" proxy = H_norm(pi~) * w(a),
               w(a) = 1 for quiet moves / sacrifices, 0.5 otherwise

U_t (computational decisions)
    1. width + share : which root moves receive computation in the next round
       (smallest prefix of pi covering 1-eps of the non-refuted mass, eps smaller
       when the margin is small, plus up to k_R deferred moves admitted because
       R_hat >= theta_R) and how much each gets (slice ~ pi(a) * (1 + lam*R_hat(a)),
       round total S_r = max(c_min, budget*s0_frac) * 2^r; every slice is at
       least c_min = c_min_mult*(2N+1) expansions, i.e. enough to reach the
       horizon once; a round stops serving candidates once it has spent S_r).
    2. stopping      : stop when a move is proven, or when rho_t < tau and the
       margin is wide (the chosen move must already have been searched).

Moves outside the width are *deferred*, not pruned: they re-enter as soon as the
mass in front of them is refuted (because the width is computed on non-refuted
moves). Only the root is regulated; below the root the search is the same
depth-first proof search with predictive ordering as system B.
"""
from dataclasses import dataclass, field

import numpy as np

from search import Node, UNK, PROVEN, DISPROVEN, softmax_logits, entropy_norm


@dataclass
class RegParams:
    T: float = 1.0               # softmax temperature (calibrated on validation)
    eps0: float = 0.10           # width: mass left out when the margin is wide
    eps_wide: float = 0.02       # width: mass left out when the margin is small
    m_wide: float = 0.5          # margin threshold for the width rule
    tau: float = 0.02            # early-stop threshold on rho_t
    m_stop: float = 0.8          # early-stop margin requirement
    lam: float = 1.0             # weight of R_hat in the share
    theta_R: float = 0.35        # admission threshold on R_hat
    k_R: int = 2                 # max deferred moves admitted per round by R_hat
    s0_frac: float = 1 / 32      # round-0 total slice as a fraction of the budget (doubles every round)
    c_min_mult: float = 1.0      # minimum slice = c_min_mult * (2N+1) expansions
    use_R: bool = True           # ablation 3: remove R_hat
    flatten: bool = False        # ablation 2: pi uniform (T -> infinity)
    early_stop: bool = True


def solve_regulated(searcher, board, n, predictor, P: RegParams, trace=None):
    searcher.attacker = att = board.side
    root = searcher.root = Node(2 * n - 1, True)
    searcher.expand(root, board)
    if root.status != UNK:
        return searcher.decision()
    moves = root.moves
    k = len(moves)
    root.children = [None] * k
    p = predictor.predict(board, moves, att, root.d)
    pi = np.full(k, 1.0 / k) if P.flatten else softmax_logits(p, P.T)
    X = predictor.features(board, moves, att, root.d)
    w = np.where((X[:, predictor.col("is_quiet")] > 0) | (X[:, predictor.col("is_sacrifice")] > 0), 1.0, 0.5)
    r = 0
    while True:
        st = np.array([c.status if c else UNK for c in root.children])
        if (st == PROVEN).any():
            i = int(np.argmax(st == PROVEN))
            root.status, root.best = PROVEN, moves[i]
            break
        nonref = np.where(st != DISPROVEN)[0]
        if len(nonref) == 0:
            root.status = DISPROVEN
            break
        pin = pi[nonref] / pi[nonref].sum()
        order = nonref[np.argsort(-pi[nonref], kind="stable")]
        a_star = order[0]
        rho = float(pi[order[1:]].sum())
        srt = np.sort(pin)[::-1]
        margin = float(srt[0] - (srt[1] if len(srt) > 1 else 0.0))
        H = entropy_norm(pin)
        R = H * w if P.use_R else np.zeros(k)
        touched = root.children[a_star] is not None and root.children[a_star].moves is not None
        if trace is not None:
            trace.append({"round": r, "used": searcher.used, "rho": rho, "margin": margin, "H": H,
                          "n_refuted": int(k - len(nonref)), "a_star": int(a_star)})
        if P.early_stop and touched and rho < P.tau and margin >= P.m_stop:
            break
        if searcher.used >= searcher.budget:
            break
        # ---- width
        eps = P.eps0 if margin >= P.m_wide else P.eps_wide
        cum = np.cumsum(pi[order])
        n_w = int(np.searchsorted(cum, (1 - eps) * pi[nonref].sum()) + 1)
        width = list(order[:n_w])
        if P.use_R:
            deferred = [i for i in order[n_w:] if R[i] >= P.theta_R]
            width += deferred[:P.k_R]
        # ---- shares
        share = np.array([pi[i] * (1 + P.lam * R[i]) for i in width])
        share /= share.sum()
        # granularity: a slice must be able to reach the horizon once (2N+1
        # expansions), otherwise spreading resolves nothing; a round never spends
        # more than S, so a wide candidate set is served in share order and the
        # tail waits for later (doubled) rounds.
        c_min = P.c_min_mult * (2 * n + 1)
        S = max(c_min, searcher.budget * P.s0_frac) * (2 ** r)
        searcher.deferred = {moves[i] for i in nonref if i not in set(width)}
        if trace is not None:
            trace[-1]["width"] = len(width)
            trace[-1]["deferred"] = len(searcher.deferred)
        spent = 0
        for j in np.argsort(-share, kind="stable"):
            if spent >= S:
                break
            i = width[j]
            sl = max(c_min, int(round(S * share[j])))
            limit = min(searcher.budget, searcher.used + sl)
            before = searcher.used
            c = root.children[i]
            if c is None:
                c = root.children[i] = Node(root.d - 1, False)
            if c.status == UNK:
                u = board.make(moves[i])
                searcher.run(c, board, limit)
                board.unmake(moves[i], u)
            spent += searcher.used - before
            if c.status == PROVEN or searcher.used >= searcher.budget:
                break
        r += 1
    if root.status == PROVEN:
        return root.best, True
    st = np.array([c.status if c else UNK for c in root.children])
    nonref = np.where(st != DISPROVEN)[0]
    if len(nonref) == 0:
        return moves[0], False
    return moves[nonref[np.argmax(pi[nonref])]], False
