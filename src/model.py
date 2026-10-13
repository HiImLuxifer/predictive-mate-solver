"""Step 3-4: training targets from the teacher and the predictor P_theta(s, a).

Target meaning
--------------
attacker move a at an OR node with d plies left:
    P*(s, a) = 1  iff a starts a forced mate within d plies     (teacher, exhaustive)
defender reply b at an AND node:
    P*(s, b) = 1  iff after b the attacker still mates in time  (b does NOT escape)

So P_theta is always "probability that the attacker is still winning after this
move". Search uses it as: OR nodes try high P first (most likely mating move),
AND nodes try low P first (most likely escape -> fastest refutation).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # sklearn HGB: thread spin-up dominates tiny batches
import hashlib
import json
import random
from multiprocessing import Pool

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

from minichess import Board
from teacher import Teacher
from board_utils import features_for_node, FEATURE_NAMES, N_FEATURES


# ------------------------------------------------------ training targets ---
def collect_examples(puzzle, seed=0):
    """Labelled (s, a) pairs from one puzzle: root + interior nodes.

    node types: 'root' (OR, d=2N-1), 'and' (defender replies after a root move),
                'or_int' (attacker moves at an interior OR node, d=2N-3).
    """
    rng = random.Random(f"{puzzle['fen']}{seed}")
    t = Teacher()
    b = Board(puzzle["fen"])
    att = b.side
    d = 2 * puzzle["n"] - 1
    rows, ys, types = [], [], []

    def add(board, d_node, is_att, typ):
        lab = t.label_moves(board, d_node, is_att)
        if not lab:
            return []
        X = features_for_node(board, [m for m, _ in lab], att, d_node)
        rows.append(X)
        ys.extend(y for _, y in lab)
        types.extend([typ] * len(lab))
        return lab

    root = add(b, d, True, "root")
    if d >= 3:
        keys = [m for m, y in root if y]
        wrong = [m for m, y in root if not y]
        hard = [m for m in wrong if b.gives_check(m) or b.is_capture(m)]
        picks = keys + rng.sample(hard or wrong, min(2, len(hard or wrong)))
        for i, m in enumerate(picks):
            u = b.make(m)
            replies = add(b, d - 1, False, "and")
            if replies and (i < len(keys) or i == len(keys)):      # interior OR nodes
                for rm, _ in rng.sample(replies, min(2 if i < len(keys) else 1, len(replies))):
                    u2 = b.make(rm)
                    add(b, d - 2, True, "or_int")
                    b.unmake(rm, u2)
            b.unmake(m, u)
    X = np.vstack(rows) if rows else np.zeros((0, N_FEATURES), np.float32)
    return X, np.array(ys, np.int8), types, puzzle["id"]


def build_dataset(puzzles, workers=2):
    with Pool(workers) as pool:
        res = pool.map(collect_examples, puzzles, chunksize=8)
    X = np.vstack([r[0] for r in res])
    y = np.concatenate([r[1] for r in res])
    types = np.array(sum([r[2] for r in res], []))
    pid = np.array(sum([[r[3]] * len(r[1]) for r in res], []))
    return X, y, types, pid


# --------------------------------------------------------------- models ---
def train_gbm(X, y, seed=0):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                       l2_regularization=1.0, early_stopping=True,
                                       validation_fraction=0.1, random_state=seed)
    m.fit(X, y)
    return m


def train_logreg(X, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m.fit(X, y)
    return m


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)


def pair_metrics(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "ece": ece(p, y), "n": int(len(y)), "pos_rate": float(y.mean())}


# ------------------------------------------------------------ predictor ---
class Predictor:
    """Wraps the learned model with a per-position cache.

    The cache only avoids recomputing the SAME node twice across the systems
    and budgets run on one puzzle; it is cleared between puzzles.
    noise_sigma > 0 adds deterministic Gaussian noise to the logits (ablation:
    'the predictor becomes less accurate').
    """

    def __init__(self, model, noise_sigma=0.0):
        self.model = model
        self.noise = noise_sigma
        self._p, self._x = {}, {}
        self.n_eval = 0

    def clear(self):
        self._p.clear()
        self._x.clear()

    @staticmethod
    def col(name):
        return FEATURE_NAMES.index(name)

    def features(self, board, moves, attacker, d_node):
        k = (board.key(), d_node, attacker, tuple(moves))
        X = self._x.get(k)
        if X is None:
            X = self._x[k] = features_for_node(board, moves, attacker, d_node)
        return X

    def predict(self, board, moves, attacker, d_node):
        k = (board.key(), d_node, attacker, tuple(moves))
        p = self._p.get(k)
        if p is None:
            X = self.features(board, moves, attacker, d_node)
            self.n_eval += len(moves)
            p = self.model.predict_proba(X)[:, 1]
            if self.noise > 0:
                z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
                h = int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)
                z = z + np.random.default_rng(h).normal(0, self.noise, len(z))
                p = 1 / (1 + np.exp(-z))
            self._p[k] = p
        return p


def root_candidates(board, n):
    moves = board.legal_moves()
    if n == 1:
        moves = [m for m in moves if board.gives_check(m)]
    return moves


def calibrate_temperature(predictor, puzzles, grid=None):
    """T minimising NLL of the winning move set under pi = softmax(logit/T)."""
    from search import softmax_logits
    grid = grid or [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    cache = []
    for pz in puzzles:
        b = Board(pz["fen"])
        moves = root_candidates(b, pz["n"])
        p = predictor.predict(b, moves, b.side, 2 * pz["n"] - 1)
        win = np.array([b.uci(m) in pz["keys"] for m in moves])
        cache.append((p, win))
        predictor.clear()
    res = {}
    for T in grid:
        res[T] = float(np.mean([-np.log(max(softmax_logits(p, T)[w].sum(), 1e-9)) for p, w in cache]))
    best = min(res, key=res.get)
    return best, res


def root_metrics(predictor, puzzles):
    """Decision-level accuracy of the predictor alone (no search)."""
    top1, top3, ranks = [], [], []
    for pz in puzzles:
        b = Board(pz["fen"])
        moves = root_candidates(b, pz["n"])
        p = predictor.predict(b, moves, b.side, 2 * pz["n"] - 1)
        order = np.argsort(-p, kind="stable")
        names = [b.uci(moves[i]) for i in order]
        r = min(names.index(k) for k in pz["keys"]) + 1
        ranks.append(r)
        top1.append(r == 1)
        top3.append(r <= 3)
        predictor.clear()
    return {"top1": float(np.mean(top1)), "top3": float(np.mean(top3)),
            "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)), "n": len(puzzles)}
