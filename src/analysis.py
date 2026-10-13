"""Tables, statistics and figures for the report / notebook."""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# validated categorical slots (dataviz reference palette, first three = all-pairs safe)
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SYSTEM_LABEL = {"A": "A  baseline (scacchi/catture)", "B": "B  predittivo", "C": "C  predittivo + regolato"}


def load_runs(tag="main"):
    return pd.read_json(os.path.join(RES, f"runs_{tag}.jsonl"), lines=True)


def load_info(tag="main"):
    return pd.read_json(os.path.join(RES, f"info_{tag}.jsonl"), lines=True)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def summary(df):
    rows = []
    for (n, b, s), g in df.groupby(["n", "budget", "system"]):
        lo, hi = boot_ci(g["correct"])
        rows.append({"N": n, "budget": b, "system": s, "solve": g["correct"].mean(), "solve_lo": lo, "solve_hi": hi,
                     "proven": g["proven"].mean(), "used": g["used"].mean(), "rho": g["rho"].mean(),
                     "rho_untouched": g["rho_untouched"].mean(), "ms": 1000 * g["time"].mean(), "puzzles": len(g)})
    return pd.DataFrame(rows)


def paired_gain(df, s1, s2, metric="correct"):
    """Gain of s2 over s1 per (N, budget) with exact McNemar test on discordant pairs."""
    out = []
    for (n, b), g in df.groupby(["n", "budget"]):
        a = g[g.system == s1].set_index("id")[metric]
        c = g[g.system == s2].set_index("id")[metric]
        ids = a.index.intersection(c.index)
        a, c = a[ids].astype(bool), c[ids].astype(bool)
        n01, n10 = int((~a & c).sum()), int((a & ~c).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        out.append({"N": n, "budget": b, "comparison": f"{s2} - {s1}", "gain": c.mean() - a.mean(),
                    f"only_{s2}": n01, f"only_{s1}": n10, "p_mcnemar": p})
    return pd.DataFrame(out)


def efficiency(df, systems=("A", "B", "C")):
    """Expansions used on puzzles that BOTH (all) systems proved: cost of a certain answer."""
    out = []
    for (n, b), g in df.groupby(["n", "budget"]):
        piv = g[g.system.isin(systems)].pivot_table(index="id", columns="system", values="proven")
        ok = piv.index[piv.all(axis=1)]
        u = g[g.id.isin(ok)].groupby("system")["used"].median()
        out.append({"N": n, "budget": b, "both_proven": len(ok), **{f"median_used_{s}": u.get(s, np.nan) for s in systems}})
    return pd.DataFrame(out)


def style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_solve_curves(summ, systems=("A", "B", "C"), metric="solve", fname="solve_vs_budget.png"):
    import matplotlib.pyplot as plt
    Ns = sorted(summ.N.unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.2 * len(Ns), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, Ns):
        d = summ[summ.N == n]
        budgets = sorted(d.budget.unique())
        xs = np.arange(len(budgets))
        for s in systems:
            e = d[d.system == s].set_index("budget").loc[budgets]
            ax.plot(xs, e[metric], color=COL[s], lw=2, marker="o", ms=6, label=SYSTEM_LABEL[s])
            if metric == "solve":
                ax.fill_between(xs, e["solve_lo"], e["solve_hi"], color=COL[s], alpha=0.12, lw=0)
            ax.annotate(f"{e[metric].iloc[-1]:.0%}", (xs[-1], e[metric].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_xticks(xs, [str(b) for b in budgets])
        ax.set_title(f"Matto in {n}", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("budget (espansioni per puzzle)", fontsize=9, color=INK2)
        ax.set_ylim(0, 1.04)
        style_ax(ax)
    axes[0].set_ylabel({"solve": "prima mossa corretta", "proven": "corretta e dimostrata"}[metric], fontsize=9, color=INK2)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    return fig


# --------------------------------------------------------- residuals ---
def residual_analysis(model, T, X, y, types, pid):
    """|r| = |P* - P_theta| on root nodes; does the R_hat proxy flag large errors?"""
    from search import softmax_logits, entropy_norm
    from board_utils import FEATURE_NAMES
    qi, si = FEATURE_NAMES.index("is_quiet"), FEATURE_NAMES.index("is_sacrifice")
    s = types == "root"
    X, y, pid = X[s], y[s], pid[s]
    p = model.predict_proba(X)[:, 1]
    r = np.abs(y - p)
    Rh = np.zeros_like(p)
    Hs = {}
    for g in np.unique(pid):
        m = pid == g
        pi = softmax_logits(p[m], T)
        H = entropy_norm(pi)
        Hs[g] = H
        w = np.where((X[m, qi] > 0) | (X[m, si] > 0), 1.0, 0.5)
        Rh[m] = H * w
    big = r > 0.5
    res = {
        "pairs": int(len(r)), "mean_abs_residual": float(r.mean()),
        "share_big_errors": float(big.mean()),
        "big_errors_are_false_negatives": float(((y == 1) & big).sum() / max(big.sum(), 1)),
        "auc_Rhat_flags_big_error": float(roc_auc_score(big, Rh)) if 0 < big.sum() < len(big) else float("nan"),
        "auc_Rhat_flags_missed_key": float(roc_auc_score(big[y == 1], Rh[y == 1])) if 0 < big[y == 1].sum() < (y == 1).sum() else float("nan"),
        "auc_lowP_flags_big_error": float(roc_auc_score(big, 1 - p)),
    }
    return res, pd.DataFrame({"pid": pid, "y": y, "p": p, "abs_r": r, "R_hat": Rh})


def stop_danger(info):
    """Is stopping at the prediction (budget 0) dangerous where the entropy is high?"""
    wrong = info["key_rank"] > 1
    out = {"puzzles": int(len(info)), "prediction_only_error": float(wrong.mean())}
    for col in ("H", "pi_top"):
        score = info[col] if col == "H" else 1 - info[col]
        out[f"auc_{col}_flags_wrong_prediction"] = float(roc_auc_score(wrong, score)) if 0 < wrong.sum() < len(wrong) else float("nan")
    return out
