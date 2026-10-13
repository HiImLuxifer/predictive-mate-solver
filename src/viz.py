"""Board diagrams and per-puzzle search traces for the report."""
import numpy as np

from minichess import Board, NAME_TO_SQ
from analysis import COL, INK, INK2, GRID, style_ax

GLYPH = {1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔"}
GLYPH_B = {1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚"}


def draw_board(ax, fen, arrow=None, title=None):
    b = Board(fen)
    light, dark = "#f0ede4", "#b9b4a6"
    flip = b.side < 0                     # attacker at the bottom
    for r in range(8):
        for f in range(8):
            rr, ff = (7 - r, 7 - f) if flip else (r, f)
            ax.add_patch(__import__("matplotlib").patches.Rectangle((f, r), 1, 1, color=light if (rr + ff) % 2 else dark, lw=0))
            p = b.b[rr * 8 + ff]
            if p:
                ax.text(f + 0.5, r + 0.47, (GLYPH if p > 0 else GLYPH_B)[abs(p)], ha="center", va="center",
                        fontsize=20, family="DejaVu Sans", color=INK)
    if arrow:
        a, c = NAME_TO_SQ[arrow[:2]], NAME_TO_SQ[arrow[2:4]]

        def xy(s):
            f, r = s % 8, s // 8
            return ((7 - f) if flip else f) + 0.5, ((7 - r) if flip else r) + 0.5
        (x0, y0), (x1, y1) = xy(a), xy(c)
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=COL["C"], lw=3, mutation_scale=18))
    files = "abcdefgh"[::-1] if flip else "abcdefgh"
    ax.set_xticks(np.arange(8) + 0.5, list(files))
    ax.set_yticks(np.arange(8) + 0.5, [str(8 - i) if flip else str(i + 1) for i in range(8)])
    ax.tick_params(length=0, labelsize=8, colors=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def plot_trace(ax, trace, budget):
    """rho_t and number of refuted root moves along the rounds of system C."""
    if not trace:
        return
    u = [t["used"] for t in trace]
    ax.step(u, [t["rho"] for t in trace], where="post", color=COL["C"], lw=2, label="ρ_t (residual mass)")
    ax.step(u, [t["margin"] for t in trace], where="post", color=COL["B"], lw=2, label="margin top1−top2")
    ax.set_xlim(0, budget)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("expansions used", fontsize=8, color=INK2)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def plot_allocation(ax, alloc_b, alloc_c, keys, top=6):
    """Expansions spent below each root move by B and C (key moves starred)."""
    moves = sorted(set(alloc_b) | set(alloc_c), key=lambda m: -(alloc_b.get(m, 0) + alloc_c.get(m, 0)))[:top]
    for k in keys:
        if k not in moves and (k in alloc_b or k in alloc_c):
            moves[-1] = k
    y = np.arange(len(moves))
    h = 0.38
    ax.barh(y - h / 2, [alloc_b.get(m, 0) for m in moves], height=h - 0.04, color=COL["B"], label="B")
    ax.barh(y + h / 2, [alloc_c.get(m, 0) for m in moves], height=h - 0.04, color=COL["C"], label="C")
    ax.set_yticks(y, [("★ " if m in keys else "") + m for m in moves])
    ax.invert_yaxis()
    ax.set_xlabel("espansioni spese sotto la mossa", fontsize=8, color=INK2)
    style_ax(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
