"""Features, move classification and the classic move-ordering heuristic.

All features are computed in a fixed *mating frame*: the attacker is the side
that must deliver mate, the defender king is the target. This lets one model
score both attacker moves (OR nodes) and defender replies (AND nodes).

Cost note: features use only make/unmake and square-attack tests, never a full
legal-move generation of the child. So computing them is NOT an expansion, in
the same way as the gives_check() test used by the baseline heuristic.
"""
import numpy as np
from minichess import KING_T, VALUE, KING, m_to, m_promo

FEATURE_NAMES = [
    "is_attacker_move", "d_after", "gives_check", "is_capture", "captured_value", "is_promotion",
    "is_quiet", "pt_pawn", "pt_knight", "pt_bishop", "pt_rook", "pt_queen", "pt_king", "moved_value",
    "dk_escapes_before", "dk_escapes_after", "dk_escapes_delta", "dk_zone_attacked_after",
    "to_attacked_by_opponent", "to_defended_by_own", "is_sacrifice", "dist_to_dk",
    "material_balance", "n_candidate_moves", "mover_in_check", "dk_on_edge_after",
    "n_attacker_pieces", "n_defender_pieces",
]
N_FEATURES = len(FEATURE_NAMES)


def king_zone(board, defender):
    """(escape squares, attacked zone squares, in_check) for the defender king."""
    ksq = board.kings[defender]
    att = -defender
    b = board.b
    b[ksq] = 0                      # remove king so sliders 'see through' it
    esc = zatt = 0
    for t in KING_T[ksq]:
        a = board.is_attacked(t, att)
        zatt += a
        if not a and b[t] * defender <= 0:
            esc += 1
    b[ksq] = defender * KING
    chk = board.is_attacked(ksq, att)
    return esc, zatt + chk, chk


def node_context(board, attacker, n_moves):
    defender = -attacker
    esc, _, chk = king_zone(board, defender)
    na = sum(1 for p in board.b if p * attacker > 0)
    nd = sum(1 for p in board.b if p * defender > 0)
    return {
        "esc_before": esc,
        "mat": board.material(attacker) - board.material(defender),
        "n_moves": n_moves,
        "mover_in_check": int(board.in_check()),
        "na": na, "nd": nd,
    }


def move_features(board, m, attacker, d_after, ctx):
    """Feature vector for candidate move m at the current node."""
    mover = board.side
    defender = -attacker
    pt = board.piece_type(m)
    cap = board.is_capture(m)
    cap_val = VALUE.get(board.captured_type(m), 0) if cap else 0
    promo = m_promo(m) != 0
    moved_val = VALUE[pt] if pt != KING else 0
    to = m_to(m)
    u = board.make(m)
    esc, zatt, chk = king_zone(board, defender)
    opp_att = board.is_attacked(to, -mover)
    own_def = board.attackers_count(to, mover)
    dk = board.kings[defender]
    board.unmake(m, u)
    gives_check = chk if mover == attacker else 0
    sac = int(opp_att and pt != KING and (moved_val - cap_val) >= 2)
    dist = max(abs(to % 8 - dk % 8), abs(to // 8 - dk // 8))
    edge = int(dk % 8 in (0, 7) or dk // 8 in (0, 7))
    one_hot = [int(pt == k) for k in range(1, 7)]
    return [
        int(mover == attacker), d_after, int(gives_check), int(cap), cap_val, int(promo),
        int(not gives_check and not cap and not promo), *one_hot, moved_val,
        ctx["esc_before"], esc, esc - ctx["esc_before"], zatt,
        int(opp_att), own_def, sac, dist,
        ctx["mat"], ctx["n_moves"], ctx["mover_in_check"], edge, ctx["na"], ctx["nd"],
    ]


def features_for_node(board, moves, attacker, d_node):
    """Feature matrix (len(moves) x N_FEATURES) for all candidate moves of a node."""
    ctx = node_context(board, attacker, len(moves))
    return np.array([move_features(board, m, attacker, d_node - 1, ctx) for m in moves], dtype=np.float32)


# ------------------------------------------------------------------ tags ---
def classify_move(board, m, attacker):
    """Tags used for the analysis of question 16 (computed on the key move)."""
    ctx = node_context(board, attacker, 1)
    f = dict(zip(FEATURE_NAMES, move_features(board, m, attacker, 0, ctx)))
    return {"sacrifice": bool(f["is_sacrifice"]), "quiet": bool(f["is_quiet"]),
            "check": bool(f["gives_check"]), "capture": bool(f["is_capture"])}


# -------------------------------------------------------- baseline order ---
def heuristic_order(board, moves):
    """Classic ordering: checks -> captures (MVV-LVA) -> other moves.

    Returns the moves sorted; ties keep generation order (deterministic)."""
    keyed = []
    for i, m in enumerate(moves):
        chk = board.gives_check(m)
        cap = board.is_capture(m)
        victim = VALUE.get(board.captured_type(m), 0) if cap else 0
        attacker_val = VALUE.get(board.piece_type(m), 0)
        keyed.append(((not chk, not cap, -victim, attacker_val, i), m))
    keyed.sort(key=lambda x: x[0])
    return [m for _, m in keyed]
