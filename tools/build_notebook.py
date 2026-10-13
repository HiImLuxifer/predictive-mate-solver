"""Builds and executes notebook.ipynb (run from the project root:
python tools/build_notebook.py). The notebook itself is self-contained and can
be re-run in Jupyter; set the RECOMPUTE_* flags to regenerate everything."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbrun import execute

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

C = []


def md(t):
    C.append(("md", t))


def code(t):
    C.append(("code", t))


md(r"""# Predictive Regulated Search — problemi di matto in N mosse

**Problema.** Data una posizione "il tratto muove e dà matto in N" (N = 1, 2, 3), scegliere la **prima mossa**.
La decisione si ottiene con una ricerca AND/OR a budget: nodi OR = attaccante (basta una mossa), nodi AND = difensore (vanno confutate tutte le difese).

| Sistema | Ordinamento | Allocazione del calcolo |
|---|---|---|
| **A** baseline | euristica classica: scacchi → catture MVV-LVA → resto | depth-first uniforme |
| **B** predittivo | predittore appreso $\hat P_\theta(s,a)$ | depth-first uniforme (come A) |
| **C** predittivo + regolato | come B | **regolatore** alla radice: ampiezza, quote di budget, stop |

Budget online = numero di **espansioni** (generazioni di mosse legali). Il calcolo dell'insegnante e del training è offline e separato.

Struttura del progetto: `src/` (codice), `data/` (puzzle e coppie etichettate), `models/`, `results/`, `figures/`, `tests/`.
Il notebook usa i risultati già calcolati; per rigenerare tutto mettere a `True` i flag qui sotto (tempi su 2 core).""")

code(r"""import os, sys, json, pickle, subprocess
ROOT = os.path.abspath('.')
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)

RECOMPUTE_DATA  = False   # ~20 min: genera e certifica i puzzle (Step 1)
RECOMPUTE_MODEL = False   # ~4 min : etichette dell'insegnante + training (Step 3-4)
RECOMPUTE_TUNING = False  # ~2 min : taratura del regolatore su validation (Step 7)
RECOMPUTE_BENCH = False   # ~15 min: benchmark sul test + tempo reale (Step 8-9)

def run(script, *args):
    r = subprocess.run([sys.executable, script, *args], cwd=SRC, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    r.check_returncode()

from minichess import Board
from analysis import (load_runs, load_info, summary, paired_gain, efficiency, plot_solve_curves,
                      residual_analysis, stop_danger, style_ax, COL, SYSTEM_LABEL, INK2)
print('ok')""")

md(r"""## Step 0 — Motore scacchistico e correttezza

L'ambiente di sviluppo non permetteva di installare `python-chess`, quindi `src/minichess.py` implementa da zero la generazione delle mosse legali (arrocco, en passant, promozioni, inchiodature, scacchi doppi).
È validato con i conteggi *perft* di riferimento su 6 posizioni standard. Il secondo test verifica che il sistema A, con budget illimitato, dimostri la mossa corretta su tutti i puzzle provati.""")

code(r"""run(os.path.join('..', 'tests', 'test_perft.py'))
run(os.path.join('..', 'tests', 'test_search.py'))""")

md(r"""## Step 1 — Dati

Il dataset principale è estratto direttamente dal database aperto di Lichess (`https://database.lichess.org/lichess_db_puzzle.csv.zst`) e certificato tramite il `Teacher` esaustivo (`src/build_lichess_dataset.py`):
posizioni umane reali da partite giocate, con rating Elo, URL Lichess e PuzzleId, tenute solo se il solver esaustivo certifica:

- matto in **esattamente** N mosse;
- **1–2 prime mosse vincenti**, cioè una decisione reale.

L'insieme di **tutte** le prime mosse vincenti è salvato: è la verità indipendente usata per valutare le decisioni.

Split: N=1,2 → train/val/test; N=3 → **solo test** (shift di distribuzione).
Un set extra di 600 matti in 3 (sempre dal database Lichess) serve per l'ablation 4b e per tarare il regolatore.
Lo split è riproducibile (`data/split.json`) e garantisce insiemi di test disgiunti da training e validazione.""")

code(r"""if RECOMPUTE_DATA:
    run('build_lichess_dataset.py', '--n1', '2500', '--n2', '2500', '--n3', '500', '--n3_extra', '600')
from experiment import load_puzzles
P = pd.DataFrame(load_puzzles())
split = json.load(open('data/split.json'))
where = {i: s for s, ids in split.items() for i in ids}
P['split'] = P.id.map(where)
P['cls'] = np.where(P.sacrifice, 'sacrifice', np.where(P.quiet, 'quiet', 'check/capture'))
tab = P.groupby('n').agg(puzzles=('id', 'size'), key_check=('check', 'mean'), key_capture=('capture', 'mean'),
                         key_quiet=('quiet', 'mean'), key_sacrifice=('sacrifice', 'mean'),
                         two_keys=('keys', lambda k: np.mean([len(x) == 2 for x in k])), legal_moves=('n_legal', 'mean'))
print(P.groupby(['split', 'n']).size().unstack(fill_value=0))
tab.round(3)""")

code(r"""from viz import draw_board
ex = P[(P.n == 2) & P.quiet].iloc[0]
fig, ax = plt.subplots(figsize=(3.6, 3.6))
draw_board(ax, ex.fen, arrow=ex['keys'][0], title=f"{ex.id}: matto in 2, chiave {ex.key_san} (mossa quieta)")
print(ex.fen)""")

md(r"""## Step 2 — Ricerca: stato, espansione, orizzonte, frontiera attiva (`src/search.py`)

- **Oggetto della ricerca**: l'albero AND/OR delle continuazioni per ≤ 2N−1 semimosse a partire dalla posizione del puzzle; la decisione è la mossa alla radice.
- **Stato**: (posizione, *d*), con *d* = semimosse ancora a disposizione dell'attaccante.
- **Espansione**: una generazione di mosse legali di un nodo. È l'unità di budget.
- **Orizzonte**: le foglie dell'albero esplicito, cioè nodi generati ma non espansi, più le foglie di profondità (*d* esaurito: lì la ricerca esplicita si ferma).
- **Frontiera attiva**: la parte dell'orizzonte ancora idonea a ricevere calcolo. Ne sono esclusi i nodi sotto un nodo già risolto (PROVEN/DISPROVEN) e, in C, i nodi sotto mosse radice *rinviate* dal regolatore.

Cosa succede alle alternative che escono dalla frontiera attiva:

1. **risolte**: il loro esito è certo, perché il solver è corretto;
2. **tagliate**: sono fratelli di un nodo che ha già deciso il padre, quindi irrilevanti;
3. **rinviate** (solo C): non sono risolte, e la loro probabilità resta nel residuo ρ.

Qui sotto: un matto in 3 con budget 200. Per ogni sistema si vede dove è andato il calcolo e com'è fatta la frontiera allo stop.""")

code(r"""from model import Predictor, root_candidates
from search import Searcher, HeuristicPolicy, ModelPolicy, frontier_stats, root_allocation, softmax_logits
from regulator import solve_regulated
from run_all import config
params, extra = config()
pred = Predictor(pickle.load(open('models/gbm.pkl', 'rb')))
pz = load_puzzles('test'); pz = {p['id']: p for p in pz}
p = pz['m3_00158']
rows = []
alloc = {}
for name in 'ABC':
    pred.clear(); b = Board(p['fen'])
    s = Searcher(HeuristicPolicy() if name == 'A' else ModelPolicy(pred), 200)
    m, proven = s.solve(b, 3) if name != 'C' else solve_regulated(s, b, 3, pred, params)
    alloc[name] = {b.uci(k): v for k, v in root_allocation(s).items() if v > 0}
    rows.append({'system': name, 'move': b.uci(m), 'correct': b.uci(m) in p['keys'], 'proven': proven, 'used': s.used, **frontier_stats(s)})
print('chiave:', p['keys'], '| mosse legali:', p['n_legal'])
display_df = pd.DataFrame(rows); print(display_df.to_string(index=False))
print('\nespansioni spese sotto ogni mossa radice:')
for k, v in alloc.items(): print(k, dict(sorted(v.items(), key=lambda x: -x[1])[:6]))""")

md(r"""## Step 3–4 — Target dall'insegnante e predittore $\hat P_\theta$ (`src/teacher.py`, `src/model.py`)

**Cosa si predice.** $\hat P_\theta(s,a)$ = probabilità che, dopo la mossa *a*, l'attaccante abbia ancora un matto forzato entro le semimosse rimaste.

- Nei nodi OR significa "*a* inizia il matto": si esplorano prima i valori alti.
- Nei nodi AND significa "la risposta *b* non salva il difensore": si esplorano prima i valori bassi (la fuga più probabile), per confutare presto.

**Target.** Il solver esaustivo senza budget etichetta **tutte** le mosse di:

- la radice;
- i nodi AND dopo la chiave e dopo 2 mosse sbagliate "difficili" (scacchi o catture);
- alcuni nodi OR interni.

Il predittore vede così la stessa distribuzione di nodi che incontra in ricerca.

**Modello.** Gradient boosting (sklearn `HistGradientBoostingClassifier`, LightGBM non installabile) su 28 feature nel riferimento "re da mattare": scacco, cattura, pezzo, case di fuga prima/dopo, zona del re attaccata, pezzo in presa (sacrificio), distanza dal re, materiale, ecc. Le feature richiedono solo make/unmake e test di attacco, **mai** una generazione di mosse del figlio: per questo non contano come espansioni, esattamente come il test di scacco usato dall'euristica di A. Ablation: regressione logistica sulle stesse feature.""")

code(r"""if RECOMPUTE_MODEL:
    run('train.py'); run('train_shift.py')
stats = json.load(open('results/dataset_stats.json')); M = json.load(open('results/predictor_metrics.json'))
print('coppie di training:', stats['train']['pairs'], '| per tipo di nodo:', stats['train']['by_type'],
      '| positivi:', round(stats['train']['pos_rate'], 3), '| tempo etichettatura:', round(stats['label_time_s']), 's')
rows = []
for mdl in ('gbm', 'logreg'):
    for sp in ('val', 'test_n12', 'test_n3'):
        for typ, v in M[mdl][sp].items():
            rows.append({'model': mdl, 'split': sp, 'node': typ, **{k: v[k] for k in ('auc', 'logloss', 'ece', 'n')}})
pd.DataFrame(rows).pivot_table(index=['split', 'node'], columns='model', values=['auc', 'ece']).round(3)""")

code(r"""rows = []
for mdl in ('gbm', 'logreg'):
    for sp, v in M[mdl]['root_decision'].items():
        rows.append({'model': mdl, 'split': sp, **v})
print('Accuratezza della sola predizione alla radice (argmax P̂ = mossa vincente), T calibrata:',
      {m: M[m]['T'] for m in ('gbm', 'logreg')})
pd.DataFrame(rows).set_index(['model', 'split']).round(3)""")

code(r"""d = np.load('data/pairs.npz', allow_pickle=True)
gbm = pickle.load(open('models/gbm.pkl', 'rb'))
fig, ax = plt.subplots(figsize=(4.2, 3.8))
for X, y, lab, col in ((d['Xte'], d['yte'], 'test N=1,2', COL['B']), (d['Xte3'], d['yte3'], 'test N=3 (shift)', COL['C'])):
    pp = gbm.predict_proba(X)[:, 1]
    bins = np.linspace(0, 1, 11); idx = np.digitize(pp, bins[1:-1])
    xs = [pp[idx == i].mean() for i in range(10) if (idx == i).sum() > 20]
    ys = [y[idx == i].mean() for i in range(10) if (idx == i).sum() > 20]
    ax.plot(xs, ys, marker='o', lw=2, color=col, label=lab)
ax.plot([0, 1], [0, 1], color='#c3c2b7', lw=1, ls='--')
ax.set_xlabel('P̂θ predetta', fontsize=9, color=INK2); ax.set_ylabel('frequenza osservata', fontsize=9, color=INK2)
ax.set_title('Calibrazione di P̂θ (tutte le coppie)', fontsize=10, loc='left'); style_ax(ax); ax.legend(frameon=False, fontsize=8)
coef = pd.Series(M['logreg_coefficients']).head(10)
print('Feature più influenti (coefficienti logreg standardizzati):'); print(coef.round(2).to_string())""")

md(r"""## Step 5–6 — Ricerca predittiva (B), distribuzione π e massa residua ρ

**B** è identico ad A tranne l'ordine delle mosse: la differenza B − A misura il guadagno della sola predizione.

**π(a | s)** è definita alla radice:

$$\pi(a\mid s)=\mathrm{softmax}\big(\mathrm{logit}\,\hat P_\theta(s,a)/T\big)$$

con T calibrata sul validation (minimizza la NLL dell'insieme delle mosse vincenti). Significa "credenza del modello che *a* sia la prima mossa del matto".

**Massa residua ρ.** Detta $a^*_t$ la scelta corrente (mossa dimostrata, altrimenti argmax π tra le non confutate):

$$\rho_t=\sum_{a\in U_t}\pi(a\mid s),\qquad U_t=\{\text{mosse radice} \neq a^*_t \text{ né dimostrate né confutate}\}$$

ρ è la probabilità, secondo il modello, che la soluzione stia tra alternative che la ricerca non ha ancora deciso. Si scompone in mosse mai toccate e mosse esplorate solo in parte. Quando una mossa viene confutata, la sua massa si risolve. Se una mossa è dimostrata, la decisione è certa e ρ non rappresenta più un rischio.""")

code(r"""b = Board(p['fen']); moves = root_candidates(b, 3)
pp = pred.predict(b, moves, b.side, 5); pi = softmax_logits(pp, params.T)
o = np.argsort(-pi)[:8]
print(pd.DataFrame({'move': [b.uci(moves[i]) for i in o], 'P_hat': pp[o].round(3), 'pi': pi[o].round(3),
                    'key': [b.uci(moves[i]) in p['keys'] for i in o]}).to_string(index=False))""")

md(r"""## Step 7 — Regolatore (sistema C, `src/regulator.py`)

$O_t \to R \to U_t$, ricalcolato dopo ogni round di allocazione alla radice.

**Osservabili $O_t$:**

- ρ_t;
- il margine $\tilde\pi_{(1)}-\tilde\pi_{(2)}$, con $\tilde\pi$ = π rinormalizzata sulle mosse non confutate;
- il proxy di errore $\hat R(a)=H_{norm}(\tilde\pi)\cdot w(a)$, con w = 1 per mosse quiete o sacrifici, 0.5 altrimenti.

**Decisioni $U_t$:**

1. **Ampiezza e quote.** Vengono servite le mosse che coprono 1−ε della massa non confutata, con ε più piccolo quando il margine è basso. Ogni mossa riceve una fetta ∝ π(a)(1+λR̂(a)). La fetta minima è 2N+1 espansioni, cioè abbastanza per raggiungere l'orizzonte. Il budget del round raddoppia a ogni round. Le mosse fuori ampiezza sono **rinviate**, non potate.
2. **Stop.** La ricerca si ferma se una mossa è dimostrata, oppure se ρ_t < τ con margine ampio e la mossa scelta è già stata esplorata.

Sotto la radice C usa la stessa ricerca di B. Nei nodi AND il regolatore non pota mai difese: una prova resta una prova.

**Taratura** solo su validation: N=2 val e 200 matti in 3 extra. Si sceglie una sola configurazione con criterio max-min sui due set.
Nota di sviluppo: la prima versione dava fette di 2 espansioni a 12 candidati e non risolveva nulla (C 0.785 < B 0.863 su N=2). La granularità minima 2N+1 ha corretto il problema.""")

code(r"""if RECOMPUTE_TUNING:
    run('tune.py', 'n2'); run('tune.py', 'n3', '100,200,400')
for which in ('n2', 'n3'):
    t = pd.DataFrame(json.load(open(f'results/tuning_{which}.json')))
    t['config'] = t.config.astype(str)
    print(f'--- taratura {which} (prime 5 + riferimento B)')
    print(pd.concat([t.head(5), t[t.config == 'B (reference)']]).round(3).to_string(index=False))
print('\nconfigurazione scelta:', {k: v for k, v in vars(params).items()})""")

md(r"""## Step 8 — Esperimento: A vs B vs C a parità di espansioni (test)

Test: 400 matti in 1, 400 in 2, 500 in 3. Il budget per puzzle è tarato sulla difficoltà di ciascun N, misurata su validation:
mediana di A per dimostrare ≈ 2.5 espansioni (N=1), 21 (N=2), 400 (N=3).

Metriche:

- **solve**: la prima mossa scelta è vincente, verificato dal solver esaustivo;
- **proven**: la mossa è anche dimostrata entro il budget;
- **used**: espansioni consumate (C può fermarsi prima).""")

code(r"""if RECOMPUTE_BENCH:
    run('run_all.py'); run('run_walltime.py'); run('timing.py')
df = load_runs('main'); info = load_info('main')
S = summary(df[df.system.isin(['A', 'B', 'C'])])
S.pivot_table(index=['N', 'budget'], columns='system', values=['solve', 'proven', 'used']).round(3)""")

code(r"""plot_solve_curves(S, metric='solve', fname='solve_vs_budget.png')
plot_solve_curves(S, metric='proven', fname='proven_vs_budget.png');""")

code(r"""G = pd.concat([paired_gain(df, 'A', 'B'), paired_gain(df, 'B', 'C')])
print('Guadagno di predizione (B − A) e di regolazione (C − B), test di McNemar esatto sulle coppie discordanti')
G[['N', 'budget', 'comparison', 'gain', 'p_mcnemar']].pivot_table(index=['N', 'budget'], columns='comparison', values=['gain', 'p_mcnemar']).round(4)""")

code(r"""E = efficiency(df)
print('Costo di una risposta certa: mediana delle espansioni sui puzzle dimostrati da tutti e tre i sistemi')
E[E.N >= 2].round(1)""")

md(r"""### Stesso confronto a parità di **tempo reale**

Il budget dell'esperimento sono le espansioni. Il calcolo delle feature e del modello (sklearn, Python) però costa 1–2 ms per nodo, contro 0,05 ms di un'espansione.
Qui ogni ricerca gira da sola, con cache vuota, e ad A si danno budget fino a 16× più grandi.""")

code(r"""W = load_runs('walltime')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, n in zip(axes, (2, 3)):
    for s in 'ABC':
        g = W[(W.n == n) & (W.system == s)].groupby('budget').agg(solve=('correct', 'mean'), ms=('time', lambda t: 1000 * t.mean()))
        ax.plot(g.ms, g.solve, marker='o', lw=2, color=COL[s], label=SYSTEM_LABEL[s])
    ax.set_xscale('log'); ax.set_title(f'Matto in {n}', fontsize=11, loc='left')
    ax.set_xlabel('ms per puzzle (wall-clock, log)', fontsize=9, color=INK2); ax.set_ylim(0, 1.04); style_ax(ax)
axes[0].set_ylabel('prima mossa corretta', fontsize=9, color=INK2)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}')); axes[1].legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig('figures/solve_vs_walltime.png', dpi=150)
print(json.dumps(json.load(open('results/timing.json')), indent=0))""")

md(r"""## Step 9 — Ablation

| variante | cosa cambia | domanda di verifica |
|---|---|---|
| `C_flat` | π uniforme (T → ∞) alla radice | distribuzione appiattita |
| `C_noR` | niente R̂ (λ = 0, nessuna ammissione) | rimuovere una sorgente di incertezza |
| `C_admit` | R̂ ammette fino a 2 mosse rinviate | il segnale "needs more thinking" serve? |
| `C_noStop` | niente stop anticipato | residuo forzato a essere risolto |
| `B_noise`, `C_noise` | rumore N(0, 1.5) sui logit, T non ricalibrata | predittore meno accurato e mal calibrato |
| `B_logreg`, `C_logreg` | regressione logistica (più debole ma calibrata) | modello più semplice |
| `B_withn3`, `C_withn3` | modello addestrato anche su matti in 3 | cambia la distribuzione di training |""")

code(r"""A = summary(df[df.n >= 2])
order = ['A', 'B', 'C', 'C_flat', 'C_noR', 'C_admit', 'C_noStop', 'B_noise', 'C_noise', 'B_logreg', 'C_logreg', 'B_withn3', 'C_withn3']
tbl = A.pivot_table(index=['N', 'budget'], columns='system', values='solve')[order]
print('solve rate'); print(tbl.round(3).to_string())
print('\nespansioni usate'); print(A.pivot_table(index=['N', 'budget'], columns='system', values='used')[order].round(1).to_string())""")

code(r"""for a, c in (('B_noise', 'C_noise'), ('B_logreg', 'C_logreg'), ('B_withn3', 'C_withn3'), ('C', 'C_noStop'), ('C', 'C_flat')):
    g = paired_gain(df[df.n == 3], a, c)
    print(f'{c} − {a} (N=3):', ', '.join(f"{r.budget}: {r.gain:+.3f} (p={r.p_mcnemar:.3f})" for r in g.itertuples()))""")

md(r"""## Step 6 bis — ρ come misura di rischio, orizzonte vs frontiera attiva""")

code(r"""u = df[(~df.proven) & df.system.isin(['A', 'B', 'C']) & (df.n >= 2)].copy()
u['rho_bin'] = pd.cut(u.rho, [-0.01, 0.05, 0.2, 0.5, 0.8, 1.0])
print('Decisioni NON dimostrate: tasso di errore in funzione di ρ allo stop')
print(u.groupby(['rho_bin'], observed=True).agg(error_rate=('correct', lambda x: 1 - x.mean()), n=('correct', 'size')).round(3).to_string())
F = df[df.system.isin(['A', 'B', 'C']) & (df.n == 3)].groupby(['budget', 'system'])[['fr_expanded', 'fr_horizon', 'fr_active', 'fr_closed', 'fr_deferred']].mean()
print('\nMatto in 3: dimensione media di albero espanso, orizzonte e frontiera attiva allo stop'); print(F.round(1).to_string())""")

md(r"""## Step 10 — Domanda 16: quando cercare conviene anche se la predizione "non è interessante"

Le mosse chiave **quiete** hanno $\hat P_\theta$ basso (rango mediano 2, $\hat P$ mediano 0.14–0.23). L'euristica "scacchi prima" di A le esplora per ultime.""")

code(r"""info['cls'] = np.where(info.sacrifice, 'sacrifice', np.where(info.quiet, 'quiet', 'check/capture'))
print(info.groupby(['n', 'cls']).agg(puzzles=('id', 'size'), pred_top1=('key_rank', lambda r: (r == 1).mean()),
      median_rank=('key_rank', 'median'), median_P_key=('p_key', 'median'), mean_R_key=('R_key', 'mean')).round(3).to_string())
X = df.merge(info[['id', 'cls']], on='id')
for n, b in ((2, 16), (3, 200), (3, 800)):
    g = X[(X.n == n) & (X.budget == b) & X.system.isin(['A', 'B', 'C'])]
    print(f'\nN={n}, budget {b}: solve rate per tipo di mossa chiave')
    print(g.pivot_table(index='cls', columns='system', values='correct').round(3).to_string())""")

code(r"""from viz import draw_board, plot_allocation
examples = ['m3_00158', 'm3_00400', 'm3_00163']
fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), gridspec_kw={'height_ratios': [1.25, 1]})
for j, pid in enumerate(examples):
    q = pz[pid]; i = info.set_index('id').loc[pid]
    draw_board(axes[0, j], q['fen'], arrow=q['keys'][0],
               title=f"{pid}: chiave {q['key_san']}" + (f" (vincente anche {q['keys'][1]})" if len(q['keys']) > 1 else "")
                     + f"\nrango P̂ = {i.key_rank}, π = {i.pi_key:.3f}, R̂ = {i.R_key:.2f}")
    al, res = {}, {}
    for name in 'BC':
        pred.clear(); b = Board(q['fen'])
        s = Searcher(ModelPolicy(pred), 200)
        m, pr = s.solve(b, 3) if name == 'B' else solve_regulated(s, b, 3, pred, params)
        al[name] = {b.uci(k): v for k, v in root_allocation(s).items() if v}
        res[name] = f"{b.uci(m)} {'✓' if b.uci(m) in q['keys'] else '✗'}{' dimostrata' if pr else ''} ({s.used} esp.)"
    plot_allocation(axes[1, j], al['B'], al['C'], q['keys'])
    axes[1, j].set_title(f"budget 200 — B: {res['B']}\nC: {res['C']}", fontsize=8, loc='left')
    print(pid, 'chiave', q['keys'], '| B spende', sorted(al['B'].items(), key=lambda x: -x[1])[:4], '| C spende', sorted(al['C'].items(), key=lambda x: -x[1])[:5])
fig.tight_layout(); fig.savefig('figures/q16_examples.png', dpi=150)""")

md(r"""## Domanda 13 — errore di predizione e bisogno di ricerca

Sulle radici del test, $r(s,a)=P^*(s,a)-\hat P_\theta(s,a)$, con $P^*$ dato dall'insegnante esaustivo. Due domande:

1. il proxy R̂ individua le coppie con |r| > 0.5?
2. a livello di stato, l'incertezza (entropia di π, oppure 1 − max π) individua le posizioni in cui fermarsi alla predizione, cioè budget zero, è pericoloso?""")

code(r"""T = params.T
for nm, X_, y_, t_, p_ in (('N=1,2', d['Xte'], d['yte'], d['tte'], d['pte']), ('N=3', d['Xte3'], d['yte3'], d['tte3'], d['pte3'])):
    res, R = residual_analysis(gbm, T, X_, y_, t_, p_)
    res.pop('auc_lowP_flags_big_error', None)
    print(nm, json.dumps({k: round(v, 3) for k, v in res.items()}))
for n in (2, 3):
    print(f'N={n} stop-danger:', {k: round(v, 3) for k, v in stop_danger(info[info.n == n]).items()})""")

md(r"""## Sintesi
 
- **Predizione (B − A)**: guadagno grande e significativo a ogni budget per N = 1 (da +7 a +38 punti), per N = 2 (da +4 a +21.5 punti) e per N = 3 a budget bassi (+13.8 punti a budget 50). L'accuratezza top-1 del predittore sulle posizioni Lichess raggiunge il 96.8% su N=1, l'89.8% su N=2 e il 78.2% su N=3.
- **Regolazione (C − B)**: nulla sui matti in 1–2, dove B satura rapidamente al 97–100% e il depth-first guidato è già ottimale. Significativa e marcata sui matti in 3 (fuori distribuzione): **+9.0 punti a budget 50** ($p = 6.8\cdot 10^{-13}$), **+8.4 punti a budget 100** ($p = 1.3\cdot 10^{-10}$) e **+3.8 punti a budget 200** ($p = 5.5\cdot 10^{-4}$), riducendo drasticamente le espansioni usate (ad es. da 27.3 a 21.4 a budget 100 e da 38.0 a 23.8 a budget 200). Il beneficio emerge soprattutto quando la chiave non è al primo posto (rango $\ge 2$), dove C supera B del +41.3% a budget 50 e del +38.5% a budget 100.
- La regolazione dipende dalla **calibrazione** di π: con π appiattita (`C_flat`) o logit rumorosi il modello perde efficienza a budget maggiori, mentre con modelli più semplici ma calibrati (logreg) il vantaggio relativo cresce (+14.2 punti a budget 50).
- Il proxy **R̂** riconosce le chiavi sottostimate tra le mosse vincenti ma non aggiunge potere discriminante rispetto alle altre alternative: in ricerca `C_noR` ≈ C (risultato negativo confermato).
- **Tempo reale**: con feature e modello in Python, ogni nodo predetto costa ~2–3 ms per espansione contro ~0.07 ms di A. A parità di tempo reale A con budget maggiori vince grazie alla velocità pura del generatore di mosse.
 
Le risposte alle 16 domande sono in `report.md`.""")

if __name__ == "__main__":
    execute(C, ROOT, os.path.join(ROOT, "notebook.ipynb"))
