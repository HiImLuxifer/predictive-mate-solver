# Predictive Regulated Search — problemi di matto in N mosse

Homework AI/ML: ricerca AND/OR a budget per la prima mossa di problemi "matto in N", confrontando tre sistemi:

- **A**: baseline euristica;
- **B**: ricerca guidata da un predittore appreso;
- **C**: predittore + regolatore.

## Contenuto

| Percorso | Che cosa contiene |
|---|---|
| `notebook.ipynb` | esperimento completo, già eseguito con tutti gli output |
| `report.md` | design, risultati e risposte alle 16 domande |
| `ai_decision_record.md` | 5 interazioni con l'AI che hanno cambiato la soluzione (inclusa la transizione a Lichess) |
| `src/minichess.py` | generatore di mosse legali (zero dipendenze) |
| `src/teacher.py` | solver esaustivo AND/OR (insegnante e verità di riferimento) |
| `src/puzzles.py` | generatore e certificazione dei puzzle, loader per il CSV Lichess |
| `src/board_utils.py` | feature, euristica di A, tag sacrificio/mossa quieta |
| `src/model.py` | target, training, predittore, calibrazione di T |
| `src/search.py` | ricerca a budget con albero esplicito, orizzonte e frontiera, sistemi A e B |
| `src/regulator.py` | sistema C: osservabili O_t → decisioni U_t |
| `src/experiment.py`, `src/analysis.py`, `src/viz.py` | benchmark, statistiche, figure |
| `src/build_lichess_dataset.py` | download, estrazione e certificazione dei puzzle reali dal database Lichess |
| `src/train.py`, `train_shift.py`, `tune.py`, `run_all.py`, `run_walltime.py`, `timing.py` | script dei singoli step |
| `tools/test_lichess_sample.py` | test interattivo dimostrativo di A, B, C su puzzle Lichess con URL e rating |
| `tools/build_notebook.py` | compilazione ed esecuzione pulita di `notebook.ipynb` |
| `tests/` | test perft e test di correttezza della ricerca |
| `data/`, `models/`, `results/`, `figures/` | puzzle reali Lichess, coppie etichettate, modelli, risultati grezzi (jsonl) e figure |

## Requisiti

Python ≥ 3.10 con `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib` e utility di sistema `zstd` / `curl`. Non servono `python-chess` né Stockfish.

## Dataset
Il dataset principale (`data/puzzles.jsonl` e `data/puzzles_n3_extra.jsonl`) è composto da **puzzle reali e legali estratti dal database aperto di Lichess** e certificati tramite il `Teacher` esaustivo:
- Matto in 1: 2.500 puzzle
- Matto in 2: 2.500 puzzle
- Matto in 3: 500 puzzle (test) + 600 puzzle (extra train/val)

I file in `data/` sono già inclusi: per riprodurre l'esperimento non serve riscaricarli.

## Riprodurre (da `src/`)

```bash
python ../tests/test_perft.py                       # correttezza del motore
python ../tests/test_search.py                      # correttezza della ricerca
python build_lichess_dataset.py --n1 2500 --n2 2500 --n3 500 --n3_extra 600   # opzionale: richiede rete, curl e zstd
python train.py                                     # ~3 min, etichette + modelli sui puzzle reali Lichess
python train_shift.py                               # ~2 min, modello con matti in 3
python tune.py n2 && python tune.py n3 100,200,400  # taratura del regolatore su validation
python run_all.py                                   # ~9 min, benchmark sul test
python run_walltime.py && python timing.py          # ~6 min, confronto a parità di tempo
```

Il notebook rilegge i risultati salvati. I flag `RECOMPUTE_*` nella seconda cella rieseguono gli step.
