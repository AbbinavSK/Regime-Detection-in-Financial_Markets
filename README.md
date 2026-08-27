# Regime Detection in Financial Markets using ML

UCL MSc dissertation project: builds correlation-network representations of stock indices, embeds them via three competing descriptor methods (Graph2Vec, eigenvector centrality, spectral/RMT features), and clusters the embeddings (K-Means / Gaussian HMM) into market regimes — Calm / Transitional / Crisis — validated against VIX for the S&P 500. Four markets are covered: S&P 500, Nikkei 225, FTSE 350, CSI 300.

See [`LOGIC.md`](LOGIC.md) for *why* the pipeline is shaped this way (the research design, the 2×2 descriptor comparison, what's been checked and fixed). See [`docs/architecture/pipeline.md`](docs/architecture/pipeline.md) for *how the code is wired together* stage by stage. See [`docs/critical-evaluation.md`](docs/critical-evaluation.md) for what can currently be concluded from results this pipeline has produced, and what's still open. See [`CLAUDE.md`](CLAUDE.md) for a terser operational reference (commands, module map, conventions).

## Environment

Python runs from the conda **base** environment at `c:\Users\Abbin\miniconda3` (not a named env). Key packages: numpy, pandas, networkx, scipy, scikit-learn, hmmlearn, gensim, yfinance, joblib. No requirements.txt/environment.yml exists.

```
"c:\Users\Abbin\miniconda3\python.exe" data_processing.py
"c:\Users\Abbin\miniconda3\python.exe" modelling_sweep_raw.py --quick
"c:\Users\Abbin\miniconda3\python.exe" sweep_run.py
"c:\Users\Abbin\miniconda3\python.exe" -m jupyter nbconvert --to notebook --execute <notebook>.ipynb
```

## Pipeline, in one paragraph

Four `data_download_*.ipynb` notebooks scrape/backfill each market's constituents and screen the price panels (`data/`). `data_processing.py` turns the cleaned prices into rolling-window correlation networks (raw arm only) and three window lengths, for all four markets (`outputs/*.pkl`). Three sweep scripts — `modelling_sweep_raw.py` (Graph2Vec), `eigencentrality_sweep_raw.py` (eigenvector centrality), `spectral_sweep_raw.py` (spectral/RMT) — each embed the `raw`-arm networks, cluster them (K-Means and HMM, K=3), rank clusters into Calm/Transitional/Crisis by mean correlation, and (S&P 500 only) score the Crisis regime against VIX. A fourth script, `embedding_geometry_sweep.py`, asks a different question, fixed to S&P 500/raw arm/Graph2Vec only: is the embedding's *shape* stable across the three nested network constructions (threshold/MST-threshold/TMFG-threshold), swept over all three epoch lengths? `sweep_run.py` runs all four in dependency order. Full detail: [`docs/architecture/pipeline.md`](docs/architecture/pipeline.md).

## Current status

The full pipeline has been run in this checkout: `data_processing.py` has produced all 12 raw-arm network-data
pickles (`outputs/*.pkl` — 4 markets × 3 epoch lengths), and the three regime-detection sweeps plus
`embedding_geometry_sweep.py` have all completed (`outputs/modelling_sweep_results_raw.csv` 36 rows,
`outputs/eigencentrality_sweep_results_raw.csv` 24 rows, `outputs/spectral_sweep_results_raw.csv` 12 rows,
`outputs/embedding_geometry_results.csv` 9 rows, backed by 72 `.joblib` bundles in `outputs/models/`), all with
no unexpected `NaN`s (only the architecturally-expected VIX-columns-are-SP500-only pattern). `COMPILED.md` and
`docs/critical-evaluation.md` describe results from earlier runs of this codebase — some predating the `mg`/`gr`
arm removal, some predating the market-scope widening to four markets — and have not yet been re-verified number-
by-number against this specific run; treat them as a record of the pipeline's known behaviours and open risks,
not as a guarantee that every cited figure matches what's on disk right now.

The three walkthrough notebooks (`modelling_walkthrough.ipynb`, `eigencentrality_walkthrough.ipynb`, `spectral_walkthrough.ipynb`) and `embedding_geometry_sweep.py` remain scoped to S&P 500 (+ Nikkei 225 for the walkthroughs) only — they were not widened alongside the three regime-detection sweep scripts' market coverage.
