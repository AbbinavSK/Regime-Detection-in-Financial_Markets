# Regime Detection in Financial Markets using Machine Learning

UCL MSc Computational Finance dissertation project (Author: Abbinav Sankar Kailasam; Supervisors: Dr. Blaz
Zlicar, Prof. Fabio Caccioli). This project asks whether the *shape* of the stock market's correlation
structure — not just individual prices — can be used to detect market regimes (calm, transitional, crisis),
and whether a sophisticated machine-learned representation of that structure is actually worth its cost.

## 1. The Research Question

- **Surface question**: can market regimes be read off the evolving topology of a stock correlation network?
- **The sharper question the project is actually designed to answer**: when you build an expensive, *learned*
  graph representation (Graph2Vec) instead of a cheap, *deterministic* one (eigenvector centrality, or a
  spectral/random-matrix-theory summary that skips graph-building entirely), does the extra complexity buy
  you anything?
- This is tested as a controlled 2×2 comparison — varying (a) whether the descriptor is learned or
  handcrafted, and (b) whether it operates on a filtered graph or the raw correlation matrix directly — so
  that any performance gap can be attributed to a specific design choice, not just "the fancier method won."

## 2. Data

- Four equity markets: **S&P 500, Nikkei 225, FTSE 350, CSI 300**. Daily adjusted closes, 2001–2026, via
  `yfinance`.
- A five-stage screen (non-positive prices → extreme return jumps → incomplete history → market-wide
  non-trading days → prolonged trading halts) cleans each raw panel:

  | Market | Scraped universe | Final panel size |
  |---|---|---|
  | S&P 500 | 503 | 349 |
  | Nikkei 225 | 225 | 177 |
  | FTSE 350 | 350 | 159 |
  | CSI 300 | 300 | **36** |

  CSI 300's attrition is severe but genuine — most excluded names simply don't have a full 2001–2026 trading
  history (late listings on exchanges/boards that didn't exist for most of the sample window). Any CSI 300
  result should be read with that small panel in mind.
- **S&P 500 is the only market with an external validation signal** (the VIX index, for crisis labelling);
  the other three markets are compared only on their own internal structure.

## 3. Method, Stage by Stage

1. **Correlation networks**: for each market, compute rolling-window Pearson correlation matrices at three
   window lengths (63/132/378 trading days ≈ one quarter / half-year / 18 months), then filter each window's
   matrix into a sparse graph three ways — a plain correlation **threshold**, a **Minimum Spanning Tree**
   (backbone-only), and a **Triangulated Maximally Filtered Graph** (denser, still planar). The three
   constructions are nested (threshold ⊆ MST-threshold ⊆ TMFG-threshold).
2. **Three competing descriptors** turn each window's graph (or raw correlation matrix) into a fixed-length
   vector:
   - **Graph2Vec** — a learned graph embedding (Weisfeiler-Lehman relabelling + Doc2Vec), 128 dimensions.
   - **Eigenvector centrality** — the leading eigenvector of the window's adjacency matrix; deterministic, no
     hyperparameters.
   - **Spectral/RMT** — 9 handcrafted features from the correlation matrix's own eigenspectrum, informed by
     Random Matrix Theory; never builds a graph at all.
3. **Clustering into regimes**: each descriptor's vectors are standardised, PCA-reduced, and clustered into
   **3 regimes** (K-Means and, separately, a Gaussian Hidden Markov Model) on a training split ending
   2019-12-31; the model is then applied out-of-sample to 2020 onward. Clusters are ranked by mean
   correlation and labelled **Calm / Transitional / Crisis** — the highest-correlation cluster is always
   "Crisis," so labels are comparable across methods and runs despite clustering's usual label-switching
   problem.
4. **Validation**: the detected Crisis regime is scored against the CBOE VIX index (VIX > 30 as the
   independent ground truth), S&P 500 only, via precision/recall/F1.
5. **A fourth, narrower branch** asks whether Graph2Vec's embedding *shape* is stable across the three
   network constructions (threshold/MST/TMFG) — independent of whether it detects regimes well.
6. **A financial application**: does trading on the detected regimes actually make money? A simple
   regime-timed exposure strategy (100%/50%/0% by Calm/Transitional/Crisis) is backtested against
   buy-and-hold and a cheap volatility-triggered control.

## 4. Key Findings

- **Correlation-network topology does carry a real, detectable regime signal** for the S&P 500 — the
  detected Crisis periods visibly align with the GFC, the Euro debt crisis, and COVID, and this shows up
  independently in three different places: the F1 scores against VIX, the 2-D embedding geometry (a visually
  distinct Crisis cluster), and a purely topological analysis (network density and clique structure both
  collapse sharply in crisis periods, with no clustering or descriptor involved at all).
- **There is no single best descriptor — and that ambiguity is itself the headline result.** On K-Means,
  the cheapest method (spectral/RMT, no graph, no learning) has the *highest* mean F1 (≈0.35) of the three,
  beating Graph2Vec (≈0.31). On the Hidden Markov Model, the ranking inverts: Graph2Vec leads clearly
  (≈0.30) while spectral collapses to F1 = 0.000 in every configuration. Which method looks "best" depends
  entirely on which clustering algorithm it's paired with — an expensive learned representation is not
  obviously worth its cost over much cheaper alternatives.
- **Cross-market comparison reveals one clear anomaly**: S&P 500, Nikkei 225, and CSI 300 all show a similar
  crisis-regime mean correlation (≈0.53–0.56), but **FTSE 350 sits far lower (≈0.35)**, with a visibly
  noisier detected regime timeline. This is unexplained and worth investigating further — a plausible
  candidate is that FTSE 350 is a derived panel (two independently-screened sub-indices concatenated) rather
  than a naturally cohesive index.
- **Detecting a regime is not the same as trading it profitably.** A regime-timed strategy built on the
  single best-performing model cell underperforms plain buy-and-hold *and* a much simpler
  volatility-triggered rule on both risk-adjusted return and drawdown — it sits out most of the COVID crash
  but also most of the sharp V-shaped recovery that followed it, which is what sinks its return.

## 5. Repository Structure

| Path | Contents |
|---|---|
| `data_download_*.ipynb` | One notebook per market: scrapes constituents, downloads prices, screens the panel |
| `data_processing.py` | Builds the rolling-window correlation networks for all markets/window lengths |
| `networks.py` | Graph construction (threshold/MST/TMFG), correlation/distance utilities |
| `graph2vec.py`, `eigencentrality.py`, `spectral.py` | The three descriptor implementations |
| `modelling.py`, `sweep_common.py` | Shared clustering, regime-ranking, and validation logic |
| `*_sweep_raw.py` (three files) | Full-grid experiment runners, one per descriptor |
| `*_walkthrough.ipynb` (three files) | Narrower, narrative-style walkthroughs of each descriptor |
| `embedding_geometry*.py/.py` | The embedding-shape-stability branch |
| `network_diagnostics.ipynb` | Descriptor-free topology analysis (density, cliques, community structure) |
| `threshold_diagnostics.ipynb` | Derives and validates the correlation threshold used in graph construction |
| `portfolio_analysis.ipynb` | The trading-strategy backtest |
| `data_visualisation.ipynb` | Animated visualisations of the evolving correlation networks |
| `sweep_run.py` | Runs the full experiment pipeline end to end |

`data/`, `outputs/`, and `images/` (raw/processed data, model results, figures) are not tracked in this
repository — they are regenerable by running the pipeline below, and are large (multiple GB).

## 6. Reproducing the Pipeline

Requires Python 3.13 with `numpy`, `pandas`, `networkx`, `scipy`, `scikit-learn`, `hmmlearn`, `gensim`,
`yfinance`, `joblib`, `matplotlib`.

```bash
# 1. Download and screen each market's data (run all four notebooks)
jupyter nbconvert --to notebook --execute data_download_sp500.ipynb

# 2. Build the correlation networks
python data_processing.py

# 3. Run all three descriptor sweeps + the embedding-geometry branch
python sweep_run.py

# 4. Explore results interactively via the walkthrough/diagnostic notebooks
jupyter nbconvert --to notebook --execute modelling_walkthrough.ipynb
```

## 7. Known Limitations

- VIX-based validation only exists for the S&P 500; claims about the other three markets rest on internal
  structural consistency, not external ground truth.
- The correlation threshold used in graph construction was calibrated on S&P 500 data only, then applied to
  all four markets without re-verification.
- No formal significance testing is performed anywhere in the pipeline — sample sizes (crisis episodes, not
  daily observations) are too small for it to be meaningful, and this is stated explicitly wherever a result
  might otherwise look more precise than it is.
- Graph2Vec's training is only approximately reproducible run-to-run (a property of its underlying neural
  training procedure), so exact F1 values can drift by a few thousandths between runs; the qualitative
  findings above are stable regardless.
