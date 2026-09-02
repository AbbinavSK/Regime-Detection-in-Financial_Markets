# Regime Detection in Financial Markets using Machine Learning

UCL MSc Computational Finance dissertation project (Author: Abbinav Sankar Kailasam; Supervisors: Dr. Blaz
Zlicar, Prof. Fabio Caccioli). This project asks whether the *shape* of the stock market's correlation
structure — not just individual prices — can be used to detect market regimes (calm, transitional, crisis),
and whether a sophisticated machine-learned representation of that structure is actually worth its cost.

**For the full reasoning behind every methodological choice below — why raw correlation only, why three
descriptors, why a causal split, why two clustering algorithms, why the financial backtest exists — see
[`LOGIC.md`](LOGIC.md).** This document covers what the project is and what it found; `LOGIC.md` covers why
it is built the way it is.

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

*(see [`LOGIC.md`](LOGIC.md) for why each of these choices was made)*

1. **Correlation networks**: for each market, compute rolling-window Pearson correlation matrices at three
   window lengths (63/132/378 trading days ≈ one quarter / half-year / 18 months), then filter each window's
   matrix into a sparse graph three ways — a plain correlation **threshold**, a **Minimum Spanning Tree**
   backbone plus the same above-threshold edges (**MST-threshold**), and a **Triangulated Maximally Filtered
   Graph** backbone plus the same above-threshold edges (**TMFG-threshold**). The threshold construction is
   nested inside *both* of the others (threshold ⊆ MST-threshold and threshold ⊆ TMFG-threshold, since both
   are built as "backbone + every above-threshold edge"), but MST-threshold and TMFG-threshold are **not**
   nested with each other — their backbones are two different edge sets (a spanning tree vs. a planar
   triangulation), so neither construction's edges are a subset of the other's.
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
6. **A financial application**: does trading on the detected regimes actually make money? A regime-timed
   exposure strategy (100%/50%/0% by Calm/Transitional/Crisis) is backtested against a fully-invested
   buy-and-hold benchmark on one pre-nominated configuration (S&P 500, T=132, MST-threshold, Graph2Vec,
   K-means), judged against a criterion fixed in advance: beat buy-and-hold on both Sharpe ratio and maximum
   drawdown, with both advantages persisting once the COVID-19 episode is excluded. Both strategies hold the
   same equal-weight stock basket, bought once on the first test-period day and never rebalanced — they
   differ only in how much of that basket versus cash each holds — and idle cash is assumed to earn a flat
   2.5%/year rather than a downloaded rate series. No transaction costs are modelled.

## 4. Key Findings

- **Correlation-network topology does carry a real, detectable regime signal** for the S&P 500 — the
  detected Crisis periods visibly align with the GFC, the Euro debt crisis, and COVID, and this shows up
  independently in three different places: the F1 scores against VIX, the 2-D embedding geometry (a visually
  distinct Crisis cluster), and a purely topological analysis (network density and clique structure both
  collapse sharply in crisis periods, with no clustering or descriptor involved at all).
- **There is no single best descriptor — and that ambiguity is itself the headline result.** On K-Means,
  the cheapest method (spectral/RMT, no graph, no learning) has the *highest* mean F1 (0.350) of the three,
  beating Graph2Vec (0.315) and eigenvector centrality (0.277). On the Hidden Markov Model, the ranking
  inverts: Graph2Vec leads clearly (0.287) while spectral collapses to F1 = 0.000 in every configuration.
  Which method looks "best" depends entirely on which clustering algorithm it's paired with — an expensive
  learned representation is not obviously worth its cost over much cheaper alternatives.
- **Cross-market generalisation is real but not uniform, and CSI 300 is the clearest failure case.** Nikkei
  225 most closely resembles the S&P 500 (persistent Calm/Crisis blocks under both clustering methods). FTSE
  350 is markedly more sensitive to the choice of clustering method — its K-means self-transition
  probabilities are noticeably lower (e.g. 0.676 for Calm) than its HMM ones (all three regimes above 0.82).
  CSI 300 is the standout anomaly: under the HMM, its test period collapses to 156 Calm windows, 2
  Transitional windows, and **zero** Crisis windows — the model does not recover a meaningful three-regime
  structure out of sample at all. Its 36-stock final panel (after screening removed 88% of the original CSI
  300 constituents for incomplete history) is the leading suspect.
- **Detecting a regime is not the same as trading it profitably — and the reason why is precise.** Judged
  against a criterion fixed before any number was computed (beat buy-and-hold on both Sharpe ratio *and*
  maximum drawdown, over the full test period, with both advantages persisting ex-COVID), the regime-timed
  strategy on the single pre-nominated configuration **fails**: it cuts maximum drawdown from −37.1% to
  −18.9% by reducing exposure in detected Crisis periods, but its Sharpe ratio (0.450) falls well short of
  buy-and-hold's (0.724). That shortfall is driven almost entirely by one episode — excluding COVID, the
  regime-timed strategy's Sharpe ratio (0.477) actually exceeds buy-and-hold's (0.454), while the drawdown
  advantage survives unchanged, because most of the full-period underperformance comes from sitting out the
  market's sharp V-shaped recovery immediately after the crash, not the crash itself.

## 5. Repository Structure

| Path | Contents |
|---|---|
| `LOGIC.md` | The full reasoning behind the project's design — start here for "why," not just "what" |
| `report.pdf` | The compiled dissertation |
| `data_download_*.ipynb` (4 notebooks) | One per market: scrapes constituents, downloads prices, screens the panel |
| `data/` | Raw and cleaned adjusted-close price panels for all four markets, plus the VIX series |
| `data_processing.py` | Builds the rolling-window correlation networks for every market and window length |
| `networks.py` | Graph construction (threshold/MST/TMFG), correlation/distance utilities |
| `graph2vec.py`, `eigencentrality.py`, `spectral.py` | The three descriptor implementations |
| `modelling.py`, `sweep_common.py` | Shared clustering, regime-ranking, and validation logic used identically by all three descriptors |
| `plot_style.py` | Single shared source of figure styling (colours, fonts, export settings) |
| `modelling_sweep_raw.py`, `eigencentrality_sweep_raw.py`, `spectral_sweep_raw.py` | Full-grid experiment runners, one per descriptor |
| `modelling_walkthrough.ipynb`, `eigencentrality_walkthrough.ipynb`, `spectral_walkthrough.ipynb` | Narrower, narrative-style walkthroughs of each descriptor |
| `embedding_geometry.py`, `embedding_geometry_sweep.py` | The embedding-shape-stability branch (§3, point 5) |
| `network_diagnostics.ipynb` | Descriptor-free topology analysis (density, cliques, community structure) |
| `threshold_diagnostics.ipynb` | Derives and validates the correlation threshold used in graph construction |
| `portfolio_analysis.ipynb` | The trading-strategy backtest (§3, point 6) |
| `sweep_run.py` | Runs the three descriptor sweeps and the embedding-geometry branch end to end |
| `outputs/` | Results CSVs, spectral-feature/distance pickles, and transition-matrix pickles. The raw network-data pickles and fitted model bundles are excluded from version control — they're regenerable by running the pipeline below, and individually exceed GitHub's file-size limit |
| `images/` | All figures produced by the sweep scripts and diagnostic notebooks |

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
