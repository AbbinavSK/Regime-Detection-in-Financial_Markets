# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It's
deliberately compact — for the *why* behind the pipeline, see [`LOGIC.md`](LOGIC.md); for *how the code is
wired together*, see [`docs/architecture/pipeline.md`](docs/architecture/pipeline.md); for the maths each
module implements, see [`docs/math/`](docs/math/); for what's currently reproducible vs. stale, see
[`README.md`](README.md)'s "Current status" section.

## Project

UCL MSc dissertation project: builds correlation-network representations of stock indices, embeds them via
three competing descriptor methods, and clusters the embeddings (K-Means / Gaussian HMM) into market regimes
(Calm / Transitional / Crisis), validated against VIX (S&P 500 only). Four markets — S&P 500, Nikkei 225,
FTSE 350, CSI 300 — are acquired and processed through `data_processing.py`, and all four are swept by all
three regime-detection descriptor branches (`INDEX_CODES` in each `*_sweep_raw.py`). A fourth branch,
`embedding_geometry_sweep.py`, asks a different question (is Graph2Vec's embedding *shape* stable across the
three nested network constructions — threshold/MST-threshold/TMFG-threshold) and stays fixed to one
market/arm/descriptor cell (S&P 500, raw, Graph2Vec) while sweeping epoch length and network construction.

This is a research codebase organized as Jupyter notebooks (the walkthroughs/experiments) backed by shared
library modules, plus sweep scripts for batch parameter runs. No test suite, no package manifest. Not
currently a git repository — there is no `.git` here, so don't assume version history or run git commands
against this directory. `old_docs/` holds an earlier iteration of this project's `CLAUDE.md`/`LOGIC.md`/`docs/`
— historical background only (untouched, not maintained); the files at the repo root (`CLAUDE.md`, `LOGIC.md`,
`README.md`, `docs/`) are the current source of truth and take precedence wherever they differ.

## Environment

Python runs from the conda **base** environment at `c:\Users\Abbin\miniconda3` (not a named env — do not
`conda activate` a different env). Key packages: numpy, pandas, networkx, scipy, scikit-learn, hmmlearn,
gensim, yfinance, joblib. No requirements.txt/environment.yml exists; if adding a dependency, install it into
that base env.

Run scripts/notebooks with that interpreter explicitly, e.g.:

```
"c:\Users\Abbin\miniconda3\python.exe" modelling_sweep_raw.py --quick
"c:\Users\Abbin\miniconda3\python.exe" -m jupyter nbconvert --to notebook --execute <notebook>.ipynb
```

Each of the three regime-detection sweep scripts (`modelling_sweep_raw.py`, `eigencentrality_sweep_raw.py`,
`spectral_sweep_raw.py`) takes `--quick` (smoke-test a single combo with low seed counts) and `--n-seeds`
(HMM restarts, default 20). `embedding_geometry_sweep.py` takes `--epoch-lengths` (subset of `63`/`132`/`378`),
`--graph-types` (subset of `threshold`/`mst`/`tmfg`), and `--quick` (T=132/mst combo only, reuses the frozen
Graph2Vec bundle, fits nothing fresh).

## Pipeline architecture

Disk-based stages, each reading the previous stage's output from a file:

1. `data_download_sp500.ipynb` / `data_download_nikkei225.ipynb` / `data_download_ftse350.ipynb` /
   `data_download_csi300.ipynb` — one self-contained notebook per market, writes `data/{INDEX}_AdjClose_Raw.csv`
   and `data/{INDEX}_AdjClose_Cleaned.csv` (FTSE 350 is a post-hoc concat of independently-screened FTSE100 +
   FTSE250, so it has no `_Raw.csv` of its own). `data/VIX_Close.csv` is the S&P 500 validation series,
   downloaded only in the SP500 notebook — no other market has any validation series.
2. `data_processing.py` — turns cleaned prices into rolling-window correlation networks (raw arm only, the
   threshold via `networks.ARM_THRESHOLDS`, derived in `threshold_diagnostics.ipynb` from S&P 500/T=132 only
   and applied globally — see `docs/math/graph-filtering.md`) for all four markets and three epoch lengths (63/132/378
   days), writing `outputs/{INDEX}_raw_network_data_{epoch_length}d.pkl` (12 files). Two other correlation
   transforms, `mg`/`gr`, were built early in the project but never used by any regime-detection branch, so
   both were dropped from `networks.py`/`data_processing.py`.
3. Three parallel regime-detection branches read those pickles, all four markets:
   `modelling_sweep_raw.py` / `modelling_walkthrough.ipynb` (Graph2Vec, `graph_type` in
   threshold/mst/tmfg), `eigencentrality_sweep_raw.py` / `eigencentrality_walkthrough.ipynb` (leading
   eigenvector centrality, mst/tmfg only — plain threshold graphs are too fragmented for Perron-Frobenius),
   `spectral_sweep_raw.py` / `spectral_walkthrough.ipynb` (spectral/RMT features on the raw correlation
   matrix, no graph construction at all). Each writes its own results CSV to `outputs/` plus a per-combo
   `.joblib` bundle under `outputs/models/`. The three walkthrough notebooks are narrower than the sweep
   scripts — S&P 500 + Nikkei 225 only, one fixed combo each — and were not widened when the sweep scripts'
   market scope grew to four. `data_visualisation.ipynb` and `threshold_diagnostics.ipynb` are independent,
   exploratory notebooks (the latter is where `networks.ARM_THRESHOLDS` was derived, S&P 500/T=132 only).
4. `embedding_geometry_sweep.py` / `embedding_geometry.py` — Graph2Vec only, raw arm only, downstream of only
   `modelling_sweep_raw.py`'s frozen `.joblib` bundles (eigencentrality and spectral are not used by this
   branch at all). Fixed to `INDEX_CODE="SP500"`, cosine distance, 2-D MDS — neither is CLI-overridable.
   `EPOCH_LENGTHS` (`--epoch-lengths`, 63/132/378) and `GRAPH_TYPES` (`--graph-types`, threshold/mst/tmfg) are
   both swept axes; all three graph types are includable since this branch never needed eigencentrality's
   connectivity guarantee. Asks whether Graph2Vec's embedding-space shape is stable across the three nested
   constructions, fits no clustering of its own.
5. `portfolio_analysis.ipynb` — the dissertation's financial-application section (§4.4): a single,
   pre-nominated backtest cell (S&P 500, raw arm, T=132, MST-threshold, Graph2Vec, K-means only — no HMM,
   no sweep over any axis), reading `outputs/models/SP500_raw_132d_mst.joblib` by inference-only replay
   (`embedding_geometry.load_graph2vec_raw`/`raw_arm_regime_labels`) to reconstruct test-period μ-ranked
   regime labels, then simulating two daily exposure strategies (buy-and-hold and graded regime-timed)
   against a risk-free leg. Not part of `sweep_run.py`'s chain — a downstream
   analysis notebook, like `data_visualisation.ipynb`/`threshold_diagnostics.ipynb`. Key decisions: the
   risk-free rate is `^IRX` (13-week T-bill) via `yfinance`, downloaded into a notebook variable only (never
   written to `data/`) and converted to a daily simple rate by 252-day compounding; the equal-weighted sleeve
   rebalances only every 10 trading days (the window-advance cadence), drifting between; Crisis episodes for
   the robustness section are identified from contiguous Crisis-labelled daily runs, cross-referenced against
   `plot_style.CRISIS_EVENTS`'s entries within 90 days. Writes nothing to `outputs/`/`images/` — every result
   is displayed inline, nothing is saved.

`sweep_run.py` sequences all four stages 3-4 scripts as subprocesses in dependency order (the three raw
sweeps, then `embedding_geometry_sweep.py` last), continuing past a failed step. Running
`embedding_geometry_sweep.py` before `modelling_sweep_raw.py` has populated `outputs/models/` with the
matching `SP500_raw_{63,132,378}d_{threshold,mst,tmfg}.joblib` bundles (all nine, one per epoch_length x
graph_type combo — `eigencentrality_sweep_raw.py`/`spectral_sweep_raw.py`'s bundles are not a dependency,
since neither descriptor is used by this branch) will fail.

**Don't fork a new notebook or script for a parameter change** (new `epoch_length`, `arm`, or `graph_type`) —
add a grid point to the relevant sweep script's `EPOCH_LENGTHS`/`ARMS`/`GRAPH_TYPES`/`INDEX_CODES` instead.
Reserve new files for genuinely new methodology.

## Library modules

- **`networks.py`** — graph filtering (threshold/MST/TMFG, via `threshold_graph`/`mst_graph`/
  `mst_threshold_graph`/`tmfg_graph`/`tmfg_threshold_graph`) on the raw correlation matrix.
  `build_arm_networks` is the end-to-end builder; `ARM_THRESHOLDS` holds the raw-arm correlation cutoff
  (`raw: 0.65`).
- **`graph2vec.py`** — self-contained Graph2Vec (Weisfeiler-Lehman relabeling + `gensim` Doc2Vec, PV-DBOW), no
  `karateclub` dependency. `fit`/`get_embedding`/`infer`/`oov_rate`.
- **`eigencentrality.py`** — largest connected component extraction, leading-eigenvector (Perron-Frobenius)
  computation via `eigsh`, participation-ratio/entropy localisation diagnostics, and the
  `fit_centrality_embed`/`transform_centrality_embed` StandardScaler+PCA pair mirroring Graph2Vec's.
- **`spectral.py`** — per-window eigendecomposition of the raw correlation matrix
  (`window_spectral`), 9 scalar features (`SPECTRAL_FEATURE_COLUMNS`), `fit_spectral_embed`/
  `transform_spectral_embed` mirroring `eigencentrality.py`'s pair. Rank is computed as `min(N, T-1)`, not
  `min(N,T)` — Pearson correlation demeans each column, costing one degree of freedom; this matters at
  `T=132` with `N` in the hundreds, where most eigenvalues are structurally zero, not just small.
- **`modelling.py`** — causal train/test split (`split_train_test`, cutoff `2019-12-31`), then shared,
  embedding-agnostic clustering/ranking/validation used identically by all three descriptor branches:
  `cluster_kmeans`/`cluster_hmm` (fit on train only; HMM sweeps seeds 0..n_seeds and keeps the best
  log-likelihood model, and test decoding is causal — `_causal_decode` re-runs Viterbi on each growing prefix
  so no label depends on later observations), `fit_regime_order`/`apply_regime_order` (ranks clusters by mean
  `mu_offdiag` on train only — highest-mu cluster is "Crisis" — so cluster index → regime name is consistent
  across runs/methods despite K-means/HMM's arbitrary label-switching), `validate_regime` (F1 against VIX >
  30, S&P 500 only).
- **`sweep_common.py`** — plotting (`plot_regime_timeline`/`plot_vix_overlay`/`plot_transition_comparison`,
  `savefig`) and `fit_and_rank_regimes` (K-means + HMM, mu-ranked, HMM dwell times included) shared verbatim
  by `modelling_sweep_raw.py`/`eigencentrality_sweep_raw.py`/`spectral_sweep_raw.py` — the three descriptor
  branches call the exact same functions here rather than each keeping its own copy, so a change (e.g. a
  new diagnostic column) only needs making once and every branch's CSV schema stays in lockstep.
- **`embedding_geometry.py`** — pure calculation, no plotting: Graph2Vec raw-vector extraction (pre-PCA 128-d,
  via a frozen bundle's own model, never refit), $L_2$-normalisation for shape comparison, cosine
  pairwise-distance and non-metric 2-D MDS primitives, and reproducing a frozen bundle's regime labels for
  colouring.
- **`plot_style.py`** — single source of truth for figure styling: rcParams (`set_publication_style`), the
  fixed regime/neutral colour palette (`REGIME_PALETTE`, ordered to match `REGIME_NAMES = ["Calm",
  "Transitional", "Crisis"]`), spine/tick styling (`style_axis`), crisis-event annotation (`CRISIS_EVENTS`),
  train/test split marker (`mark_split`), date-axis formatting, fixed-DPI export (`save_figure`). Adopted by
  all three regime-detection sweep scripts, all three walkthrough notebooks, `embedding_geometry_sweep.py`
  (partially), and the four `data_download_*.ipynb` notebooks — don't redefine these constants locally in a
  new script.

## Working conventions

- Stock/column identifiers are always `{SECTOR}_{TICKER}` (e.g. `IT_AAPL`); sector codes come from each
  `data_download_*.ipynb`'s own `INDICES` entry. `networks.load_logreturns` sorts columns by sector prefix.
- Comments in the library modules are one-line docstring-style summaries directly above each function, no
  multi-line docstrings and no comments inside class method bodies — match that style.
- Random seeds are fixed at 42 throughout (Graph2Vec, KMeans, PCA); `cluster_hmm` instead sweeps seeds
  `0..n_seeds` and keeps the best log-likelihood model, since HMM fitting is seed-sensitive. `gensim`'s
  Doc2Vec training is only approximately reproducible run-to-run even with a fixed seed (multi-threaded
  Hogwild-style SGD).
- Regime count is fixed at `k=3` (Calm/Transitional/Crisis).
- All three regime-detection sweep scripts are raw-arm only (`ARMS = ["raw"]`) and cover all four markets
  (`INDEX_CODES = ["SP500", "Nikkei225", "FTSE350", "CSI300"]`). Graph2Vec's grid additionally includes
  `graph_type = "threshold"` (plain threshold graph); eigenvector centrality's grid excludes it (Perron-
  Frobenius needs a connected graph, and plain threshold graphs fragment badly); spectral has no `graph_type`
  axis at all (it never constructs a graph). `embedding_geometry_sweep.py` sweeps `EPOCH_LENGTHS` and
  `GRAPH_TYPES` (all three of each) but stays fixed to one market/arm/descriptor cell (S&P 500, raw, Graph2Vec).
  Keep bundle key-naming (`kmeans_order`/`hmm_order`, not just `rank_of_cluster`) consistent in
  `modelling_sweep_raw.py`'s `.joblib` bundle schema if changed — `embedding_geometry_sweep.py` depends on it
  for every `(epoch_length, graph_type)` combo.
- Sweep-script plots colour the VIX curve by epoch length (`EPOCH_LENGTH_COLOURS`) and annotate historical
  crisis events on the VIX overlay (`CRISIS_EVENTS`); predicted-crisis shading stays crisis-red regardless of
  timescale. Dashboard diagnostics are saved at 150dpi via `sweep_common.savefig` (the three regime-detection
  scripts) or `embedding_geometry_sweep.py`'s own local `_savefig` (final manuscript figures use
  `plot_style.save_figure`'s 600dpi default).
- `outputs/` (network-data pickles, model bundles, results CSVs) and `images/` are pipeline artifacts —
  regenerable by re-running the relevant stage. `outputs/*.pkl` files are large (hundreds of MB to a few GB
  for `SP500`/`Nikkei225` at short epoch lengths); be mindful before loading multiple at once. In this
  checkout, only the stage-2 network pickles exist — see `README.md`'s "Current status".
