# COMPILED.md — Research Reconstruction for Dissertation Planning

This document reconstructs the project's actual research structure — not its file structure — as a bridge from codebase to dissertation narrative. It is not a technical reference (that's `CLAUDE.md`, `LOGIC.md`, `docs/`); it exists to answer: what did this project build, why, what was tested, what actually happened, what does it mean, and what belongs in the dissertation.

**Everything numeric in §1–§15 below is sourced from `outputs/modelling_sweep_results_raw.csv`, `outputs/eigencentrality_sweep_results_raw.csv`, `outputs/spectral_sweep_results_raw.csv`, and `outputs/embedding_geometry_results.csv` as they exist on disk right now** — modification times 2026-08-23 23:57 to 2026-08-24 01:50, postdating the full mid-session pipeline changes: `pr_corr_v1` reinstated into `spectral.py`'s feature set, the shared `sweep_common.py` extraction (which also fixed a gap where only the Graph2Vec branch reported HMM dwell time), and `embedding_geometry_sweep.py`'s restructure from a 3-descriptor/3-arm design to Graph2Vec-only/raw-arm-only swept over network construction. The user emptied `outputs/`/`images/` and rewound `sweep_run.py` from scratch after all of that landed, so every number below reflects the current code, not a stale partial run. §16–§17 are new sections sourced from two new, independent notebooks (`network_diagnostics.ipynb`, `portfolio_analysis.ipynb`) rather than the four CSVs — each says so explicitly. I opened the actual figure files referenced below, not just their filenames.

---

## 1. Project Overview

A network-science approach to financial regime detection: build rolling-window stock correlation networks for four equity markets, reduce each window to a fixed-length descriptor via three competing methods, cluster the descriptor sequence into three regimes, and check whether the detected "Crisis" regime lines up with real market stress. The project's actual question is narrower and sharper than "can regimes be detected from networks" — see §2.

**Current state, in one sentence**: the full experimental grid has been run — all three regime-detection branches across all four markets, all epoch lengths, and all graph types, plus the full embedding-geometry construction comparison — and the results below are real, current output, not projections. Two further, independent analyses now exist alongside the grid: a topology-only diagnostic (§16) and a financial-application backtest (§17).

## 2. Research Question and Objective

The surface question — can market regimes be read off correlation-network topology — is not really what the pipeline is built to test. The question the *design* actually answers is: **when you build an expensive, learned graph representation (Graph2Vec) instead of a cheap, deterministic one (eigenvector centrality, or a spectral/RMT summary that skips graph-building entirely), does the extra complexity buy anything?**

This is operationalised as a controlled 2×2 comparison, varying two independent axes while holding everything else fixed:

|                          | Input: filtered graph  | Input: raw correlation matrix |
|--------------------------|------------------------|--------------------------------|
| **Descriptor: learned**     | Graph2Vec               | *(not built)* |
| **Descriptor: handcrafted** | Eigenvector centrality | Spectral/RMT |

- Graph2Vec vs. eigenvector centrality isolates *descriptor complexity* (same graph, different reduction).
- Eigenvector centrality vs. spectral isolates *how much the sparse-graph step itself costs* (same descriptor style — both are "take an eigenvector of something" — different input).
- Graph2Vec vs. spectral differs on both axes and is only interpretable through the other two.

A fourth, structurally separate question is asked by the embedding-geometry branch (§8): does Graph2Vec's embedding *shape* stay stable across the three nested network constructions (threshold ⊂ MST-threshold ⊂ TMFG-threshold) built from the same raw correlation matrix?

Two more questions sit outside this 2×2 design entirely, each answered by its own standalone notebook rather than a sweep: §16 asks what the raw network *topology itself* — density, connectivity, clustering, cliques — looks like across market regimes, independent of any descriptor or clustering step; §17 asks whether the single best-performing regime-detection cell actually earns money as a trading signal, benchmarked against a cheap volatility trigger.

Full reasoning: `LOGIC.md` §1–3.

## 3. Current End-to-End Pipeline

The pipeline is **not a single straight line** — it is one shared preprocessing trunk feeding **three parallel, independent descriptor branches**, plus a fourth branch downstream of only one of them, plus two standalone analyses downstream of the trunk and the frozen bundles respectively. This is the single most important structural fact a generic "raw data → results" template would miss.

```text
Raw prices (4 markets, yfinance)
        │
        ▼
5-filter screening panel  (data_download_*.ipynb)
        │
        ▼
Rolling-window log-return correlation matrices  (networks.rolling_correlations)
        │
        ▼
Graph filtering: threshold | MST | TMFG   (networks.py, raw arm only)
        │
        ├──────────────┬──────────────────┐
        ▼              ▼                  ▼
   Graph2Vec    Eigenvector        Spectral/RMT
  (learned,     centrality         (no graph —
   graph input) (deterministic,    reads the raw
                 graph input)      correlation
                                    matrix directly)
        │              │                  │
        └──────────────┴──────────────────┘
                        │
                        ▼
        Shared clustering/ranking/validation  (modelling.py)
        K-means & Gaussian HMM → μ-ranked into Calm/Transitional/Crisis
        → VIX validation (S&P 500 only)
                        │
        ┌───────────────┴────────────────────┐
        ▼                                     ▼
Embedding-geometry branch (§8):        Portfolio backtest (§17):
Graph2Vec only, raw arm only,          one frozen K-means bundle
replays modelling_sweep_raw.py's       (SP500/raw/132d/MST) replayed
frozen bundles across threshold/       by inference only, turned into
MST/TMFG — no eigencentrality           a daily exposure schedule and
or spectral involved                   benchmarked against two controls

Network topology diagnostics (§16) sits outside this diagram entirely — it reads
data_processing.py's raw network pickles and VIX directly, upstream of every
descriptor/clustering step, and uses none of their output.
```

Only the raw arm is built at all: two other correlation transforms, `mg`/`gr`, were used earlier in the project for structural-robustness checks (§6) but were never swept through regime detection or the embedding-geometry branch, and have since been removed from `networks.py`/`data_processing.py` entirely. Full stage-by-stage detail: `docs/architecture/pipeline.md`.

## 4. Data and Preprocessing

Four markets: S&P 500, Nikkei 225, FTSE 350 (= FTSE 100 ∪ FTSE 250), CSI 300. Daily adjusted closes, 2001-01-01 to 2026-07-15, via `yfinance`. A five-filter screen (non-positive price → |log-return| > 0.8 jump → incomplete history → market-wide non-trading day → ≥63-day flat-return halt) reduces each scraped universe to a clean panel:

| Market | Scraped universe | Final N | Attrition |
|---|---|---|---|
| S&P 500 | 503 | 349 | 31% |
| Nikkei 225 | 225 | 177 | 21% |
| FTSE 100 | 100 | 58 | 42% |
| FTSE 250 | 250 | 101 | 60% |
| FTSE 350 (derived) | — | 159 | — |
| CSI 300 | 300 | **36** | **88%** |

CSI 300's attrition is severe and real, not a screening bug — 223 of 300 constituents fail the full-history requirement (many CSI 300 names listed on the Shanghai STAR Market, active only since 2019, or under ChiNext's current rules, in force since 2009 — well inside the 2001–2026 sample window), plus 33 more fail the halt screen. **This is worth stating plainly in the dissertation**: any CSI 300 result rests on a 36-stock panel, an order of magnitude smaller than the other three markets, which should visibly qualify how much weight CSI 300 findings can carry. Full derivation: `docs/architecture/pipeline.md` Stage 1; already written into `report/main.tex`'s Methodology → Data section this session.

## 5. Network Construction

Each window's correlation matrix is filtered into a sparse graph three ways: a plain correlation-threshold graph (no connectivity guarantee — measured at 43% mean node coverage on real SP500 data), an MST (N−1 edges, always connected), and a TMFG (3N−6 edges, always connected, denser). The threshold (`raw: 0.65`) was derived once in `threshold_diagnostics.ipynb` from a pooled 5%-exceedance target on **S&P 500 at T=132 only** and applied to all four markets and all three window lengths — a real, unverified external-validity assumption (see §15). Full math: `docs/math/graph-filtering.md`.

## 6. Network / RMT / Structural Analysis

Earlier in the project, two other correlation transforms existed specifically to test structural robustness, though neither was ever swept through regime detection: `mg` clipped the Marchenko–Pastur noise bulk to its own mean (RMT denoising); `gr` subtracted the dominant "market mode" eigenvalue/eigenvector. One concrete finding from that work: the `mg` arm's median MST-threshold edge count for S&P 500/T=132 was 349 — essentially exactly N−1=348, i.e. for the *typical* window the threshold-augmentation step added almost nothing and the graph collapsed to the bare MST. Both transforms, and the pickles that finding was computed from, have since been removed from the codebase (the analysis only ever depended on `raw`), so this can no longer be re-verified or extended without reinstating them.

## 7. Graph Representation and Embedding

Three descriptors, the heart of the 2×2 design (§2):

- **Graph2Vec** (`graph2vec.py`): Weisfeiler-Lehman relabelling + Doc2Vec (PV-DBOW), 128-d, `wl_iterations=3`, trained on train windows only, `infer_vector` for test. The real, current-grid OOV rate — the fraction of a window's structural features missing from the trained vocabulary — averages **48.1% in-sample (train) and 50.6% out-of-sample (test)** across all 36 combos. This is not a small caveat: roughly half of every graph's own structural fingerprint is dropped before it's even embedded.
- **Eigenvector centrality** (`eigencentrality.py`): the leading eigenvector of each window's binary adjacency matrix (MST/TMFG only — plain threshold graphs fragment too badly for Perron–Frobenius). Zero hyperparameters, deterministic. The interpretability feature `argmax_id` (the single most-central stock) is genuinely unstable: across the current 6-combo raw-arm grid, every market shows exactly 2 distinct modal answers (e.g. S&P 500: `FIN_BEN`/`UTL_LNT`).
- **Spectral/RMT** (`spectral.py`): **9** scalar features from the correlation matrix's own eigendecomposition, no graph at all — up from 8 earlier this session, when `pr_corr_v1` (the leading eigenvector's participation ratio) was reinstated by explicit request. Exactly reproducible (`np.linalg.eigh` has no learned component). `pr_corr_v1`'s reinstatement carries a documented caveat (`docs/math/spectral.md`): a prior redundancy audit found it substantially correlated with $\mu$/$\lambda_1$ (r=0.74–0.90), so it is back in the fitted PCA space but not a re-validated independent signal — see §15 for a concrete, current-data consequence of this.

Full math per method: `docs/math/graph2vec.md`, `docs/math/eigencentrality.md`, `docs/math/spectral.md`.

## 8. Embedding Geometry

**This branch's design changed mid-session** — it used to compare all three descriptors' embeddings across the `raw`/`mg`/`gr` arms; it now compares only Graph2Vec's embedding across the three nested network constructions (threshold ⊂ MST-threshold ⊂ TMFG-threshold), raw arm only, replaying `modelling_sweep_raw.py`'s frozen bundles rather than eigencentrality's or spectral's. Swept over all three epoch lengths (63/132/378d), S&P 500 only. Real, current stress values (lower = more trustworthy 2-D layout):

| Epoch length | Threshold | MST-threshold | TMFG-threshold |
|---|---|---|---|
| 63d | 0.375 | 0.399 | 0.411 |
| 132d | 0.363 | 0.390 | 0.401 |
| 378d | 0.334 | 0.371 | 0.381 |

Stress rises **monotonically threshold < MST < TMFG at every epoch length** — TMFG's embedding space is consistently the hardest of the three to lay out faithfully in two dimensions. OOV rate follows the identical ordering, also at every epoch length (e.g. at T=132: 35.7%/53.2%/69.8% train-window mean) — TMFG windows are also the most structurally novel relative to what Graph2Vec's vocabulary saw in training. Both are genuinely new, previously-undocumented findings from the current grid, not carried over from the old arm-comparison design. Full math and scope: `docs/math/embedding-geometry.md`.

## 9. Clustering and Regime Detection

Shared, embedding-agnostic machinery (`modelling.py`) used identically by all three branches — this is what makes cross-descriptor comparison fair rather than apples-to-oranges (`LOGIC.md` §7). K-means (`n_init=20`) and a Gaussian HMM (`n_seeds=20` restarts, causal Viterbi decoding on test) both cluster into $K=3$. Raw cluster labels are arbitrary; `fit_regime_order` fixes this once, on train μ only, by ranking clusters into Calm/Transitional/Crisis by mean off-diagonal correlation — the single design choice that makes "Graph2Vec's Crisis regime" and "eigenvector centrality's Crisis regime" comparable at all. $K=3$'s only justification anywhere in the codebase is a K-means elbow scan and an HMM AIC scan, both inline in `modelling_walkthrough.ipynb`'s S&P 500 section only — not reproduced for any other market or in any sweep script. Full math: `docs/math/regime-detection.md`.

**A concrete, current-data finding worth flagging**: a regime can vanish from the test period entirely (`*_regimes_absent_test > 0`). K-means: 2/36 Graph2Vec combos, 10/24 eigencentrality combos, 4/12 spectral combos. HMM: 9/36 Graph2Vec combos, 12/24 (exactly half) eigencentrality combos, 4/12 spectral combos. Half of eigencentrality's HMM fits losing a whole regime on test is a real, current-grid instability worth investigating, not a one-off.

## 10. Validation

`validate_regime` scores predicted-Crisis membership against a real series (VIX > 30) — but **only for S&P 500**; no other market has any validation series downloaded. This is the single biggest limitation on any cross-market claim (§15). For the other three markets, `empirical_transitions`' train-vs-test structural comparison is the only available check.

---

## 11. Experimental Programme

| Branch | Grid | Combos | Research question isolated |
|---|---|---|---|
| Graph2Vec (`modelling_sweep_raw.py`) | 4 markets × 3 epochs × {threshold, mst, tmfg} | 36 | Does a learned descriptor detect regimes at all, and how does graph-filtering choice affect it? |
| Eigenvector centrality (`eigencentrality_sweep_raw.py`) | 4 markets × 3 epochs × {mst, tmfg} | 24 | Does a zero-hyperparameter deterministic descriptor do as well, holding the graph input fixed vs. Graph2Vec? |
| Spectral/RMT (`spectral_sweep_raw.py`) | 4 markets × 3 epochs | 12 | Does skipping graph-filtering entirely (same descriptor *style* as eigencentrality, different input) cost real information? |
| Embedding geometry (`embedding_geometry_sweep.py`) | 1 fixed cell (S&P 500, raw, Graph2Vec) × 3 epochs × 3 constructions | 9 | Does Graph2Vec's embedding shape survive threshold → MST → TMFG filtering? |

Every regime-detection branch also varies **market** (4) and **window length** (63/132/378 trading days ≈ one quarter / half-year / eighteen months) as secondary axes — these aren't separate experiments so much as robustness dimensions layered onto the primary 2×2 design. All 81 regime-detection combos (36+24+12+9) now have real output on disk.

## 12. Results

All numbers below: S&P 500, the only market with VIX-based F1 (§10). K-means test-set F1, mean over each branch's S&P 500 combos (n = number of epoch×graph_type combos for that branch):

| Descriptor | Mean K-means F1 (n) | Range | Mean HMM F1 | Range |
|---|---|---|---|---|
| Graph2Vec | 0.312 (9) | 0.160–0.444 | 0.297 | 0.160–0.450 |
| Eigenvector centrality | 0.277 (6) | 0.111–0.400 | 0.141 | 0.000–0.244 |
| Spectral/RMT | **0.350** (3) | 0.167–0.471 | **0.000** | 0.000–0.000 |

**The headline finding does not have a single clean winner, and that itself is the finding.** On K-means, spectral — the cheapest, no-graph, no-learning descriptor — still has the highest mean F1 of all three, beating Graph2Vec despite Graph2Vec's far greater representational complexity. On HMM, the ranking fully inverts: Graph2Vec clearly leads (0.297), spectral now collapses **completely** — F1 = 0.000 for all three S&P 500 window lengths, not just two of three as in the previous grid (§13 has the detail). Which method "wins" depends on which clustering algorithm is paired with it — a genuinely reportable tension, not resolved by any single number, and if anything sharper now than before `pr_corr_v1` was reinstated.

Best individual S&P 500 K-means combo per method: Graph2Vec 132d/threshold (F1=0.444), eigencentrality 132d/mst (F1=0.400), spectral 63d (F1=0.471, spectral's best result overall).

**Cross-market structural comparison** (no F1 available — VIX is S&P 500-only): mean K-means crisis-cluster μ (Graph2Vec, raw arm) is broadly similar for S&P 500 (0.547), Nikkei 225 (0.533), and CSI 300 (0.557), but **FTSE 350 is a clear outlier at 0.347** — its detected "Crisis" regime has a mean correlation barely above the other markets' Transitional regimes. I looked at FTSE 350's regime timeline directly (`images/FTSE350_raw_132d_mst/kmeans_timeline.png`): it shows visibly noisier, more rapidly alternating regime shading in the early sample years than S&P 500's cleaner, more persistent regime blocks. This is a genuine, currently-unexplained cross-market difference worth a dissertation subsection of its own, not a footnote — it could reflect FTSE 350's derived-panel construction (§4, two independently-screened sub-indices concatenated) rather than a genuine market-structure difference, but that hasn't been tested. (These exact crisis-μ values drift by roughly a thousandth between reruns — Graph2Vec's Doc2Vec training is only approximately reproducible even at a fixed seed, documented in `docs/critical-evaluation.md` — the outlier finding itself is unaffected.)

**Embedding-geometry visual result**: the S&P 500 Graph2Vec MDS scatter, coloured by detected regime (`images/SP500_132d/geometry_mds_graph2vec_regime.png` — this path changed with the branch restructure, §8), shows a real, visually distinct Crisis cluster (red) occupying a separate region of the 2-D embedding space from Calm (green), with Transitional (blue) filling the space between — genuine geometric evidence that the regimes are not an artefact of the clustering step alone, now shown once per network construction rather than once per arm.

## 13. Negative / Failed / Degenerate Results

- **VGAE (Variational Graph Auto-Encoder)**, the natural fourth "learned + graph-free-eligible... learned + graph-based" cell attempt, was built, tested, and removed from the codebase entirely (recoverable from a prior git history reference, `git show 8903f5c:shelved/vgae.py`, if this project is ever put under version control). It collapsed to a bit-identical embedding across every window on TMFG-threshold graphs — a genuine, diagnosed (not just abandoned) failure. This is why the top-left cell of §2's 2×2 table is empty. Full account: `LOGIC.md`, final section.
- **HMM collapsing entirely on spectral/S&P 500**: F1 = 0.000 exactly for **all three** window lengths (63d, 132d, 378d) — the HMM's Crisis regime, when paired with the spectral descriptor, now systematically fails to align with VIX at every tested window length, not just two of three as in the pre-`pr_corr_v1` grid. Worth stating as a hypothesis, not a proven cause: reinstating a feature the original redundancy audit found substantially correlated with $\mu$ (r=0.74–0.90, §7) may have further destabilised the HMM's diagonal-covariance fit — plausible, not confirmed, and a natural thing to check if `pr_corr_v1` is ever reconsidered.
- **Eigenvector centrality's localisation gate firing on real data**: `pr_localised_flag = True` for CSI 300/raw/132d/mst (mean PR 9.04) and CSI 300/raw/378d/mst (mean PR 7.79) — both below the `mean_pr < 10` threshold, meaning the leading eigenvector has collapsed onto a small hub of stocks rather than spreading across the panel. This is not hypothetical or historical — it's live in the current CSI 300 results, and CSI 300's already-small 36-stock panel (§4) makes this more concerning, not less.

## 14. Main Findings

**Supported by current evidence**: correlation-network topology carries a real, visually and numerically detectable regime signal for S&P 500 — the MDS geometry (§12), the μ timeline shading aligning with GFC/Euro-debt/COVID (`images/SP500_raw_132d_mst/kmeans_timeline.png`), non-trivial F1 across all three descriptors, and now an entirely independent, descriptor-free corroboration from raw topology alone (§16) all point the same direction.

**Genuinely ambiguous, not resolved by this grid**: which descriptor is "best" — it depends on clustering algorithm, window length, and graph-filtering choice, with no descriptor dominating across all conditions. The fresh grid sharpens rather than resolves this: spectral's K-means lead over Graph2Vec still holds, but its HMM performance moved from "weak" to "exactly zero everywhere." This is arguably a more interesting and honest finding than a clean ranking would have been, and should be framed as such rather than smoothed over.

**Not established**: any cross-market claim beyond S&P 500 rests entirely on structural self-consistency (§10), not external validation; the FTSE 350 anomaly (§12) is observed but not explained; CSI 300's results rest on a 36-stock panel; the `ARM_THRESHOLDS` values (§5) were never checked against any market but S&P 500; and — new this session — whether the single best-performing regime-detection cell translates into a profitable trading rule at all (§17 answers this directly, and the answer is no).

## 15. Methodological Limitations

From `docs/critical-evaluation.md`, cross-checked against the fresh full-grid data — each item below is now either *demonstrated on current data* or remains *historical/theoretical*:

- **Demonstrated on current data**: the localisation gate's weakness (§13, live in CSI 300); regimes vanishing on test (§9, live in half of eigencentrality's HMM fits); OOV rate (§7, 48–51% average, current full grid); spectral+HMM's total collapse on S&P 500 (§13), newly total rather than partial since `pr_corr_v1` was reinstated (§7) — a concrete, current-data instance of that feature's documented redundancy caveat actually mattering downstream, not just a theoretical concern.
- **No longer reproducible**: the `mg`-arm edge-count collapse (§6) was demonstrated from a real pickle earlier in the project, but the `mg`/`gr` transforms have since been removed from the codebase, so it can't be re-verified without reinstating them.
- **Still historical/theoretical, not re-verified on the current 4-market grid**: HMM's conditional-independence violation from ~92–94% window overlap; PCA's sample:dimension ratio risk in high-`pca_dims` combos; the single, untested 2019-12-31 train/test cutoff; F1 instability on small positive-class samples (this *is* visible in the current S&P 500 spread — F1 ranging 0.16 to 0.47 for the same descriptor across window lengths — but no formal significance test has been run on the current grid, and §17's own backtest deliberately avoids one for the same small-sample reason).
- **New since the 4-market widening**: `ARM_THRESHOLDS` (§5) is now applied to three markets it was never calibrated against; the FTSE 350 crisis-μ anomaly (§12) is a new, unexplained observation.

## 16. Network Topology (`network_diagnostics.ipynb`)

A standalone notebook, independent of every sweep script — no clustering, no embedding, no sweep-script output anywhere in it. It examines the raw topology of the three graph constructions directly (S&P 500, raw arm, T=132), across Calm/Transitional/Crisis periods selected from **VIX alone** (highest/lowest/closest-to-22.5 end-of-window VIX), not from $\mu$ or any clustering step — an intentionally independent corroboration of the regime story rather than a restatement of it.

- **Density converges in Crisis despite very different Calm baselines**: Threshold/MST-threshold/TMFG-threshold sit at 0.005/0.009/0.019 in Calm (TMFG ~4× denser by construction) but all three land at 0.214–0.216 in Crisis.
- **Fragmentation is Threshold-specific**: the plain Threshold graph has 250 separate components in Calm and still 32 in Crisis; MST-threshold and TMFG-threshold are always exactly 1 component, by construction — this independently reproduces, from a completely different data path, the same connectivity argument `docs/math/eigencentrality.md` uses to justify restricting `eigencentrality_sweep_raw.py` to `["mst", "tmfg"]`.
- **Small-world collapse where connectivity is guaranteed**: MST-threshold's average shortest path on its own largest component falls from 12.5 (Calm) to 2.0 (Crisis), diameter 31→5; TMFG-threshold similarly, 7.5→1.9 and 20→4. The plain Threshold graph does *not* fall monotonically (1.8→3.4→1.9), because which stocks are even in its largest component changes every period.
- **Degree assortativity flips sign**: 0.57–0.66 (Calm, sector hubs bonding with sector hubs) to −0.09 (Crisis, flat market-wide comovement) for Threshold and MST-threshold.
- **Sector structure is what Crisis actually destroys**: normalised mutual information between detected communities (greedy modularity) and true GICS-style sectors falls monotonically Calm→Crisis for all three constructions (0.61→0.29, 0.54→0.21, 0.47→0.24), and the full 629-window time series dips sharply at every annotated historical crisis event and recovers in between — the cleanest, most consistent finding in the notebook, and a topology-level analogue of Mantegna's classical MST sector-clustering result.
- **Clique enumeration itself becomes evidence**: maximal-clique counts stay in the low hundreds in Calm/Transitional across all three constructions, then blow through a 20,000-clique enumeration cap in *every* Crisis snapshot, for *every* construction — direct combinatorial confirmation of near-complete-graph collapse under stress, not just a denser sparse graph.
- **A closing hypothesis, explicitly not a verified claim**: reasoned from `graph2vec.py`'s own WL-hashing mechanics (degree-rooted, 3-hop, discrete tokens) rather than from any sweep result, the notebook argues Threshold/MST-threshold hand Graph2Vec a genuine discrete *shape*-level phase transition to key on, while TMFG-threshold's Crisis-invariant clustering coefficient (~0.73 throughout, unlike the others' 4× jump) means its signal rides mostly on degree *scale* — a form of change a non-proportional hashing scheme represents more diffusely. Confirming this against actual detection performance would need the sweep scripts' own embeddings, deliberately out of scope for this notebook.

Figures: `images/network_diagnostics/network_structure_by_period.png` (3×3 construction×period grid, sector-coloured, fixed node positions per row), `topology_timeseries.png` (density/clustering/NMI, all 629 windows), `degree_distribution_by_period.png`, `clique_statistics.png`.

## 17. Financial Application (`portfolio_analysis.ipynb`)

The dissertation's §4.4: a single, pre-nominated backtest cell — S&P 500, raw arm, T=132, MST-threshold, Graph2Vec, **K-means only** (the HMM's Viterbi decoding isn't usable for a live daily exposure decision without replacement by forward filtering, so it's excluded on principle, not convenience) — turning the frozen `outputs/models/SP500_raw_132d_mst.joblib` bundle into a daily exposure schedule via inference-only replay, no refitting. Test period: 2020-01-07 to 2026-07-06.

Three equal-weighted strategies, differing only in daily equity exposure: buy-and-hold (100% always); regime-timed (100%/50%/0% by detected Calm/Transitional/Crisis, one trading day's execution lag applied throughout); and a volatility-triggered control (0% when trailing 21-day realised vol exceeds its 80th percentile, calibrated on train-period data only) — included specifically so any apparent edge from the network pipeline can be checked against what a much simpler scalar signal achieves on its own.

| Strategy | CAGR | Ann. vol | Sharpe (excess) | Max drawdown | Calmar |
|---|---|---|---|---|---|
| Buy-and-hold | 15.64% | 20.48% | 0.68 | −37.25% | 0.42 |
| Regime-timed | 5.87% | 10.74% | 0.33 | −17.31% | 0.34 |
| Vol-triggered | 10.56% | 12.50% | 0.65 | −16.48% | 0.64 |

**Verdict, against the criterion fixed before any number was computed (exceed both benchmarks on Sharpe *and* max drawdown, surviving COVID's exclusion): FAILS.** The regime-timed strategy has the *worst* Sharpe of the three, and a worse (not better) max drawdown than the much simpler vol-triggered control — stated first and plainly, per this document's own results-discipline (§18), not qualified before being stated.

- Of regime-timed's +44.9% cumulative gross return, +7.8 percentage points is risk-free carry earned while sitting in cash, not equity-timing skill (vol-triggered: +3.3pp of +91.9%) — a real but modest fraction of the total, computed by comparing each strategy against a version of itself with the risk-free leg zeroed out.
- The detected COVID "Crisis" episode runs 190 days, 2020-03-19 to 2020-12-16 — the crash *and* most of the V-shaped recovery. Buy-and-hold returns +64.9% across those exact days; regime-timed, having been at 0% exposure through most of them, returns +0.4%. This is most of why the strategy underperforms so badly, and it is a finding about the *regime detector's persistence*, not a backtest artefact.
- Excluding COVID entirely, regime-timed's Sharpe (0.35) edges narrowly past buy-and-hold's (0.34) but still trails vol-triggered's (0.73) by a wide margin — the exclusion doesn't rescue the headline verdict.

No significance test is computed anywhere in this analysis — the effective sample is a handful of crisis episodes, not the count of trading days, and a p-value on daily returns here would misrepresent the evidence; a block bootstrap (132-day blocks, matching the window overlap) is used only for an interquartile dispersion estimate. Figure: a three-panel equity-curve/drawdown/rolling-Sharpe chart, display-only, never written to disk.

## 18. Figures and Tables

Real inventory from `images/` (not an aspirational list):

| Figure type | Produced by | What it shows | Answers |
|---|---|---|---|
| `*_timeline.png` | all 3 regime-detection scripts, every combo | μ time series with Calm/Transitional/Crisis shading, train+test continuous | Does the detected regime sequence look sensible against known history? |
| `*_vix_overlay.png` | all 3 scripts, S&P 500 only | VIX with predicted-crisis shading, crisis events annotated | Does the detected Crisis regime line up with real volatility spikes? |
| `*_transition_comparison.png` | all 3 scripts, every combo | Train vs. test empirical transition matrices, K=3×3, annotated with occupancy | Is regime persistence/transition structure stable from train to test? |
| `geometry_selfsimilarity.png` | embedding geometry, 1 per epoch length | 1×3 row (threshold/MST/TMFG) of window-to-window cosine-distance heatmaps, shared colour scale | Does embedding-space self-similarity change across network constructions? |
| `geometry_mds_graph2vec_{date,regime,edgecount,oov}.png` | embedding geometry | 2-D MDS scatter, one construction per column | Is the embedding space's shape stable across constructions, and does it separate by regime? |
| `network_structure_by_period.png`, `topology_timeseries.png`, `degree_distribution_by_period.png`, `clique_statistics.png` | `network_diagnostics.ipynb` (§16) | Raw network topology across Calm/Transitional/Crisis, all three constructions | Does topology itself — no descriptor, no clustering — carry a regime signal? |

`portfolio_analysis.ipynb` (§17) produces one three-panel figure (equity curve / drawdown / rolling Sharpe) that is deliberately never written to disk — display-only, per that notebook's own scope.

Data-summary table (§4) is already written into `report/main.tex`. No results table currently exists in `report/main.tex` — §12's tables above are the direct source for one, and §16/§17 are entirely new material with no `main.tex` counterpart yet.

## 19. Current Execution / Reproducibility State

**Complete, on current code, verified by file timestamp**: data acquisition and screening (all 4 markets), network construction (36 pickles, 4×3×3), all three regime-detection sweeps (72 combos total, rerun from empty this session after the `pr_corr_v1`/`sweep_common.py`/dwell-time changes landed), the full embedding-geometry run (9 rows, new construction-swept design). `network_diagnostics.ipynb` and `portfolio_analysis.ipynb` (§16, §17) are both new this session and independently reproducible — the former from disk pickles plus VIX, the latter from the frozen SP500/raw/132d/mst bundle plus a live `^IRX` download, now guarded by sanity assertions (added this session after a transient bad download once produced silently-wrong headline numbers with no raised exception) so a corrupted download fails loudly instead of shipping wrong results.

**Not done**: a second train/test cutoff (robustness check, §15); a formal significance test across the F1 grid; re-verification of `ARM_THRESHOLDS` against any market besides S&P 500; the K-means elbow/HMM AIC scan extended to any market besides S&P 500; the portfolio backtest (§17) extended to any other cell of the main grid (deliberately — it was pre-nominated specifically so the result couldn't be a best-of-thirty-six selection).

**Recommended order if anything needs re-running**: `data_processing.py` (only if data changes) → the three `*_sweep_raw.py` scripts, any order → `embedding_geometry_sweep.py` last (depends on `modelling_sweep_raw.py`'s frozen `.joblib` bundles only). `sweep_run.py` already sequences this correctly. `network_diagnostics.ipynb` only needs Stage 2's pickles; `portfolio_analysis.ipynb` only needs `modelling_sweep_raw.py`'s SP500/raw/132d/mst bundle specifically — neither needs a full sweep rerun to reproduce.

## 20. Proposed Dissertation Narrative

The project's honest story is: *building an expensive, learned graph embedding is not obviously worth it over much cheaper alternatives — but "not obviously worth it" is itself a finding, not a failure, and it depends on exactly which clustering algorithm and window length you pair each descriptor with.* The strongest, most defensible single result is the S&P 500 evidence base (§12, §14, and now §16's fully independent topology corroboration) — visual, numerical, structural, *and* topological evidence converging on "regimes are real and detectable," with the *descriptor comparison* being the nuanced, no-clean-winner part of the story. §17 adds a second honest, negative result: detecting regimes is not the same as trading them profitably, and the dissertation should say so plainly rather than treat the backtest as a formality. The FTSE 350 anomaly and CSI 300's data-quality constraint are honest limitations that strengthen rather than weaken the dissertation if stated plainly, per `DISSERTATION_FORMAT.md`'s results-discipline rules (§8 there: no unsupported ranking claims, explicit in-sample/out-of-sample distinction, negative results reported not hidden).

## 21. Recommended Dissertation Structure

UCL's suggested 5-chapter shape (Introduction / Literature Review / Methodology / Results / Conclusions) is `[UCL-REC]`, not mandatory (`report/DISSERTATION_FORMAT.md` §7) — but nothing about this project's actual shape argues for deviating from it; the deviation risk here is generic-template Results/Methodology chapters that don't reflect the branch structure in §3. Proposed section-level structure, cross-referenced against `report/main.tex`'s current skeleton:

| Section | Research question | Methods | Experiments | Figures/tables | Status in `main.tex` |
|---|---|---|---|---|---|
| Methodology → Data | What markets, what screening? | §4 | — | Data-summary table | **Drafted** this session |
| Methodology → Network construction | How are correlation networks built and filtered? | §5, §6 | — | One example heatmap/graph (schematic) | Not drafted |
| Methodology → Descriptors | What are the three methods being compared? | §7 | — | — | Not drafted |
| Methodology → Clustering/validation | How are regimes detected and scored? | §9, §10 | — | — | Not drafted |
| Results → Descriptor comparison | Which descriptor detects regimes best, and does it depend on conditions? | — | §11 (36+24+12 combos) | §12's F1 table, timeline, VIX overlay | Empty |
| Results → Cross-market structure | Does regime structure generalise beyond S&P 500? | — | §11 | Cross-market crisis-μ table, FTSE 350 anomaly discussion | Empty |
| Results → Embedding geometry | Does Graph2Vec's embedding shape survive threshold → MST → TMFG filtering? | — | §11 (9 combos) | Self-similarity grid, MDS scatters | Empty |
| Results → Network topology | Does raw topology carry a regime signal independent of any descriptor? | §16 | — | Topology figures (§18) | Empty |
| Results → Financial application (§4.4) | Does the best-performing cell translate into a profitable trading rule? | §17 | — | Equity-curve/drawdown/Sharpe figure, metrics table | Empty |
| Conclusions | What did this establish, and what's next? | — | — | — | Empty |

Introduction and Literature Review are currently empty `\iffalse` stubs in `main.tex` (a fuller draft exists in `report/`'s own git history, commit `118b218`, if that was removed unintentionally — see `report/TEMPLATE_MANIFEST.md`). This document (§2, §20) gives the material to redraft the Introduction's hypothesis statement in the instructions' own suggested form ("The hypothesis of this study is that...").
