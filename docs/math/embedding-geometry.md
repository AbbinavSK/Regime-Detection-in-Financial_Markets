# Embedding geometry: does Graph2Vec's shape change across network constructions?

Implemented in `embedding_geometry.py` (pure calculation, no plotting) and `embedding_geometry_sweep.py` (orchestration + all plotting for this branch).

This is not a regime-detection branch (§1-8 of `LOGIC.md`); see [`../../LOGIC.md`](../../LOGIC.md#9-a-fourth-lens-embedding-geometry-across-network-constructions) for why it exists and how it relates to the three descriptor branches. In one line: is Graph2Vec's embedding space's *shape* stable across the three nested network constructions (threshold, MST-threshold, TMFG-threshold) built from the same raw-arm correlation matrix?

## Scope: one fixed cell, two swept axes

Unlike `modelling_sweep_raw.py`/`eigencentrality_sweep_raw.py`/`spectral_sweep_raw.py`, this branch does **not** sweep `index_code`/arm/descriptor/metric — all four are fixed:

```text
INDEX_CODE = "SP500"    # module constant, no CLI override
# arm is hardcoded to "raw" throughout -- not a parameter anywhere in the module
# descriptor is Graph2Vec only -- eigencentrality and spectral are not used by this branch
```

Two axes are swept: `EPOCH_LENGTHS = [63, 132, 378]`, CLI-overridable via `--epoch-lengths`, and `GRAPH_TYPES = ["threshold", "mst", "tmfg"]`, CLI-overridable via `--graph-types` — nine `(epoch_length, graph_type)` combos by default, each producing its own row and its own figure set (constructions are compared *within* one epoch length, not across epoch lengths). There is no `--metric` flag — the distance metric is fixed to cosine (`embedding_geometry.distance_matrix` hardcodes `metric="cosine"`; there is no euclidean path). MDS is likewise fixed to 2 components (`fit_mds` hardcodes `n_components=2`) — there is no 3-D variant. `--quick` smoke-tests the `T=132`/`mst` combo only, reusing the frozen Graph2Vec bundle, fitting nothing fresh.

All three constructions are includable here, including plain `threshold`, unlike `eigencentrality_sweep_raw.py`'s `["mst", "tmfg"]`-only grid (see [`eigencentrality.md`](eigencentrality.md)) — that restriction exists purely because eigenvector centrality needs a connected graph (Perron-Frobenius), and this branch does not use eigenvector centrality at all. The graph-key resolution is therefore conditional, not the unconditional `f"{graph_type}_threshold_graphs"` pattern a `mst`/`tmfg`-only grid can get away with: `"threshold_graphs" if graph_type == "threshold" else f"{graph_type}_threshold_graphs"`, matching `modelling_sweep_raw.py`'s own resolution exactly (see [`../architecture/pipeline.md`](../architecture/pipeline.md) for why the unconditional form is unsafe once `threshold` is a real value).

Extending `INDEX_CODE`/the distance metric/the arm/the descriptor set into swept dimensions remains a loop-structure change, not new methodology — the underlying maths in `embedding_geometry.py` doesn't depend on any of them being fixed. `_load_split` is called once per `epoch_length` (each length is a genuinely different pickle, not a subset of another), then reused across that length's `graph_type` loop, and freed before the next `epoch_length` loads — the same memory discipline the rest of the pipeline applies to these pickles (see `CLAUDE.md`).

## Vector per window

Graph2Vec's full 128-d embedding *before* PCA (`embedding_geometry.load_graph2vec_raw`, replaying a frozen bundle's already-fitted model) — not the PCA-reduced `X` the regime-detection branch clusters on. No library-module changes were needed to expose it: `get_embedding()`/`infer()` already return the raw embedding directly.

## Preparing the vector for distance/MDS — a different question from regime detection

`embedding_geometry.prepare_graph2vec(raw)` L2-normalises each row, so distance measures *shape* rather than vocabulary-coverage-driven magnitude (Graph2Vec's raw embedding norm can be affected by how much of a window's WL-vocabulary survived `min_count`, not just by genuine graph structure — see [`graph2vec.md`](graph2vec.md)'s OOV discussion). This is deliberately different from `modelling.py`'s `fit_embed`, which standardises and PCA-reduces instead.

Distances are computed on this prepared vector directly (`distance_matrix`, `sklearn.metrics.pairwise_distances(vectors, metric="cosine")`), **never in PCA space** — a PCA-space MDS would just redraw a scatter that already exists elsewhere in the pipeline (the regime-detection branch's own PCA plots), not add information.

## Distance metric and MDS

Cosine only (`embedding_geometry.distance_matrix` hardcodes `metric="cosine"`; there is no `--metric` flag and no euclidean path in the current code). On unit-norm vectors, $\|a-b\|^2 = 2(1-\cos)$ is a monotone transform of cosine distance, so a euclidean comparison would be largely redundant here regardless.

`fit_mds(D)` runs non-metric MDS (`sklearn.manifold.MDS(n_components=2, metric=False, dissimilarity="precomputed", random_state=42, n_init=8, normalized_stress=True)`) and returns 2-D coordinates plus Kruskal stress-1, normalised to roughly $[0, 1]$ so it's comparable across constructions and window counts (raw SMACOF stress would scale with both $n^2$ and the distance matrix's own units). As a manuscript-caption convention rather than an enforced code threshold: above ~0.2, a layout shouldn't be read as reliable geometry.

## Regime labels: loaded, not fitted

No clustering is fitted anywhere in this branch. K-means labels are loaded from `modelling_sweep_raw.py`'s frozen bundle for each `graph_type` (`outputs/models/{tag}.joblib`) via `embedding_geometry.raw_arm_regime_labels`, replayed through `modelling.apply_regime_order` — never reimplemented. Unlike when this branch swept `mg`/`gr` arms, every `graph_type` now gets a real, non-degenerate regime ranking (all three constructions are built on the raw arm, whose $\mu$ is a genuine, non-null crisis proxy), so there is no special-casing left: the regime-coloured MDS figure plots all three constructions, not a single restricted case.

**A subtlety worth being explicit about, since it's easy to get backwards**: a frozen bundle's `kmeans_model` was fit on **post-PCA** space using the **exact same raw-vector representation** the original regime-detection sweep used — Graph2Vec's *unnormalised* raw embedding. This branch's own prepared vector (L2-normalised) is a *different* representation used only for distances/MDS. `raw_arm_regime_labels(bundle, raw_test, n_train_live)` therefore takes the *un-prepared* raw vector, not the prepared one — every call site in `embedding_geometry_sweep.py` passes `test_raw`, never `prepared`. Passing the wrong one would silently feed a frozen `StandardScaler` a vector in the wrong space.

`raw_arm_regime_labels` also checks the raw vector's column count against `bundle["scaler"].n_features_in_` before calling `.transform()`, and checks the frozen bundle's `kmeans_model` was fit on the same number of train windows as the live split — both raise a clear, actionable error (naming a likely cause: a stale bundle, or stage 3 not having been re-run since `data_processing.py` last changed) rather than letting a cryptic sklearn dimension-mismatch error surface.

## Dependency on stage 3

Running `embedding_geometry_sweep.py` requires `modelling_sweep_raw.py` to have already been run for the matching `(SP500, epoch_length, graph_type)` combo, for every `(epoch_length, graph_type)` requested — it loads that script's frozen `.joblib` bundles rather than fitting anything. `eigencentrality_sweep_raw.py` and `spectral_sweep_raw.py` are not dependencies of this branch at all, since neither descriptor is used here. Requesting `--epoch-lengths 63 --graph-types threshold` therefore needs `SP500_raw_63d_threshold.joblib` on disk specifically; the default (all nine combos) needs all nine of `SP500_raw_{63,132,378}d_{threshold,mst,tmfg}.joblib`. Since `modelling_sweep_raw.py`'s own grid already sweeps all three epoch lengths and all three graph types for every market, a full run of that script satisfies this branch's dependency completely. `sweep_run.py` sequences this script last, after the three raw sweeps, for exactly this reason.

## Outputs

`outputs/embedding_geometry_results.csv` — one row per `(epoch_length, graph_type)` (`descriptor` is recorded as the fixed `"graph2vec"`, `arm` as the fixed `"raw"`, `metric` as the fixed `"cosine"`): `n_windows`, `n_train`, `stress` (2-D Kruskal stress-1), off-diagonal distance min/median/max, `mean_oov_rate`/`median_oov_rate`, `mean_edge_count`/`median_edge_count`. Writes are merged into the existing CSV by exact `(epoch_length, graph_type)` key, so a scoped `--epoch-lengths`/`--graph-types` re-run only replaces the rows it actually reproduces — every other already-written row is left untouched (falls back to a full overwrite, with a printed warning, if the on-disk file's schema doesn't match the current run's columns). The key is `(epoch_length, graph_type)`, not `graph_type` alone — the same `graph_type` recurs once per `epoch_length`, and only the pair together is unique.

`outputs/{index_code}_raw_{epoch_length}d_{graph_type}_graph2vec_cosine_distance.pkl` — the distance matrix for one `(epoch_length, graph_type)` combo, one file per combo (these are genuinely per-combo artifacts, not compared side by side in a single file the way the figures below are).

Figures under `images/{index_code}_{epoch_length}d/`, one such folder per `epoch_length` (no arm or graph_type suffix within it, since `graph_type` is the axis compared *within* each figure rather than the thing a folder is named after): `geometry_selfsimilarity.png` (one self-similarity heatmap panel per construction, one shared colour scale and colourbar, train/test split marked on both axes via `plot_style.mark_split`) and `geometry_mds_graph2vec_{date,regime,edgecount,oov}.png` — non-metric 2-D MDS scatter panels, one construction per column, all four colourings covering every construction (no restricted case remains, now that arm is always raw). Running the default nine-combo sweep therefore produces three independent folders (`SP500_63d/`, `SP500_132d/`, `SP500_378d/`), each with its own five-figure set comparing constructions within that epoch length — never a figure comparing epoch lengths against each other.

Each `(epoch_length, graph_type)` combo is isolated — if one fails (`main` catches the exception and prints which epoch_length/graph_type failed), that combo is skipped entirely, but every other successfully-collected combo's output is still written, and each epoch_length's comparison figures are built across whichever constructions succeeded for it.
