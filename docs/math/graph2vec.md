# Graph2Vec: turning a graph into a fixed-length vector

Implemented in `graph2vec.py` (self-contained, no `karateclub` dependency), driven by `modelling.fit_embed`/`transform_embed`. The problem: we have one graph per rolling window and need a fixed-length vector per graph to cluster — but graphs are variable-size, unordered structures with no natural vector representation. Graph2Vec (Narayanan et al., 2017) solves this by treating each graph as a *document* made of structural *words*, then reusing an off-the-shelf text-embedding model unmodified.

## Step 1 — seed node labels

`modelling.prepare_graph` relabels nodes as consecutive integers and seeds each one's `"feature"` attribute with its own degree: `int(G.degree(node))`. This gives every node a cheap, purely structural starting label before any relabeling happens.

## Step 2 — Weisfeiler-Lehman relabeling

`_wl_graph_features(G, wl_iterations)` implements the classic Weisfeiler-Lehman (WL) graph-isomorphism-test update rule. Starting from the degree labels above, each iteration recomputes every node's label as a hash of *its current label plus its neighbors' current labels, sorted*:

```python
neighbour_labels = sorted([labels[n] for n in G.neighbors(node)])
raw = labels[node] + "_" + "_".join(neighbour_labels)
new_labels[node] = str(zlib.crc32(raw.encode()))
```

After $k$ iterations, a node's label is a compact fingerprint of its entire $k$-hop neighborhood structure — two nodes (anywhere, in the same or different graphs) get identical WL-labels at iteration $k$ if and only if their $k$-hop neighborhoods are structurally identical (same recursive pattern of labels), which is exactly the WL kernel's definition of substructure similarity. `modelling.fit_embed`/`transform_embed` both run this for `wl_iterations=3`.

All labels from *every* round — the initial degree labels plus all 3 WL iterations — are pooled into one bag: a graph's "document" is the multiset union of its degree-labels ∪ 1-hop ∪ 2-hop ∪ 3-hop WL-labels. This mixes fine-grained (degree) and increasingly coarse/global (WL) structural signal in one bag-of-words.

## Step 3 — Doc2Vec (PV-DBOW)

Each graph's WL-word bag becomes a `gensim` `TaggedDocument`, and `gensim.models.Doc2Vec` learns one `dimensions`-length vector per document — this is exactly the same model used for paragraph/document embedding in NLP, just fed structural hashes instead of real words. Two configuration choices make this a genuine bag-of-words, matching the fact that WL-labels have no meaningful order:

- `dm=0` — **PV-DBOW** (Distributed Bag-of-Words), not PV-DM. The model predicts a document's words directly from the document vector, without also learning word-order-sensitive context windows.
- `window=0` — no local word-order context is used at all.

Training learns a `dimensions`-dim vector per graph (128, per `modelling.fit_embed`) such that graphs with similar WL-word bags — i.e. structurally similar graphs — end up with similar embedding vectors.

## Fitting vs. inferring: the train/test asymmetry

`modelling.py`'s causal split means Graph2Vec is fit once, on train windows only, and must separately handle test windows it never saw:

- `fit_embed(graphs)` calls `Graph2Vec.fit`, which builds the Doc2Vec vocabulary from train windows' WL-word bags (subject to `min_count=5` — a WL-hash string that doesn't recur at least 5 times across the whole train set never enters the vocabulary) and learns each train window's embedding vector directly as a side effect of training.
- `transform_embed(graphs, g2v, ...)` calls `Graph2Vec.infer`, which embeds *unseen* (test) graphs into that same fixed vocabulary/vector space via `Doc2Vec.infer_vector` — a separate, approximate gradient-based optimisation against the frozen model, not a simple lookup. This is a materially different, noisier operation than what produced the train embeddings: any WL-feature in a test window that isn't in the train vocabulary is silently dropped from that inference rather than imputed.

`Graph2Vec.oov_rate(graphs)` quantifies exactly this: for a set of graphs, the fraction of each graph's WL features absent from `self.model.wv.key_to_index`. `transform_embed` reports this per test window (mean logged as `oov_rate_test_mean` in `modelling_sweep_raw.py`'s sweep rows); `run_combo` additionally reports `oov_rate_train_mean` by running `oov_rate` over the *train* graphs against their own just-fitted vocabulary. This isn't a small effect: on one historical `SP500/raw/63d/threshold` run, `oov_rate_train_mean = 0.41` and `oov_rate_test_mean = 0.48` — i.e. **even in-sample**, on average 41% of a train window's own WL-hash features aren't frequent enough to make the `min_count=5` vocabulary cut. This is a real limitation, not just a generalisation caveat — see [`../critical-evaluation.md`](../critical-evaluation.md).

## After embedding: scaling + PCA

Both `fit_embed` and `transform_embed` finish with the same `StandardScaler`→`PCA` pattern, split across the train/test boundary the same way as the embedding itself:

```python
# fit_embed (train only)
scaler = StandardScaler().fit(embeddings)
pca = PCA(n_components=0.95, random_state=42).fit(scaler.transform(embeddings))

# transform_embed (test, using the already-fitted scaler/pca)
X = pca.transform(scaler.transform(embeddings))
```

`StandardScaler` puts all 128 embedding dimensions on comparable scales (Doc2Vec dimensions aren't naturally unit-variance). `PCA(n_components=0.95)` then keeps only as many principal components as needed to retain 95% of *train* variance — this denoises the embedding and reduces dimensionality for K-Means/HMM (see [`regime-detection.md`](regime-detection.md)) while discarding very little information, at the cost that the retained dimensionality is only ever validated against train's own variance structure, not test's (see [`../critical-evaluation.md`](../critical-evaluation.md)'s note on PCA sample:dimension ratio).

## Weisfeiler-Lehman hash: why `zlib.crc32`, not Python's `hash()`

`_wl_graph_features` hashes each WL-label string with `zlib.crc32(raw.encode())` rather than Python's built-in `hash()`. This isn't a style choice — Python randomizes `hash()` for strings by default (`PYTHONHASHSEED`, since PEP 456), so the same label text maps to a *different* hash value in every fresh process, even with every other seed in the project fixed at 42. That would silently change the WL-label vocabulary — and therefore Graph2Vec's entire input vocabulary — between runs, undermining the project's seed=42 reproducibility convention at the one place it matters most. `zlib.crc32` is a fixed, deterministic CRC with no process-level randomization: the same label string always hashes to the same value, on any run, on any machine.

## Fixed hyperparameters and a remaining reproducibility caveat

`modelling.fit_embed` always calls Graph2Vec with `dimensions=128, wl_iterations=3, epochs=100, seed=42`. Even with the WL-hash deterministic, `gensim`'s Doc2Vec/Word2Vec training is still only *approximately* reproducible once `workers > 1` (the default here is 4) — multi-threaded Hogwild-style SGD means thread scheduling affects the exact result. This is a known `gensim` property, not a bug, and explains the (small, but present) run-to-run variance in downstream cluster labels and VIX-validation F1 scores even with every seed fixed.

## Consumers beyond regime detection

`embedding_geometry_sweep.py` also uses Graph2Vec directly, on the raw arm only: for each `graph_type` it replays `modelling_sweep_raw.py`'s frozen bundle for that combo's already-fitted model (`embedding_geometry.load_graph2vec_raw`) rather than refitting. It works from the *raw*, pre-PCA 128-d embedding, then L2-normalises it (`embedding_geometry.prepare_graph2vec`) before computing distances — a different preparation from the StandardScaler→PCA pipeline described above, because that branch asks a shape-comparison question, not a clustering one. See [`embedding-geometry.md`](embedding-geometry.md).
