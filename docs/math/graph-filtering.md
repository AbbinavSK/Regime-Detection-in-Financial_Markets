# Graph filtering: Mantegna distance, threshold, MST, TMFG

Every correlation matrix is a fully connected graph with $\binom{N}{2}$ edges — far too dense to embed or interpret directly. `networks.py` filters it down to a sparse graph three ways: a plain correlation threshold, a Minimum Spanning Tree, and a Triangulated Maximally Filtered Graph, the latter two also keyed on the same distance transform.

## Mantegna distance

Correlation isn't a metric — higher values mean "more similar," which is backwards for graph algorithms (MST, shortest paths, etc.) that expect *lower* values to mean "closer." Mantegna's (1999) transform converts correlation into a proper Euclidean-embeddable distance:

$$D_{ij} = \sqrt{2(1 - C_{ij})}$$

This maps $C_{ij}=1$ (perfectly correlated) to $D_{ij}=0$ and $C_{ij}=-1$ (perfectly anti-correlated) to $D_{ij}=2$, and satisfies the triangle inequality — which is what makes MST/shortest-path algorithms on it meaningful. It's computed once in `build_arm_networks` and reused by MST/TMFG below; the plain threshold graph doesn't need it for edge selection (that's decided on raw correlation) but still stores it as each edge's `weight`, for consistency with the other two graph types.

## Plain threshold graph

`threshold_graph(C, D, stock_names, threshold)` is the simplest of the three: an edge for every pair whose raw correlation exceeds `threshold` (see `networks.ARM_THRESHOLDS`), full stop — no spanning-tree or planarity backbone underneath it. This makes it the only one of the three graph types with **no connectivity guarantee**: at low average correlation, most pairs fall below threshold and the graph fragments into many small/singleton components. Measured directly on 629 SP500/132d windows: mean node coverage of the largest connected component is 43% (range 3%–98%), averaging ~158 separate components per window — see [`eigencentrality.md`](eigencentrality.md) for why this specifically breaks any analysis that needs a fixed-size, consistently-indexed feature vector per window. MST and TMFG below exist largely *because* the plain threshold graph doesn't guarantee connectivity.

## MST — Minimum Spanning Tree

`mst_graph(D, stock_names)` runs `scipy`'s `minimum_spanning_tree` on the full $N\times N$ distance matrix: the tree connecting all $N$ nodes with minimum total distance (equivalently, maximum total correlation in a greedy sense), using exactly $N-1$ edges and containing no cycles. It's the single strongest-connectivity "backbone" of the market — every node is reachable, but only through the most direct correlation paths.

Because a spanning tree only has $N-1$ edges, it necessarily omits many genuinely strong pairwise correlations that don't happen to lie on the greedy tree. `mst_threshold_graph(C, D, stock_names, threshold)` augments the MST backbone additively: keep every MST edge, then add any other pair whose *raw correlation* (not distance) exceeds `threshold` that isn't already in the tree. The result is always connected (MST guarantees that) with a variable number of extra edges on top.

## TMFG — Triangulated Maximally Filtered Graph

`tmfg_graph(C, D, stock_names)` implements the Massara/Aste/Di Matteo (2016) construction: a maximal planar graph with exactly $3N-6$ edges — richer than the MST's $N-1$ while still sparse and fast to build. Construction:

1. **Seed a tetrahedron.** Score every vertex by the sum of its above-mean edge weights, `v_score = (W * (W > W.mean())).sum(axis=1)`, and take the top 4 as the seed. Connect all $\binom{4}{2}=6$ pairs among them — this is the starting 4-clique, whose 4 triangular faces are the graph's initial "open faces."
2. **Grow greedily, face by face.** For each open triangular face `(a, b, c)`, the best candidate vertex to insert is whichever unplaced vertex maximizes the total weight gained by connecting to all three face vertices: `gain(v) = W[v,a] + W[v,b] + W[v,c]`. Insert the single best `(face, vertex)` pair across the *whole* graph, connect the new vertex to all 3 face vertices (3 new edges), then split that face into 3 new sub-faces (new vertex + any 2 of the original 3) for future rounds.
3. **Repeat** until all $N$ vertices are placed: $6$ seed edges $+\ 3\times(N-4)$ growth edges $= 3N-6$.

The implementation tracks candidate insertions in a **lazy max-priority queue** keyed on gain: each open face pushes its current best `(gain, vertex)` guess, but by the time an entry is popped its vertex may have already been placed by a different face's insertion — those stale entries are detected and recomputed on pop rather than eagerly updated on every insertion elsewhere in the queue, which is what keeps the construction efficient.

`tmfg_threshold_graph(C, D, stock_names, threshold)` augments the full TMFG the same way `mst_threshold_graph` augments the MST: keep every TMFG edge (all $3N-6$ of them), then add any other pair whose raw correlation exceeds `threshold` that isn't already in the graph. The result always contains the full TMFG backbone — so it's connected and its edge count floors at $3N-6$ — plus a variable number of extra above-threshold edges on top. That extra-edge count (and hence overall density) varies directly with market conditions — more pairs clear the threshold when average correlation is high — which is exactly the density time series plotted as a regime diagnostic in `data_processing.py`.

## A naming note

Every edge in all three graph types stores the **Mantegna distance** as its `weight` attribute (`weight=float(D[i, j])`), even the edges added by thresholding — but the threshold comparison itself is always against the *raw correlation* `C[i, j] > threshold`, not the distance. Don't read `weight` as "correlation strength"; low `weight` (low distance) means high correlation. This is also why every consumer of these graphs that wants an unweighted/binary adjacency (e.g. `eigencentrality.py`) must pass `weight=None` explicitly rather than trust the default — see [`eigencentrality.md`](eigencentrality.md).

**One threshold, not tuned per market or window length.** `networks.ARM_THRESHOLDS = {"raw": 0.65}` — 0.65 was derived in `threshold_diagnostics.ipynb` from a pooled-5%-exceedance target (see [`regime-detection.md`](regime-detection.md) for how it feeds `build_arm_networks`), scoped to **S&P 500 at T=132 only** — Nikkei 225, FTSE 350, CSI 300, and the other two epoch lengths (63d/378d) are never checked directly, even though it is then applied uniformly to all four markets and all three epoch lengths in `data_processing.py`. This is a genuine, still-open external-validity gap: a threshold tuned against one market's off-diagonal correlation distribution is assumed (not verified) to be a reasonable cutoff for three structurally different markets and two other window lengths. (The pipeline previously computed this same target separately for two other correlation transforms, `mg`/`gr`; both were dropped since the analysis only ever depended on the raw correlation matrix, and `ARM_THRESHOLDS` now carries a single entry.)
