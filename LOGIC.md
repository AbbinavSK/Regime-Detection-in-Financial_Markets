# LOGIC.md — The Reasoning Behind This Project

This document explains *why* the pipeline is built the way it is — not what each function does (the code is
short and readable on its own), but what each design choice is for, and how the pieces fit together as one
coherent research design rather than a collection of separate scripts. It assumes a stats/ML background
(PCA, eigenvectors, Hidden Markov Models) but no prior exposure to this codebase.

## 1. The Research Question

- **Surface question**: can market regimes (calm / transitional / crisis) be detected from the evolving
  *topology* of a stock correlation network, instead of from price statistics directly?
- **The sharper, falsifiable question underneath it**: when you build an expensive, *learned* representation
  of that topology (Graph2Vec) against a cheap, *deterministic* one (eigenvector centrality, or a
  spectral/RMT summary that skips network-building entirely), does the expensive one actually buy you
  anything?
- **Why this framing matters**: a pipeline that only ever ran Graph2Vec could never answer the second
  question, no matter how good its numbers looked — there would be nothing to compare it against. Every
  structural choice below (three descriptors instead of one, identical downstream scoring machinery, the
  same clustering/validation code reused everywhere) exists specifically to make this comparison fair and
  interpretable, not to make Graph2Vec look good.

## 2. The Experimental Design: A Controlled 2×2 Comparison

Rather than running one new method and reporting one number, the project varies two independent axes and
holds everything else fixed, so that any difference in outcome can be traced to one specific cause:

|  | Input: filtered graph | Input: raw correlation matrix |
|---|---|---|
| **Descriptor: learned** | Graph2Vec | *(not built — would need a graph-free learned method)* |
| **Descriptor: handcrafted** | Eigenvector centrality | Spectral / RMT |

- **Graph2Vec vs. eigenvector centrality** — holds the *input* fixed (both read the same graph) and varies
  the *descriptor*. This isolates: does a 128-dimensional learned embedding beat a single deterministic
  eigenvector, given the same graph to work from?
- **Eigenvector centrality vs. spectral/RMT** — holds the *descriptor style* fixed (both are handcrafted,
  both are "take an eigenvector of something") and varies the *input*. This isolates: how much does
  discarding everything except a sparse graph backbone actually cost? A full correlation matrix has
  roughly $N(N-1)/2$ pairwise entries; a minimum spanning tree keeps exactly $N-1$ edges — the overwhelming
  majority of pairwise structure is thrown away before eigenvector centrality ever runs.
- **Graph2Vec vs. spectral/RMT** — differs on *both* axes at once, so it is never reported as a standalone
  headline result; it is only interpretable through the two comparisons above, as the corner-to-corner
  check that ties them together.
- Eigenvector centrality sits at the pivot of both comparisons, which is exactly why it has to be the most
  defensible, least-hyperparameter method in the whole design (§6).

## 3. Foundational Design Choices

These choices are shared by every branch, so they only need to be justified once — and every downstream
comparison depends on them being *genuinely* shared, not quietly re-implemented per branch.

- **Rolling windows, three timescales.** Each window of trading-day log returns produces one Pearson
  correlation matrix; the window then slides forward and repeats. Three window lengths (63 / 132 / 378
  trading days ≈ one quarter / half-year / eighteen months) test three different answers to "how much
  history should define a regime." Windows overlap heavily by construction (consecutive 132-day windows
  share 122 days), which is cheap to compute but has a real statistical cost that resurfaces in the HMM
  discussion below.
- **Four markets, one validated.** S&P 500, Nikkei 225, FTSE 350, and CSI 300 are all put through every
  regime-detection branch, but only the S&P 500 has an independent stress signal (VIX) to validate against
  — see below. Results for the other three markets are internally-consistent structural comparisons, not
  externally validated detections, and that distinction is treated as load-bearing throughout, not a
  footnote.
- **Raw correlation only, deliberately.** Two other correlation transforms were tried early in the project
  — one clipping RMT noise-bulk eigenvalues, one subtracting out the dominant "market mode" — but neither
  was ever depended on by any regime-detection result, so both were dropped. Working from the raw,
  unmodified correlation matrix means no assumption about what counts as "signal" versus "noise" is baked
  in before the descriptor even sees the data.
- **The graph-construction threshold, and its limit.** Networks are filtered by a single correlation cutoff,
  derived once from a pooled-sparsity target on the S&P 500 at the 132-day window length only, then applied
  uniformly to all four markets and all three window lengths. This is a real, still-open external-validity
  gap: a cutoff tuned on one market's correlation distribution is *assumed*, not verified, to be reasonable
  for three structurally different markets.
- **The causal train/test split.** Every branch fits everything — descriptor, scaler, dimensionality
  reduction, clustering — only on windows ending on or before 2019-12-31, then strictly *applies* those
  frozen fits to later windows. This is not a generic best-practice reflex; it is load-bearing for the whole
  premise. If clustering had seen test-period correlation structure while fitting, "detects the 2020 crash"
  would partly just mean "was shown the 2020 crash," and any validation score would measure memorisation,
  not detection.
- **Two clustering algorithms, not one.** Every descriptor is clustered into three regimes twice: once with
  K-means (each window judged independently) and once with a Gaussian Hidden Markov Model (which also
  models regime *persistence* — the tendency to stay in the same state day-to-day). Running both is itself
  a robustness check: a finding that only holds under one clustering algorithm is a weaker finding than one
  that holds under both, and the two are compared explicitly rather than one being silently preferred.
- **Mean correlation as the regime-ranking key.** Every clustering method produces *arbitrary* cluster
  labels — cluster "0" has no inherent meaning and can flip between runs (the standard label-switching
  problem). This is resolved once, using train data only: whichever cluster has the highest mean
  off-diagonal correlation *is* defined as "Crisis," full stop, and that mapping is frozen and reused on
  test data. This single choice is what makes it meaningful to compare "Graph2Vec's Crisis regime" against
  "eigenvector centrality's Crisis regime" in the same sentence — without it, comparing detection accuracy
  across descriptors would just be comparing differently-defined labels that happen to share a colour in a
  plot.
- **VIX validation, and its limits.** The detected "Crisis" regime is scored against real VIX spikes
  (VIX > 30) for the S&P 500 only — no equivalent series exists for the other three markets. Every claim
  about cross-market generalisation therefore rests on structural comparison (does the *shape* of
  regime-transition behaviour look similar across markets), never on a non-S&P-500 market detecting a real,
  externally verified stress episode. This limitation recurs under every branch below and is never smoothed
  over.

## 4. Three Network Constructions — A Shared Floor, Not a Ladder

Each window's correlation matrix is filtered into a sparse graph three different ways:

- **Threshold** — keep only pairs whose correlation clears the cutoff. Simplest, but gives **no
  connectivity guarantee**: at low average correlation the graph fragments into many disconnected pieces
  (measured directly: an average of 43% of the stock universe in the largest connected component, with
  roughly 158 separate fragments in a typical window).
- **MST-threshold** — start from the minimum spanning tree (the single most efficient "backbone" connecting
  every node), then add back any pair clearing the same correlation cutoff. Always connected.
- **TMFG-threshold** — start from a Triangulated Maximally Filtered Graph (a denser, still-planar backbone),
  then add back the same above-threshold pairs. Always connected, and structurally richer than the MST.

**These three are not fully nested — it is important to be precise about exactly what is and isn't
guaranteed here.** Every threshold-graph edge survives into *both* the MST-threshold graph and the
TMFG-threshold graph, since both are built as "backbone plus every above-threshold edge" — that part is
nested by construction. But MST-threshold and TMFG-threshold are **not** nested with each other: a minimum
spanning tree and a TMFG are two different algorithms selecting two different backbones (an efficient
connecting tree vs. a planar triangulation), so the MST-threshold graph's edges are not a subset of the
TMFG-threshold graph's, or vice versa — checked directly, the two backbones disagree on the overwhelming
majority of windows. The correct picture is a shared floor with two separate, structurally different ways
of building on top of it, not a single ladder of increasingly permissive constructions. This is still
exactly what makes it meaningful to ask, later, whether a finding is stable across "which backbone the
graph-filtering step chose" — the comparison just isn't a simple more-structure-vs-less-structure ordering.

## 5. Branch 1 — Graph2Vec: Learning What a Graph's Shape Looks Like

- **The problem it solves.** A graph is a variable-size, unordered structure with no natural fixed-length
  vector representation, but clustering needs one fixed-length vector per window. Graph2Vec treats "embed a
  graph" as "embed a document": each node is relabelled by a hash of its local neighbourhood
  (Weisfeiler-Lehman relabelling), pooled into a bag of features, and fed to a Doc2Vec model.
- **Identity is deliberately erased.** Every node is anonymised to a plain integer before embedding — a
  choice to capture the network's *shape*, not the *identity* of which stocks are in it. Two
  structurally-identical networks built from completely different stock universes would embed near
  identically. This is the exact opposite choice eigenvector centrality makes (§6), and that contrast is
  the point of running both.
- **Fitting and inferring are not the same operation.** A training window's embedding falls directly out of
  the model once training converges. A test window's embedding is a *separate optimisation problem* run
  against the already-frozen, trained vocabulary — and any structural feature that never made the frequency
  cut during training is silently dropped, not approximated. This is not a small effect in practice: on
  real data, an average of roughly 48% of a training window's own structural features already miss the
  vocabulary cut, rising to around 51% at test time. This asymmetry does not invalidate comparisons against
  the other two descriptors, but it does mean Graph2Vec's held-out performance carries a real, structural
  handicap that the other two descriptors don't share (§8).
- **Reproducibility is only approximate.** Even with every explicit random seed fixed, the underlying
  training procedure is multi-threaded and only approximately reproducible — two runs of the identical
  configuration can differ by a single predicted-regime window, moving a reported accuracy score in the
  third decimal place. Any Graph2Vec number quoted to three decimals should be read as "this run," not as a
  fixed constant of the method.

## 6. Branch 2 — Eigenvector Centrality: The Classical, Zero-Hyperparameter Baseline

- **Why it exists.** Eigenvector centrality's definition is a fixed point, not an algorithm — a node's
  importance is proportional to the sum of its neighbours' importances. There is nothing to fit, seed, or
  tune, which is the entire point: it is the null hypothesis Graph2Vec's added complexity has to beat.
- **Why the plain threshold graph is excluded here.** Computing the leading eigenvector this way requires a
  connected graph; the plain threshold construction's fragmentation problem (§4) rules it out for this
  branch specifically, so it is only ever run on the MST-threshold and TMFG-threshold constructions.
- **A genuine, but overstated, interpretability edge.** Unlike Graph2Vec, this method keeps node identity
  throughout, so it can always name the single most-central stock in a window. In practice this is weaker
  than it sounds — across a market's full grid of configurations, the *modal* most-central stock still took
  several different values, not one dominant answer. Read this as "this descriptor can always name a
  plausible, currently-central stock," not "this descriptor has identified *the* systemically important
  one."
- **A known weak point.** The pipeline's localisation diagnostic (flagging when a window's leading
  eigenvector concentrates too heavily on one node) is based on the *mean* of a statistic across all
  windows in a configuration, which a handful of well-behaved windows can pull comfortably above the
  warning threshold even while the *typical* window stays badly localised. Passing this diagnostic should
  not be read as "this configuration is problem-free."

## 7. Branch 3 — Spectral / RMT: Reading the Correlation Matrix Directly

- **Why it exists.** Every graph-construction step is a *choice* about what correlation structure to keep
  and what to discard. This branch asks what happens if that choice is never made at all: eigendecompose
  the correlation matrix itself and read regime structure directly off its spectrum — the market-mode
  eigenvalue's level, how much is carried by the next few eigenvalues, a Random-Matrix-Theory count of
  how many eigenvalues exceed the noise floor, and how "spread out" the noise bulk itself is.
- **A subtle but important correction.** A sample correlation matrix built from $T$ observations over $N$
  assets has rank at most $\min(N, T-1)$, not $\min(N, T)$ — Pearson correlation demeans each column,
  costing one degree of freedom. At a 132-day window with a few hundred assets, well over half the
  eigenvalues are *structurally* zero, not just small, and every eigenvalue-counting step in this branch
  respects that distinction rather than relying on a magnitude tolerance that could quietly drift.
- **No learned component at all.** Eigendecomposition of a fixed matrix is exactly reproducible — same
  input, same output, on any machine, every time. This is the sharpest possible contrast with Graph2Vec,
  and even slightly sharper than eigenvector centrality (whose iterative solver is empirically, but not
  guaranteed, reproducible).

## 8. Why Comparing Across Branches Is Fair

- **The scoring machinery is identical, not merely similar.** Past the point where each branch produces its
  own descriptor matrix, every subsequent step — clustering, regime-ranking, validation — is the *same
  function call* for all three branches, with no branch-specific logic anywhere inside it. This is what
  turns "Graph2Vec's accuracy modestly beats eigenvector centrality's" from a vague impression into a
  controlled comparison: any difference is attributable to what went into the descriptor matrix, not to
  divergent scoring logic.
- **Fair scoring does not mean fair inputs.** Graph2Vec's fit-vs-infer asymmetry (§5) means its held-out
  descriptor carries materially more approximation noise than the other two branches'. This doesn't make
  the comparison meaningless — if Graph2Vec still wins on average *despite* that handicap, that is a mildly
  more interesting result than the bare number suggests. But it does mean a single combo where a
  handcrafted descriptor wins is not automatically evidence that the handcrafted descriptor is "genuinely
  better" — Graph2Vec was carrying a structural disadvantage in how it was tested, not just in what it is.
- **A single combo's score is a noisy estimate, not a verdict.** On one closely inspected case, Graph2Vec
  and eigenvector centrality *agreed* on 86% of test windows, yet their accuracy scores differed
  substantially — the entire gap traced back to a small number of disagreement windows, against a small
  number of true crisis windows in the test set. With that few positive examples, a handful of boundary
  misclassifications can swing an accuracy score by a large margin. This is exactly why the project reports
  a full grid of configurations rather than a single headline number.

## 9. A Fourth Question: Is Graph2Vec's Embedding Shape Stable?

Sections 5–8 are all "detect regimes, then score them" — one shared question, three descriptors, four
markets. A fourth, narrower branch asks something structurally different: not *whether* a descriptor
detects regimes, but whether Graph2Vec's embedding *space* keeps the same shape across the three network
constructions from §4 (which share a threshold floor but otherwise build on two structurally different
backbones — see §4 for exactly what is and isn't nested).

- **Why this comparison matters here.** An embedding shape that survives all three constructions is
  evidence that the regime-detection signal doesn't depend on which construction the graph-filtering step
  chose, while a shape that changes noticeably between them helps locate *where* the signal actually comes
  from.
- **Why this is narrower than the main comparison.** Only Graph2Vec is used, and only on the S&P 500 — this
  branch never needed eigenvector centrality's connectivity requirement, which is precisely what makes it
  possible to include the plain threshold construction here, unlike in §6. The distance metric (cosine) and
  the projection (2-D, non-metric multidimensional scaling) are both fixed, not swept, keeping the branch
  narrow enough to verify thoroughly rather than broad and unchecked.

## 10. The Financial Application: Does Detection Imply Profit?

Every result up to this point is about *statistical* detection — does the signal exist, and can it be
scored against an independent benchmark. This is a genuinely different question: if you actually traded on
the detected regimes, would you make money?

- **Design choice: one pre-nominated cell, not a search.** The backtest uses a single configuration decided
  in advance (S&P 500, the 132-day window, MST-threshold graphs, Graph2Vec, K-means only), rather than
  sweeping many configurations and reporting whichever looks best. Reporting the best of many backtests
  would be reporting noise, not a finding.
- **Why K-means and not the HMM here.** The HMM's causal decoding re-runs on every growing prefix of the
  test data, which is appropriate for retrospective validation but not directly usable as a live daily
  exposure signal without further adaptation — so it is excluded here on principle, not convenience.
- **Why a volatility-triggered control exists.** Alongside buy-and-hold, the backtest includes a much
  simpler rule (reduce exposure when trailing realised volatility crosses a threshold learned from training
  data only) specifically so that any apparent edge from the whole network-based pipeline can be checked
  against what a single cheap scalar signal achieves on its own.
- **Why no significance test is computed.** The effective sample size here is a handful of distinct crisis
  episodes, not the number of trading days — a p-value computed on daily returns would misrepresent how
  much evidence is actually present. Dispersion is instead estimated with a block bootstrap sized to match
  the correlation window length.
- **The verdict is stated plainly, not softened.** Against a criterion fixed before any number was
  computed, the regime-timed strategy does not beat both a risk-adjusted-return benchmark and a
  drawdown benchmark simultaneously — detecting a regime and profiting from trading it are shown here to be
  two different claims, and the project treats that as a real finding, not a disappointing footnote to
  qualify away.

## 11. What Was Tried and Not Pursued

- **Two other correlation transforms** (RMT-denoised and market-mode-removed) were implemented and swept
  early in the project but never adopted by any regime-detection result, and were removed once the project
  settled on raw correlation only (§3).
- **A fourth, learned-and-graph-free descriptor** would complete the 2×2 table in §2, but no such method
  was built — doing so was out of scope for this project rather than attempted and abandoned.
- **A Variational Graph Auto-Encoder** was explored as a candidate learned embedding method but was shelved:
  it collapsed to a near-identical embedding across every window when applied to the TMFG-threshold
  construction specifically, and was never validated as reliable enough to promote to a full branch. This
  is why there is no second learned-descriptor branch alongside Graph2Vec.

## 12. How the Pieces Fit Together

The project is not four unrelated scripts; it is one research design executed four times against a shared
foundation. Sections 3–4 establish the substrate every branch depends on. Sections 5–7 vary exactly two
axes (learned-vs-handcrafted, graph-vs-matrix input) so that Section 8's cross-branch comparisons are
attributable to a specific cause rather than an unexplained gap. Section 9 asks a narrower structural
question using only the strongest branch. Section 10 steps outside statistical detection entirely and asks
whether any of it is actually worth money. Read together, the project's honest answer to its own question
(§1) is: correlation-network topology does carry a real, multiply-corroborated regime signal — but no
descriptor dominates cleanly across every condition, and detecting a regime statistically is demonstrably
not the same as being able to trade it profitably. That ambiguity, stated plainly rather than resolved
artificially, is treated throughout as the actual finding.
