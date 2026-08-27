import os

import joblib
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from modelling import prepare_graph, apply_regime_order

# WL-prepare test graphs, infer their embeddings from a fitted Graph2Vec model, and report OOV rate -- shared by fit/load below.
def _graph2vec_infer_test(g2v, test_graphs):
    prepared_test = [prepare_graph(G) for G in test_graphs]
    return g2v.infer(prepared_test), g2v.oov_rate(prepared_test)


# Load a frozen raw-arm Graph2Vec bundle's train/test raw 128-d embeddings and OOV rates, without refitting.
def load_graph2vec_raw(tag, train_graphs, test_graphs, models_dir="outputs/models"):
    bundle = joblib.load(os.path.join(models_dir, f"{tag}.joblib"))
    g2v = bundle["g2v"]
    train_raw = g2v.get_embedding()
    train_oov = g2v.oov_rate([prepare_graph(G) for G in train_graphs])
    test_raw, test_oov = _graph2vec_infer_test(g2v, test_graphs)
    return bundle, train_raw, test_raw, train_oov, test_oov


# L2-normalise each row of the raw Graph2Vec embedding matrix so distances measure shape, not magnitude.
def prepare_graph2vec(raw):
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# Compute the n_windows x n_windows cosine distance matrix for a prepared vector matrix.
def distance_matrix(vectors):
    return pairwise_distances(vectors, metric="cosine")


# Return (min, median, max) off-diagonal distance from a square distance matrix.
def offdiag_distance_summary(D):
    mask = ~np.eye(D.shape[0], dtype=bool)
    values = D[mask]
    return float(values.min()), float(np.median(values)), float(values.max())


# Fit 2-D non-metric MDS on a precomputed distance matrix and return coordinates plus normalised Kruskal stress-1.
def fit_mds(D):
    model = MDS(n_components=2, metric=False, dissimilarity="precomputed",
                random_state=42, n_init=8, normalized_stress=True)
    coords = model.fit_transform(D)
    return coords, model.stress_


# Reproduce a raw-arm combo's frozen K-means regime labels; raw_test must be the bundle's own un-prepared vector space.
def raw_arm_regime_labels(bundle, raw_test, n_train_live):
    scaler, pca = bundle["scaler"], bundle["pca"]
    kmeans_model, kmeans_order = bundle["kmeans_model"], bundle["kmeans_order"]

    n_expected = scaler.n_features_in_
    if raw_test.shape[1] != n_expected:
        raise ValueError(
            f"raw_test has {raw_test.shape[1]} columns but the frozen bundle's scaler expects {n_expected} -- "
            "the bundle is stale relative to the current Graph2Vec implementation (e.g. its dimensions/"
            "wl_iterations hyperparameters changed since the bundle was written). Re-run modelling_sweep_raw.py "
            "to refresh it."
        )

    train_labels = kmeans_model.labels_
    if len(train_labels) != n_train_live:
        raise ValueError(
            f"frozen bundle's kmeans_model was fit on {len(train_labels)} train windows, but the live split has "
            f"{n_train_live} -- the bundle is stale relative to the current data (e.g. data_processing.py was "
            "re-run since the bundle was written). Re-run modelling_sweep_raw.py to refresh it."
        )

    X_test = pca.transform(scaler.transform(raw_test))
    test_labels = kmeans_model.predict(X_test)

    all_labels = np.concatenate([train_labels, test_labels])
    return apply_regime_order(all_labels, kmeans_order)
