import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
from graph2vec import Graph2Vec


# Load the VIX close series.
def load_vix(path="data/VIX_Close.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze().sort_index()


# Relabel a graph's nodes with integers and their degree as a feature.
def prepare_graph(G):
    G_int = nx.convert_node_labels_to_integers(G)
    for node in G_int.nodes():
        G_int.nodes[node]["feature"] = int(G_int.degree(node))
    return G_int


# Per-window keys of an arm dict that split_train_test partitions by epoch_dates.
PER_WINDOW_KEYS = ["epoch_dates", "correlation_matrices", "distance_matrices", "mu_values", "threshold_graphs", "mst_threshold_graphs", "tmfg_threshold_graphs"]


# Split an arm dict into train and test windows by date.
def split_train_test(data, cutoff="2019-12-31"):
    cutoff = pd.Timestamp(cutoff)
    is_train = pd.DatetimeIndex(pd.to_datetime(data["epoch_dates"])) <= cutoff
    train = {k: v for k, v in data.items() if k not in PER_WINDOW_KEYS}
    test = dict(train)
    for key in PER_WINDOW_KEYS:
        values = data[key]
        train[key] = [v for v, m in zip(values, is_train) if m]
        test[key] = [v for v, m in zip(values, is_train) if not m]
    return train, test


# Embed graphs by fitting Graph2Vec, a scaler, and PCA.
def fit_embed(graphs):
    prepared = [prepare_graph(G) for G in graphs]

    g2v = Graph2Vec(dimensions=128, wl_iterations=3, epochs=100, seed=42)
    g2v.fit(prepared)
    embeddings = g2v.get_embedding()

    scaler = StandardScaler().fit(embeddings)
    scaled = scaler.transform(embeddings)
    pca = PCA(n_components=0.95, random_state=42).fit(scaled)

    X = pca.transform(scaled)
    return X, g2v, scaler, pca, {"pca_dims": pca.n_components_}


# Embed graphs using an already-fitted Graph2Vec, scaler, and PCA.
def transform_embed(graphs, g2v, scaler, pca):
    prepared = [prepare_graph(G) for G in graphs]

    embeddings = g2v.infer(prepared)
    scaled = scaler.transform(embeddings)

    X = pca.transform(scaled)
    return X, {"oov_rate": g2v.oov_rate(prepared)}


# Cluster embeddings into regimes with K-means, fit on train only.
def cluster_kmeans(X_train, X_test, k=3):
    model = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X_train)
    return model.labels_, model.predict(X_test), model


# Decode each timestep from only observations up to and including it (Viterbi re-run on each growing prefix), so no label depends on later observations.
def _causal_decode(model, X):
    return np.array([model.predict(X[: t + 1])[-1] for t in range(len(X))])


# Cluster embeddings into regimes with a Gaussian HMM, fit on train only.
def cluster_hmm(X_train, X_test, k=3, n_seeds=20):
    best_model, best_ll = None, -np.inf
    for seed in range(n_seeds):
        m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=200, tol=1e-4, random_state=seed)
        try:
            m.fit(X_train)
            ll = m.score(X_train)
        except Exception:
            continue
        if ll > best_ll:
            best_model, best_ll = m, ll
    if best_model is None:
        raise RuntimeError(f"HMM fitting failed for all {n_seeds} seeds - X may be degenerate (e.g. a collapsed embedding)")
    train_labels = best_model.predict(X_train)
    test_labels = _causal_decode(best_model, X_test)
    return train_labels, test_labels, best_model


# Rank clusters by mean correlation to find the crisis regime.
def fit_regime_order(labels, mu, k=3):
    labels = np.asarray(labels)
    mu = np.asarray(mu)

    cluster_mean_mu = np.array([mu[labels == c].mean() for c in range(k)])
    order = np.argsort(cluster_mean_mu)
    rank_of_cluster = np.argsort(order)
    crisis_id = order[-1]

    return order, rank_of_cluster, crisis_id, cluster_mean_mu


# Relabel cluster IDs into rank order.
def apply_regime_order(labels, order):
    rank_of_cluster = np.argsort(order)
    return rank_of_cluster[np.asarray(labels)]


# Reorder the HMM's transition matrix by regime rank and compute dwell times.
def hmm_transitions(model, order):
    P = model.transmat_[np.ix_(order, order)]
    dwell = 1.0 / (1.0 - np.diag(P))
    return P, dwell


# Compute an empirical transition matrix from a label sequence.
def empirical_transitions(labels, k, order):
    rank_of_cluster = np.argsort(order)
    ranks = rank_of_cluster[np.asarray(labels)]

    occupancy = np.array([(ranks == r).sum() for r in range(k)])
    counts = np.zeros((k, k))
    for a, b in zip(ranks[:-1], ranks[1:]):
        counts[a, b] += 1

    row_sums = counts.sum(axis=1)
    P = np.full((k, k), np.nan)
    nonzero = row_sums > 0
    P[nonzero] = counts[nonzero] / row_sums[nonzero, None]
    return P, occupancy


# Compute precision, recall, and F1 for a binary prediction.
def binary_f1(pred, actual):
    pred, actual = np.asarray(pred), np.asarray(actual)
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# Score predicted crisis labels against a validation series like VIX.
def validate_regime(labels, dates, series, threshold, crisis_id):
    dates_idx = pd.DatetimeIndex(pd.to_datetime(dates))
    series_eod = series.reindex(series.index.union(dates_idx)).sort_index().ffill().reindex(dates_idx).values
    crisis_point = (series_eod > threshold).astype(int)
    pred = (np.asarray(labels) == crisis_id).astype(int)
    return {
        "series_eod": series_eod, "pred": pred,
        "crisis_point": crisis_point,
        "point": binary_f1(pred, crisis_point),
    }