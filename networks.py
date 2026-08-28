import os
import heapq
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree


# Mean of the off-diagonal (upper-triangle, k=1) entries of a correlation matrix.
def mu_offdiag(C):
    return float(C[np.triu_indices_from(C, k=1)].mean())


# Standard deviation of the off-diagonal (upper-triangle, k=1) entries of a correlation matrix.
def sigma_offdiag(C):
    return float(C[np.triu_indices_from(C, k=1)].std())


# Threshold graph, an edge for every pair whose correlation exceeds the cutoff.
def threshold_graph(C, D, stock_names, threshold):
    G = nx.Graph()
    G.add_nodes_from(stock_names)
    for i in range(len(stock_names)):
        for j in range(i + 1, len(stock_names)):
            if C[i, j] >= threshold:
                G.add_edge(stock_names[i], stock_names[j], weight=float(D[i, j]), edge_type="threshold")
    return G


# Minimum spanning tree of the distance matrix, as an nx.Graph with distance weights.
def mst_graph(D, stock_names):
    mst_coo = minimum_spanning_tree(D).tocoo()
    G = nx.Graph()
    G.add_nodes_from(stock_names)
    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        G.add_edge(stock_names[u], stock_names[v], weight=float(w), edge_type="mst")
    return G


# MST backbone augmented with every correlation edge above the threshold.
def mst_threshold_graph(C, D, stock_names, threshold):
    G = mst_graph(D, stock_names)
    for i in range(len(stock_names)):
        for j in range(i + 1, len(stock_names)):
            if C[i, j] >= threshold and not G.has_edge(stock_names[i], stock_names[j]):
                G.add_edge(stock_names[i], stock_names[j], weight=float(D[i, j]), edge_type="threshold")
    return G


# Triangulated Maximally Filtered Graph (Massara/Aste).
def tmfg_graph(C, D, stock_names):
    n = len(stock_names)
    W = C.copy()
    np.fill_diagonal(W, 0.0)

    v_score = (W * (W > W.mean())).sum(axis=1)
    seed = list(np.argsort(v_score)[::-1][:4])

    placed = np.zeros(n, dtype=bool)
    placed[seed] = True

    triangles = [
        [seed[0], seed[1], seed[2]],
        [seed[0], seed[1], seed[3]],
        [seed[0], seed[2], seed[3]],
        [seed[1], seed[2], seed[3]],
    ]

    G = nx.Graph()
    G.add_nodes_from(stock_names)

    def add_edge(a, b):
        if not G.has_edge(stock_names[a], stock_names[b]):
            G.add_edge(stock_names[a], stock_names[b], weight=float(D[a, b]), edge_type="tmfg")

    for a in range(4):
        for b in range(a + 1, 4):
            add_edge(seed[a], seed[b])

    def best_gain(tri):
        gains = W[:, tri[0]] + W[:, tri[1]] + W[:, tri[2]]
        gains = np.where(placed, -np.inf, gains)
        k = int(np.argmax(gains))
        return float(gains[k]), k

    pq = []
    for ti, tri in enumerate(triangles):
        gain, vertex = best_gain(tri)
        heapq.heappush(pq, (-gain, vertex, ti))

    inserted = 0
    while inserted < n - 4:
        _, new_vertex, triangle_index = heapq.heappop(pq)
        if placed[new_vertex]:
            gain, vertex = best_gain(triangles[triangle_index])
            heapq.heappush(pq, (-gain, vertex, triangle_index))
            continue

        triangle = triangles[triangle_index]
        for vertex in triangle:
            add_edge(new_vertex, vertex)
        placed[new_vertex] = True
        inserted += 1

        triangles[triangle_index] = [triangle[0], triangle[1], new_vertex]
        triangles.append([triangle[0], triangle[2], new_vertex])
        triangles.append([triangle[1], triangle[2], new_vertex])

        updated_triangles = (triangle_index, len(triangles) - 2, len(triangles) - 1)
        for ti in updated_triangles:
            gain, vertex = best_gain(triangles[ti])
            heapq.heappush(pq, (-gain, vertex, ti))

    return G

# TMFG backbone unioned with all above-threshold edges (additive; floors at 3(n-2)).
def tmfg_threshold_graph(C, D, stock_names, threshold):
    G = tmfg_graph(C, D, stock_names)
    n = len(stock_names)
    for a in range(n):
        for b in range(a + 1, n):
            if C[a, b] >= threshold and not G.has_edge(stock_names[a], stock_names[b]):
                G.add_edge(stock_names[a], stock_names[b], weight=float(D[a, b]), edge_type="threshold")
    return G


# Load cleaned prices (columns sorted by sector prefix) and return daily log returns.
def load_logreturns(path="data/SP500_AdjClose_Cleaned.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    stock_cols = sorted([c for c in df.columns if c != "Date"], key=lambda x: x.split("_")[0])
    df = df[["Date"] + stock_cols]

    df_logreturns = pd.DataFrame()
    df_logreturns["Date"] = df["Date"]
    for col in stock_cols:
        df_logreturns[col] = np.log(df[col]) - np.log(df[col].shift(1))
    return df_logreturns.dropna(how="any", axis=0).reset_index(drop=True)


# Rolling Pearson correlation matrices over the log returns; returns (matrices, end-dates, stock_names).
def rolling_correlations(df_logreturns, epoch_length, epoch_shift=22):
    stock_names = df_logreturns.columns[1:].tolist()
    correlation_matrices = []
    epoch_dates = []
    for i in range(0, len(df_logreturns) - epoch_length + 1, epoch_shift):
        epoch = df_logreturns.iloc[i:i + epoch_length]
        correlation_matrices.append(epoch.iloc[:, 1:].corr().values)
        epoch_dates.append(epoch["Date"].iloc[-1])
    return correlation_matrices, epoch_dates, stock_names


# Correlation threshold for network construction, derived from S&P 500/T=132 data only.
ARM_THRESHOLDS = {"raw": 0.65}


# Build one arm end-to-end: per window compute distance, mu, and threshold/MST/TMFG graphs.
def build_arm_networks(correlation_matrices, arm, stock_names, epoch_length, epoch_dates, threshold=None, index_code="SP500"):
    if threshold is None:
        threshold = ARM_THRESHOLDS[arm]

    correlation_matrices_arm = []
    distance_matrices = []
    mu_values = []
    threshold_graphs = []
    mst_threshold_graphs = []
    tmfg_threshold_graphs = []

    for C_raw in correlation_matrices:
        C = np.clip(C_raw, -1.0, 1.0)
        mu_values.append(mu_offdiag(C))
        D = np.sqrt(2 * (1 - C))

        G_threshold = threshold_graph(C, D, stock_names, threshold)
        G_mst = mst_threshold_graph(C, D, stock_names, threshold)
        G_tmfg = tmfg_threshold_graph(C, D, stock_names, threshold)

        correlation_matrices_arm.append(C)
        distance_matrices.append(D)
        threshold_graphs.append(G_threshold)
        mst_threshold_graphs.append(G_mst)
        tmfg_threshold_graphs.append(G_tmfg)

    return {
        "index_code": index_code,
        "arm": arm,
        "epoch_length": epoch_length,
        "threshold": threshold,
        "stock_names": stock_names,
        "epoch_dates": epoch_dates,
        "correlation_matrices": correlation_matrices_arm,
        "distance_matrices": distance_matrices,
        "mu_values": mu_values,
        "threshold_graphs": threshold_graphs,
        "mst_threshold_graphs": mst_threshold_graphs,
        "tmfg_threshold_graphs": tmfg_threshold_graphs,
    }


# Pickle one arm's network data to outputs/{index_code}_{arm}_network_data_{scale}d.pkl.
def save_arm(data, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{data['index_code']}_{data['arm']}_network_data_{data['epoch_length']}d.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


# Load one arm's pickled network data.
def load_arm(arm, epoch_length, outdir="outputs", index_code="SP500"):
    with open(f"{outdir}/{index_code}_{arm}_network_data_{epoch_length}d.pkl", "rb") as f:
        return pickle.load(f)