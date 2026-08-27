import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Largest connected component of G, with node order following stock_names.
def largest_connected_component(G, stock_names):
    lcc_nodes = max(nx.connected_components(G), key=len)
    node_order = [s for s in stock_names if s in lcc_nodes]
    return G.subgraph(node_order), node_order


# Leading eigenvector of a binary adjacency matrix, unit-L2 normalised, sign-flipped to positive if needed.
def leading_eigenvector(adjacency):
    _, eigenvectors = eigsh(adjacency, k=1, which="LA")
    v1 = eigenvectors[:, 0]
    if v1.sum() < 0:
        v1 = -v1
    return v1 / np.linalg.norm(v1)


# Eigenvector centrality and localisation diagnostics for one window's graph, computed on its LCC.
def window_centrality(G, stock_names):
    lcc_subgraph, node_order = largest_connected_component(G, stock_names)
    adjacency = nx.to_numpy_array(lcc_subgraph, nodelist=node_order, weight=None)
    v1 = leading_eigenvector(adjacency)
    p = v1 ** 2
    log_p = np.zeros_like(p)
    np.log(p, where=p > 0, out=log_p)
    plogp = p * log_p
    return {
        "v1": v1,
        "node_order": node_order,
        "pr": float(1.0 / (v1 ** 4).sum()),
        "max_comp": float(v1.max()),
        "entropy": float(-plogp.sum()),
        "argmax_id": node_order[int(np.argmax(v1))],
        "lcc_frac": len(node_order) / len(stock_names),
    }


# Fit a StandardScaler and PCA(0.95) on a raw-indexed v1 matrix, mirroring modelling.fit_embed.
def fit_centrality_embed(X_train_raw):
    scaler = StandardScaler().fit(X_train_raw)
    scaled = scaler.transform(X_train_raw)
    pca = PCA(n_components=0.95, random_state=42).fit(scaled)
    X_train = pca.transform(scaled)
    return X_train, scaler, pca, {"pca_dims": pca.n_components_}


# Transform a raw-indexed v1 matrix with an already-fitted scaler and PCA, mirroring modelling.transform_embed.
def transform_centrality_embed(X_raw, scaler, pca):
    scaled = scaler.transform(X_raw)
    return pca.transform(scaled)