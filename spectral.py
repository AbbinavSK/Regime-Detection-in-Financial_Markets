import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from networks import mu_offdiag, sigma_offdiag

# Descriptor columns fed to StandardScaler/PCA; n_zero_eigenvalues/q/mp_edge are reported diagnostics, not embedded.
SPECTRAL_FEATURE_COLUMNS = [
    "mu", "sigma_offdiag",
    "lambda_2_share", "lambda_3_share", "lambda_4_share",
    "n_above_mp", "spectral_entropy_bulk", "pr_corr_v1", "pr_corr_v2",
]


# Participation ratio of a vector (unit-L2-normalised first), mirroring eigencentrality's adjacency PR on a different object.
def participation_ratio(v):
    v = v / np.linalg.norm(v)
    return float(1.0 / (v ** 4).sum())


# Spectral/RMT diagnostics for one window's (already clipped to [-1,1]) correlation matrix.
def window_spectral(C, stock_names, epoch_length):
    N = len(stock_names)
    T = epoch_length
    q = T / N
    mp_edge = (1 + np.sqrt(N / T)) ** 2

    lambdas, V = np.linalg.eigh(C)
    order = np.argsort(lambdas)[::-1]
    lambdas, V = lambdas[order], V[:, order]

    n_nonzero = min(N, T - 1)
    n_zero_eigenvalues = N - n_nonzero
    lambdas_nonzero = lambdas[:n_nonzero]

    bulk = lambdas_nonzero[lambdas_nonzero < mp_edge]
    if bulk.size > 1:
        p = bulk / bulk.sum()
        log_p = np.zeros_like(p)
        np.log(p, where=p > 0, out=log_p)
        spectral_entropy_bulk = float(-(p * log_p).sum() / np.log(bulk.size))
    else:
        spectral_entropy_bulk = float("nan")

    return {
        "mu": mu_offdiag(C),
        "sigma_offdiag": sigma_offdiag(C),
        "lambda_2_share": float(lambdas[1] / N),
        "lambda_3_share": float(lambdas[2] / N),
        "lambda_4_share": float(lambdas[3] / N),
        "n_above_mp": int((lambdas > mp_edge).sum()),
        "spectral_entropy_bulk": spectral_entropy_bulk,
        "pr_corr_v1": participation_ratio(V[:, 0]),
        "pr_corr_v2": participation_ratio(V[:, 1]),
        "n_zero_eigenvalues": int(n_zero_eigenvalues),
        "q": float(q),
        "mp_edge": float(mp_edge),
    }


# Fit a StandardScaler and PCA(0.95) on the spectral feature matrix, mirroring eigencentrality.fit_centrality_embed.
def fit_spectral_embed(X_train_raw):
    scaler = StandardScaler().fit(X_train_raw)
    scaled = scaler.transform(X_train_raw)
    pca = PCA(n_components=0.95, random_state=42).fit(scaled)
    X_train = pca.transform(scaled)
    return X_train, scaler, pca, {"pca_dims": pca.n_components_}


# Transform a spectral feature matrix with an already-fitted scaler and PCA, mirroring eigencentrality.transform_centrality_embed.
def transform_spectral_embed(X_raw, scaler, pca):
    scaled = scaler.transform(X_raw)
    return pca.transform(scaled)